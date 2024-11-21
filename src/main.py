import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.init_tools import tools
from src.logger import LOGGER
from src.file_monitor import check_architecture_file, create_empty_project_structure
from src.models import TaskQuery, Query, CodeGenerationResponse

load_dotenv(override=True)

LOGGER.info("BEGIN")

class ProjectPipeline:
    def __init__(self):
        self.architecture_version: Optional[str] = None
        self.arch_dir: str = os.getenv("ARCH_PATH", "arch")
        self.project_dir: str = os.getenv("PROJECT_PATH", "test")
        self.db_connection = None
        self.task_agent = None
        self.code_agent = None
        self.review_agent = None
        self.task_observer = None

    async def initialize_database(self) -> bool:
        """Инициализация базы данных PostgreSQL"""
        try:
            DB_URI = f"postgresql://{os.getenv('PERSISTENCE_PG_USER')}:{os.getenv('PERSISTENCE_PG_PASSWORD')}@{os.getenv('PERSISTENCE_PG_CONTAINER')}:{os.getenv('PERSISTENCE_PG_PORT')}/{os.getenv('PERSISTENCE_PG_DB')}?sslmode=disable"
            
            self.db_connection = ConnectionPool(
                conninfo=DB_URI,
                max_size=20,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            
            checkpointer = PostgresSaver(self.db_connection)
            checkpointer.setup()
            LOGGER.info("Database initialized successfully")
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing database: {e}")
            return False

    async def check_architecture(self) -> bool:
        """Шаг 1: Проверка наличия файла архитектуры в цикле"""
        try:
            LOGGER.info(f"Monitoring architecture file in directory: {self.arch_dir}")
            max_attempts = 60
            for attempt in range(max_attempts):
                self.architecture_version = check_architecture_file(self.arch_dir, timeout=1)
                if self.architecture_version:
                    LOGGER.info(f"Found architecture file version: {self.architecture_version}")
                    return True
                await asyncio.sleep(1)
            return False
        except Exception as e:
            LOGGER.error(f"Error checking architecture: {e}")
            return False

    async def create_project_structure(self) -> bool:
        """Шаг 2: Создание структуры проекта"""
        try:
            LOGGER.info(f"Creating project structure in: {self.project_dir}")
            return create_empty_project_structure(self.architecture_version, self.project_dir)
        except Exception as e:
            LOGGER.error(f"Error creating project structure: {e}")
            return False

    async def initialize_task_agent(self) -> bool:
        """Шаг 3: Инициализация агента для разбиения задач"""
        try:
            agent_llm = ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini",
                streaming=False,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            prompt = os.getenv("task_agent_prompt", "You are a task splitting agent...")
            system_message = SystemMessage(content=prompt)
            
            checkpointer = PostgresSaver(self.db_connection)
            self.task_agent = create_react_agent(
                agent_llm,
                tools=tools,
                state_modifier=system_message,
                checkpointer=checkpointer
            )
            LOGGER.info("Task agent initialized successfully")
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing task agent: {e}")
            return False

    async def split_tasks(self) -> bool:
        """Выполнение разбиения задач"""
        try:
            LOGGER.info("Splitting tasks into subtasks")
            
            arch_file = Path(self.arch_dir) / f"arch_services_{self.architecture_version}.json"
            if not arch_file.exists():
                LOGGER.error(f"Architecture file not found: {arch_file}")
                return False
                
            with open(arch_file, 'r') as f:
                architecture = json.load(f)
            
            task_description = {
                "task": "split_architecture",
                "architecture": architecture,
                "output_format": {
                    "subtasks": [
                        {
                            "task_description": "string",
                            "file_path": "string",
                            "technology": "string",
                            "requirements": ["string"],
                            "acceptance_criteria": ["string"],
                            "dependencies": ["string"],
                            "implementations": ["string"],
                            "integration_points": {}
                        }
                    ]
                }
            }
            
            config = {"configurable": {"thread_id": "split_arch"}}
            input_message = HumanMessage(content=str(task_description))
            response = self.task_agent.invoke({"messages": [input_message]}, config)
            
            result = transform_response_format(dict(response))
            
            tasks_file = Path(self.project_dir) / "final_tasks.json"
            with open(tasks_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            LOGGER.info(f"Tasks split and saved to: {tasks_file}")
            return True
            
        except Exception as e:
            LOGGER.error(f"Error splitting tasks: {e}")
            return False

    async def review_code(self) -> bool:
        """Выполнение ревью кода"""
        try:
            LOGGER.info("Reviewing generated code")
            
            tasks_file = Path(self.project_dir) / "final_tasks.json"
            if not tasks_file.exists():
                LOGGER.error("final_tasks.json not found")
                return False

            with open(tasks_file, 'r') as f:
                tasks = json.load(f).get('subtasks', [])
                
            review_results = []
            for task in tasks:
                file_path = task.get('file_path')
                if not file_path:
                    continue
                    
                full_path = Path(self.project_dir) / file_path
                if not full_path.exists():
                    continue
                    
                with open(full_path, 'r') as f:
                    code_content = f.read()
                    
                query = ReviewQuery(
                    task_id=f"review_{file_path}",
                    file_path=file_path,
                    code_content=code_content,
                    requirements=task.get("requirements", []),
                    acceptance_criteria=task.get("acceptance_criteria", [])
                )
                
                review_result = await self.execute_review(query)
                review_results.append({
                    "file_path": file_path,
                    "review_result": review_result
                })
                
            review_file = Path(self.project_dir) / f"code_review_v{self.architecture_version}.json"
            with open(review_file, 'w') as f:
                json.dump({"reviews": review_results}, f, indent=2)
                
            return True
        except Exception as e:
            LOGGER.error(f"Error reviewing code: {e}")
            return False

    async def create_success_file(self) -> bool:
        """Шаг 7: Создание файла успешного завершения"""
        try:
            success_file = Path(self.project_dir) / f"project_success_v{self.architecture_version}.txt"
            with open(success_file, "w") as f:
                f.write(f"version: {self.architecture_version}\n")
                f.write("status: completed\n")
            LOGGER.info(f"Created success file: {success_file}")
            return True
        except Exception as e:
            LOGGER.error(f"Error creating success file: {e}")
            return False

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.task_observer:
                self.task_observer.stop()
                self.task_observer.join()
            
            if self.db_connection:
                self.db_connection.close()
                
            LOGGER.info("Cleanup completed successfully")
        except Exception as e:
            LOGGER.error(f"Error during cleanup: {e}")

    async def start_task_monitoring(self) -> bool:
        """Запуск мониторинга задач для генерации кода"""
        try:
            class TaskFileHandler(FileSystemEventHandler):
                def __init__(self, pipeline):
                    super().__init__()
                    self.pipeline = pipeline
                    self.is_processing = False
                
                def on_created(self, event):
                    if event.src_path.endswith('final_tasks.json') and not self.is_processing:
                        self.is_processing = True
                        LOGGER.info(f"Detected new tasks file: {event.src_path}")
                        asyncio.run(self.process_tasks(event.src_path))
                        self.is_processing = False
                
                async def read_tasks(self, file_path: str) -> List[Dict]:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        return data.get('subtasks', [])
                    except Exception as e:
                        LOGGER.error(f"Error reading tasks file: {e}")
                        return []

                async def process_tasks(self, file_path: str):
                    tasks = await self.read_tasks(file_path)
                    if not tasks:
                        LOGGER.error("No tasks found to process")
                        return
                    
                    LOGGER.info(f"Found {len(tasks)} tasks to process")
                    
                    for index, task in enumerate(tasks, 1):
                        try:
                            task_description = {
                                "description": task.get("task_description"),
                                "file_path": task.get("file_path"),
                                "technology": task.get("technology"),
                                "requirements": task.get("requirements", []),
                                "acceptance_criteria": task.get("acceptance_criteria", []),
                                "dependencies": task.get("dependencies", []),
                                "implementations": task.get("implementations", []),
                                "integration_points": task.get("integration_points", {})
                            }
                            
                            query = Query(
                                task_id=f"task_{index}",
                                task_description=task_description,
                                task_type="implementation"
                            )
                            
                            LOGGER.info(f"Processing task {index}/{len(tasks)}: {task['task_description']}")
                            # Используем code_agent вместо task_agent
                            response = await self._execute_code_generation(query)
                            LOGGER.info(f"Task {index} completed. Response: {response}")
                            
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            LOGGER.error(f"Error processing task {index}: {e}")
                            continue

                async def _execute_code_generation(self, query: Query):
                    """Выполнение генерации кода с помощью code_agent"""
                    try:
                        LOGGER.info(f"Task ID: {query.task_id}, Task Type: {query.task_type}")
                        
                        task_description = query.task_description
                        LOGGER.info(f"Task Description: {task_description}")

                        config = {"configurable": {"thread_id": query.task_id}}
                        input_message = HumanMessage(content=str(task_description))
                        response = self.pipeline.code_agent.invoke({"messages": [input_message]}, config)

                        response = transform_response_format(dict(response))
                        LOGGER.info(f"Response: {response}")

                        # Очистка сообщений
                        messages = self.pipeline.code_agent.get_state(config).values["messages"]
                        tools_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
                        ai_tool_calls = [
                            msg for msg in messages 
                            if isinstance(msg, AIMessage) and len(getattr(msg, "tool_calls", [])) > 0
                        ]

                        for msg in tools_messages + ai_tool_calls:
                            self.pipeline.code_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})

                        # Удаление старых сообщений
                        messages = self.pipeline.code_agent.get_state(config).values["messages"]
                        for msg in messages[:-10]:
                            self.pipeline.code_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})

                        return response
                    except Exception as e:
                        LOGGER.error(f"Error in code generation: {str(e)}")
                        raise

            path = Path(self.project_dir)
            event_handler = TaskFileHandler(self)
            self.task_observer = Observer()
            self.task_observer.schedule(event_handler, path, recursive=False)
            self.task_observer.start()
            
            LOGGER.info(f"Started monitoring directory: {path}")
            return True
        except Exception as e:
            LOGGER.error(f"Error starting task monitoring: {e}")
            return False

    async def execute_review(self, query: ReviewQuery):
        """Выполнение ревью кода"""
        try:
            LOGGER.info(f"Reviewing file: {query.file_path}")
            
            review_description = {
                "file_path": query.file_path,
                "code": query.code_content,
                "requirements": query.requirements,
                "acceptance_criteria": query.acceptance_criteria
            }

            config = {"configurable": {"thread_id": query.task_id}}
            input_message = HumanMessage(content=str(review_description))
            response = self.review_agent.invoke({"messages": [input_message]}, config)

            response = transform_response_format(dict(response))
            LOGGER.info(f"Review response: {response}")

            # Очистка сообщений
            messages = self.review_agent.get_state(config).values["messages"]
            tools_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            ai_tool_calls = [
                msg for msg in messages 
                if isinstance(msg, AIMessage) and len(getattr(msg, "tool_calls", [])) > 0
            ]

            for msg in tools_messages + ai_tool_calls:
                self.review_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})

            return response
        except Exception as e:
            LOGGER.error(f"Error in code review: {str(e)}")
            raise

    async def initialize_code_agent(self) -> bool:
        """Инициализация агента для генерации кода"""
        try:
            agent_llm = ChatOpenAI(
                temperature=0,
                model="gpt-4-0613",
                streaming=False,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            prompt = os.getenv("code_agent_prompt", "You are a code generation agent...")
            system_message = SystemMessage(content=prompt)
            
            checkpointer = PostgresSaver(self.db_connection)
            self.code_agent = create_react_agent(
                agent_llm,
                tools=tools,
                state_modifier=system_message,
                checkpointer=checkpointer
            )
            LOGGER.info("Code agent initialized successfully")
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing code agent: {e}")
            return False

    async def initialize_review_agent(self) -> bool:
        """Инициализация агента для ревью кода"""
        try:
            agent_llm = ChatOpenAI(
                temperature=0,
                model="gpt-4-0613",
                streaming=False,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            prompt = os.getenv("review_agent_prompt", "You are a code review agent...")
            system_message = SystemMessage(content=prompt)
            
            checkpointer = PostgresSaver(self.db_connection)
            self.review_agent = create_react_agent(
                agent_llm,
                tools=tools,
                state_modifier=system_message,
                checkpointer=checkpointer
            )
            LOGGER.info("Review agent initialized successfully")
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing review agent: {e}")
            return False

    async def wait_for_code_generation(self) -> bool:
        """Ожидание завершения генерации кода"""
        try:
            max_wait_time = 3600  # 1 час
            check_interval = 5  # 5 секунд
            
            while max_wait_time > 0:
                if await self.check_generation_complete():
                    return True
                await asyncio.sleep(check_interval)
                max_wait_time -= check_interval
            return False
        except Exception as e:
            LOGGER.error(f"Error waiting for code generation: {e}")
            return False

    async def check_generation_complete(self) -> bool:
        """Проверка завершения генерации кода"""
        try:
            tasks_file = Path(self.project_dir) / "final_tasks.json"
            if not tasks_file.exists():
                return False
                
            with open(tasks_file, 'r') as f:
                tasks = json.load(f).get('subtasks', [])
                
            for task in tasks:
                file_path = task.get('file_path')
                if not file_path:
                    continue
                    
                full_path = Path(self.project_dir) / file_path
                if not full_path.exists():
                    return False
                    
            return True
        except Exception as e:
            LOGGER.error(f"Error checking generation status: {e}")
            return False

    async def execute_code_generation(self, query: TaskQuery) -> dict:
        """Выполнение генерации кода"""
        try:
            LOGGER.info(f"Generating code for task: {query.task_id}")
            
            config = {"configurable": {"thread_id": query.task_id}}
            input_message = HumanMessage(content=str(query.task_description))
            response = self.code_agent.invoke({"messages": [input_message]}, config)
            
            response = transform_response_format(dict(response))
            LOGGER.info(f"Code generation response: {response}")
            
            # Очистка сообщений
            messages = self.code_agent.get_state(config).values["messages"]
            tools_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            ai_tool_calls = [
                msg for msg in messages 
                if isinstance(msg, AIMessage) and len(getattr(msg, "tool_calls", [])) > 0
            ]
            
            for msg in tools_messages + ai_tool_calls:
                self.code_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})
                
            return response
        except Exception as e:
            LOGGER.error(f"Error in code generation: {str(e)}")
            raise

pipeline = ProjectPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        # Инициализация базы данных
        if not await pipeline.initialize_database():
            raise Exception("Failed to initialize database")

        # 1. Проверка файла архитектуры
        if not await pipeline.check_architecture():
            raise Exception("Architecture file not found")

        # 2. Создание структуры проекта
        if not await pipeline.create_project_structure():
            raise Exception("Failed to create project structure")

        # 3. Разбиение задач
        if not await pipeline.initialize_task_agent():
            raise Exception("Failed to initialize task agent")
        if not await pipeline.split_tasks():
            raise Exception("Failed to split tasks")

        # 4. Инициализация и запуск генерации кода
        if not await pipeline.initialize_code_agent():
            raise Exception("Failed to initialize code agent")
        if not await pipeline.start_task_monitoring():
            raise Exception("Failed to start task monitoring")

        # Ждем завершения генерации кода
        await pipeline.wait_for_code_generation()

        # 5. Ревью кода
        if not await pipeline.initialize_review_agent():
            raise Exception("Failed to initialize review agent")
        if not await pipeline.review_code():
            raise Exception("Failed to review code")

        # 6. Создание файла успешного завершения
        if not await pipeline.create_success_file():
            raise Exception("Failed to create success file")

        yield

    finally:
        pipeline.cleanup()

app = FastAPI(lifespan=lifespan)

def transform_response_format(json_new_format):
    messages_list = json_new_format["messages"]
    actual_messages = messages_list
    for i in range(len(messages_list) - 1, -1, -1):
        if isinstance(messages_list[i], HumanMessage):
            actual_messages = messages_list[i:]
            break

    LOGGER.info(f"ACTUAL MESSAGES: {actual_messages}")

    input_value = actual_messages[0].content
    output_value = actual_messages[-1].content

    intermediate_steps = []
    for message in actual_messages:
        if (
            isinstance(message, AIMessage)
            and len(getattr(message, "tool_calls", [])) > 0
        ):
            for tool_call in message.tool_calls:
                tool_output = None
                for next_message in actual_messages:
                    if isinstance(next_message, ToolMessage):
                        if next_message.tool_call_id == tool_call["id"]:
                            tool_output = next_message.content
                            break

                tool_input = tool_call["args"] if isinstance(tool_call["args"], str) else json.dumps(tool_call["args"])
                
                step = {
                    "type": "AgentAction",
                    "thought": message.content,
                    "tool": tool_call["name"],
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                }
                intermediate_steps.append(step)

    json_old_format = {
        "input": input_value,
        "output": output_value,
        "intermediate_steps": intermediate_steps,
    }

    return json_old_format

@app.post("/chat")
async def chat(query: Query = Body(...)):
    try:
        return await asyncio.wait_for(_execute_chat_logic(query), timeout=150)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

async def _execute_chat_logic(query: Query):
    try:
        LOGGER.info(f"Task ID: {query.task_id}, Task Type: {query.task_type}")
        
        task_description = query.task_description
        LOGGER.info(f"Task Description: {task_description}")

        config = {"configurable": {"thread_id": query.task_id}}
        input_message = HumanMessage(content=str(task_description))
        response = pipeline.task_agent.invoke({"messages": [input_message]}, config)

        response = transform_response_format(dict(response))
        LOGGER.info(f"Response: {response}")

        # Очистка сообщений
        messages = pipeline.task_agent.get_state(config).values["messages"]
        tools_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        ai_tool_calls = [
            msg for msg in messages 
            if isinstance(msg, AIMessage) and len(getattr(msg, "tool_calls", [])) > 0
        ]

        for msg in tools_messages + ai_tool_calls:
            pipeline.task_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})

        # Удаление старых сообщений
        messages = pipeline.task_agent.get_state(config).values["messages"]
        for msg in messages[:-10]:
            pipeline.task_agent.update_state(config, {"messages": RemoveMessage(id=msg.id)})

        return response
    except Exception as e:
        LOGGER.error(f"Error in _execute_chat_logic: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})

@app.get("/health")
async def health():
    return "OK"

@app.get("/version")
async def get_version():
    return {"version": pipeline.architecture_version}

@app.get("/status")
async def get_status():
    return {
        "architecture_version": pipeline.architecture_version,
        "arch_dir": pipeline.arch_dir,
        "project_dir": pipeline.project_dir,
        "agent_initialized": pipeline.task_agent is not None,
        "db_connected": pipeline.db_connection is not None
    }

import signal
import sys

def signal_handler(signum, frame):
    LOGGER.info(f"Received signal {signum}")
    if pipeline:
        pipeline.cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)