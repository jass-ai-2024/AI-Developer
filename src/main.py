import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from src.init_tools import tools
from src.logger import LOGGER
from src.file_monitor import check_architecture_file

load_dotenv(override=True)

LOGGER.info("BEGIN")

# Глобальные переменные
CURRENT_VERSION: Optional[str] = None
PROJECT_PATH: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global db_connection, agent, CURRENT_VERSION, PROJECT_PATH
    
    # Получаем путь к проекту из переменной окружения
    PROJECT_PATH = os.getenv("PROJECT_PATH", "./test")
    LOGGER.info(f"Project path: {PROJECT_PATH}")
    
    # Ждем появления файла архитектуры
    CURRENT_VERSION = await wait_for_architecture_file()
    if not CURRENT_VERSION:
        LOGGER.error("Failed to get architecture version")
        raise Exception("Failed to get architecture version")
    
    LOGGER.info(f"Architecture version: {CURRENT_VERSION}")

    # Инициализируем PostgreSQL
    DB_URI = f"postgresql://{os.getenv('PERSISTENCE_PG_USER')}:{os.getenv('PERSISTENCE_PG_PASSWORD')}@{os.getenv('PERSISTENCE_PG_CONTAINER')}:{os.getenv('PERSISTENCE_PG_PORT')}/{os.getenv('PERSISTENCE_PG_DB')}?sslmode=disable"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    db_connection = ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )
    checkpointer = PostgresSaver(db_connection)
    checkpointer.setup()

    # Инициализируем агента
    agent_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        streaming=False,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = os.getenv("agent_prefix_prompt")
    system_message = SystemMessage(content=prompt)

    agent = create_react_agent(
        agent_llm, 
        tools=tools, 
        state_modifier=system_message,
        checkpointer=checkpointer
    )

    yield

    # Создаем файл завершения с текущей версией
    await create_done_file()
    
    # Закрываем соединение с БД
    if db_connection:
        db_connection.close()

async def wait_for_architecture_file() -> Optional[str]:
    """Ожидает появления файла архитектуры"""
    try:
        version = check_architecture_file(PROJECT_PATH, interval=5, timeout=60)
        return version
    except Exception as e:
        LOGGER.error(f"Error waiting for architecture file: {e}")
        return None

async def create_done_file():
    """Создает файл code_done_vX.txt по завершении работы"""
    if CURRENT_VERSION:
        try:
            done_path = os.path.join(PROJECT_PATH, f"code_done_{CURRENT_VERSION}.txt")
            with open(done_path, "w") as f:
                f.write(f"version: {CURRENT_VERSION}\n")
                f.write("status: completed\n")
            LOGGER.info(f"Created completion file: {done_path}")
        except Exception as e:
            LOGGER.error(f"Error creating done file: {e}")

app = FastAPI(lifespan=lifespan)

cache_map = {}


class Query(BaseModel):
    task_id: str
    task_description: dict
    task_type: str


db_connection_pool = None
agent = None


def composite_key_builder(func, *args, **kwargs):
    query_text = kwargs.get("query").text
    conversation_id = kwargs.get("query").conversation_id
    key = f"{func.__name__}:{conversation_id}:{query_text}"
    LOGGER.info(f"Generated cache key: {key}")
    return key


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

    return {
        "input": input_value,
        "output": output_value,
        "intermediate_steps": intermediate_steps,
    }

@app.post("/chat")
async def chat(query: Query = Body(...)):
    try:
        return await asyncio.wait_for(_execute_chat_logic(query), timeout=150)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

async def _execute_chat_logic(query: Query):
    try:
        global agent
        LOGGER.info(f"Task ID: {query.task_id}, Task Type: {query.task_type}")
        LOGGER.info(f"Task Description: {query.task_description}")

        config = {"configurable": {"thread_id": query.task_id}}
        input_message = HumanMessage(content=str(query.task_description))
        response = agent.invoke({"messages": [input_message]}, config)

        response = transform_response_format(dict(response))
        LOGGER.info(f"Response: {response}")

        # Очищаем старые сообщения
        messages = agent.get_state(config).values["messages"]
        
        # Удаляем сообщения инструментов
        tools_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        ai_tool_calls = [
            message
            for message in messages
            if type(message) is AIMessage
            and len(getattr(message, "tool_calls", [])) > 0
        ]
        delete_messages = tools_messages + ai_tool_calls
        
        for message in delete_messages:
            agent.update_state(config, {"messages": RemoveMessage(id=message.id)})

        # Оставляем только последние 10 сообщений
        messages = agent.get_state(config).values["messages"]
        early_messages = messages[:-10]
        for message in early_messages:
            agent.update_state(config, {"messages": RemoveMessage(id=message.id)})

        return response
    except Exception as e:
        LOGGER.error(f"Error in _execute_chat_logic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.get("/health")
async def health():
    """
    Эндпоинт для проверки работоспособности сервера.
    
    Returns:
        str: "OK" если сервер работает
        HTTPException: 504 если проверка заняла больше 10 секунд
    """
    try:
        # Ждем выполнения _execute_health_logic() максимум 10 секунд
        return await asyncio.wait_for(_execute_health_logic(), timeout=10)
    except asyncio.TimeoutError:
        # Если проверка заняла больше 10 секунд, возвращаем ошибку
        raise HTTPException(status_code=504, detail="Request timed out")


async def _execute_health_logic():
    """
    Логика проверки работоспособности.
    Здесь можно добавить проверки:
    - Соединения с базой данных
    - Доступности OpenAI API
    - Состояния агента
    """
    try:
        # Проверяем соединение с PostgreSQL
        if db_connection:
            async with db_connection.connection() as conn:
                await conn.execute("SELECT 1")
        
        # Проверяем наличие OpenAI API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception("OpenAI API key not found")
            
        # Проверяем инициализацию агента
        if not agent:
            raise Exception("Agent not initialized")
            
        return "OK"
    except Exception as e:
        LOGGER.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер на порту 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)