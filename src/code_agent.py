import asyncio
import json
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, List
from pydantic import BaseModel

from src.logger import LOGGER
from src.main import ProjectPipeline
from src.models import TaskQuery, CodeGenerationResponse

class TaskQuery(BaseModel):
    task_id: str
    task_description: dict
    task_type: str

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
        try:
            tasks = await self.read_tasks(file_path)
            if not tasks:
                LOGGER.error("No tasks found to process")
                return False
            
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
                    
                    query = TaskQuery(
                        task_id=f"task_{index}",
                        task_description=task_description,
                        task_type="implementation"
                    )
                    
                    LOGGER.info(f"Processing task {index}/{len(tasks)}: {task['task_description']}")
                    # Используем code_agent вместо task_agent
                    response = await self.pipeline.execute_code_generation(query)
                    LOGGER.info(f"Task {index} completed. Response: {response}")
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    LOGGER.error(f"Error processing task {index}: {e}")
                    continue
            
            # Создаем маркер завершения
            marker_file = Path(self.pipeline.project_dir) / f"code_generated_v{self.pipeline.architecture_version}.txt"
            marker_file.touch()
            
            LOGGER.info("Code generation completed successfully")
            return True
            
        except Exception as e:
            LOGGER.error(f"Error in code generation process: {e}")
            return False

async def run_code_agent():
    pipeline = ProjectPipeline()
    try:
        if not await pipeline.initialize_database():
            raise Exception("Failed to initialize database")
            
        if not await pipeline.initialize_code_agent():
            raise Exception("Failed to initialize code agent")
            
        # Запускаем мониторинг файла final_tasks.json
        path = Path('/app/test_project')
        event_handler = TaskFileHandler(pipeline)
        observer = Observer()
        observer.schedule(event_handler, path, recursive=False)
        observer.start()
        
        LOGGER.info(f"Started monitoring directory: {path}")
        
        # Бесконечный цикл для мониторинга
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        LOGGER.error(f"Code agent error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(run_code_agent()) 