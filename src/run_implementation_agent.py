import asyncio
import json
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, List

from src.logger import LOGGER
from src.main import _execute_chat_logic
from pydantic import BaseModel

class TaskQuery(BaseModel):
    task_id: str
    task_description: dict
    task_type: str

class TaskFileHandler(FileSystemEventHandler):
    def __init__(self):
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
        
        try:
            for index, task in enumerate(tasks, 1):
                task_description = {
                    "project_path": "./test_project",
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
                response = await _execute_chat_logic(query)
                LOGGER.info(f"Task {index} completed. Response: {response}")
                
                await asyncio.sleep(1)
            
            # Create implementation_success file after all tasks are completed
            success_file_path = Path('./test_project/implementation_success')
            success_file_path.touch()
            LOGGER.info("Created implementation_success file to signal completion")
                
        except Exception as e:
            LOGGER.error(f"Error processing tasks: {e}")

def start_monitoring():
    path = Path('/app/test_project')
    event_handler = TaskFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    
    LOGGER.info(f"Started monitoring directory: {path}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    LOGGER.info("Starting file monitor for tasks")
    start_monitoring() 