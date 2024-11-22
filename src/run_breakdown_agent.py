import asyncio
import json
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.logger import LOGGER
from src.main import _execute_chat_logic
from pydantic import BaseModel

class TaskQuery(BaseModel):
    task_id: str
    task_description: dict
    task_type: str

class BreakdownFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.is_processing = False
    
    def on_created(self, event):
        if 'arch_services_v1.json' in event.src_path and not self.is_processing:  # not event.is_directory and 
            self.is_processing = True
            LOGGER.info(f"Detected architecture services file: {event.src_path}")
            asyncio.run(self.process_breakdown(event.src_path))
            self.is_processing = False
    
    async def process_breakdown(self, file_path: str):
        try:
            # Read the architecture services file
            with open(file_path, 'r') as f:
                arch_services = json.load(f)
            
            # Create a breakdown task
            task_description = {
                "description": "Break down architecture into implementation tasks",
                "architecture": arch_services,
                "file_path": "*",
                "technology": "all",
                "requirements": ["Create detailed implementation tasks"],
                "acceptance_criteria": ["All components and services are properly broken down into tasks"],
            }
            
            query = TaskQuery(
                task_id="breakdown_task",
                task_description=task_description,
                task_type="task_breakdown"
            )
            
            LOGGER.info("Starting task breakdown process")
            response = await _execute_chat_logic(query)
            LOGGER.info(f"Task breakdown completed. Response: {response}")
            
            # Create implementation_ready file to signal the next phase
            success_file_path = Path('./test_project/implementation_ready')
            success_file_path.touch()
            LOGGER.info("Created implementation_ready file to signal completion")
                
        except Exception as e:
            LOGGER.error(f"Error during task breakdown: {e}")

def start_monitoring():
    path = Path('./test_project')
    event_handler = BreakdownFileHandler()
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
    LOGGER.info("Starting file monitor for task breakdown")
    start_monitoring() 