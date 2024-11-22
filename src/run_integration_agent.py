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

class IntegrationFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.is_processing = False
    
    def on_created(self, event):
        if event.src_path.endswith('implementation_success') and not self.is_processing:
            self.is_processing = True
            LOGGER.info(f"Detected implementation success file: {event.src_path}")
            asyncio.run(self.process_integration())
            self.is_processing = False
    
    async def process_integration(self):
        try:
            # Create a simple integration task
            task_description = {
                "project_path": "./test_project",
                "description": "Perform final integration and verification",
                "file_path": "*",  # Indicates all project files
                "technology": "all",
                "requirements": ["Verify all components are properly integrated"],
                "acceptance_criteria": ["All services are working together correctly"],
            }
            
            query = TaskQuery(
                task_id="integration_task",
                task_description=task_description,
                task_type="integration"
            )
            
            LOGGER.info("Starting integration process")
            response = await _execute_chat_logic(query)
            LOGGER.info(f"Integration completed. Response: {response}")
            
            # Create project_success file after integration is completed
            
            success_file_path = Path('./test_project/project_success')
            success_file_path.touch()
            LOGGER.info("Created project_success file to signal completion")
            
                
        except Exception as e:
            LOGGER.error(f"Error during integration: {e}")

def start_monitoring():
    path = Path('/app/test_project')
    event_handler = IntegrationFileHandler()
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
    LOGGER.info("Starting file monitor for integration")
    start_monitoring() 