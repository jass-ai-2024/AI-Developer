import asyncio
from pathlib import Path
import json

from src.logger import LOGGER
from src.main import ProjectPipeline

async def run_task_agent():
    pipeline = ProjectPipeline()
    try:
        if not await pipeline.initialize_database():
            raise Exception("Failed to initialize database")
            
        if not await pipeline.initialize_task_agent():
            raise Exception("Failed to initialize task agent")
            
        # Ждем появления файла архитектуры
        if not await pipeline.check_architecture():
            raise Exception("Architecture file not found")
            
        # Разбиваем задачи
        arch_file = Path(pipeline.arch_dir) / f"arch_services_{pipeline.architecture_version}.json"
        with open(arch_file, 'r') as f:
            architecture = json.load(f)
            
        # Формируем запрос для разбиения задач
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
        
        # Выполняем разбиение
        if not await pipeline.split_tasks():
            raise Exception("Failed to split tasks")
            
        # Создаем маркер завершения
        marker_file = Path(pipeline.project_dir) / f"tasks_split_v{pipeline.architecture_version}.txt"
        marker_file.touch()
        
    except Exception as e:
        LOGGER.error(f"Task agent error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(run_task_agent()) 