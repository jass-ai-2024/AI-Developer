import asyncio
import json
from pathlib import Path
from typing import List, Dict

from src.logger import LOGGER
from src.main import ProjectPipeline
from src.models import ReviewQuery, ReviewResponse

class ReviewFileHandler:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.is_processing = False

    async def get_generated_files(self) -> List[Dict]:
        """Получение списка сгенерированных файлов из final_tasks.json"""
        try:
            tasks_file = Path(self.pipeline.project_dir) / "final_tasks.json"
            if not tasks_file.exists():
                LOGGER.error("final_tasks.json not found")
                return []

            with open(tasks_file, 'r') as f:
                data = json.load(f)
            return data.get('subtasks', [])
        except Exception as e:
            LOGGER.error(f"Error reading tasks file: {e}")
            return []

    async def read_file_content(self, file_path: str) -> str:
        """Чтение содержимого файла"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            LOGGER.error(f"Error reading file {file_path}: {e}")
            return ""

    async def process_reviews(self):
        """Обработка ревью для всех сгенерированных файлов"""
        try:
            tasks = await self.get_generated_files()
            if not tasks:
                LOGGER.error("No tasks found for review")
                return

            LOGGER.info(f"Found {len(tasks)} files to review")
            review_results = []

            for index, task in enumerate(tasks, 1):
                try:
                    file_path = task.get("file_path")
                    if not file_path:
                        continue

                    full_path = Path(self.pipeline.project_dir) / file_path
                    if not full_path.exists():
                        LOGGER.warning(f"File not found: {full_path}")
                        continue

                    code_content = await self.read_file_content(str(full_path))
                    if not code_content:
                        continue

                    query = ReviewQuery(
                        task_id=f"review_{index}",
                        file_path=file_path,
                        code_content=code_content,
                        requirements=task.get("requirements", []),
                        acceptance_criteria=task.get("acceptance_criteria", [])
                    )

                    LOGGER.info(f"Reviewing file {index}/{len(tasks)}: {file_path}")
                    review_result = await self.pipeline.execute_review(query)
                    review_results.append({
                        "file_path": file_path,
                        "review_result": review_result
                    })

                    await asyncio.sleep(1)

                except Exception as e:
                    LOGGER.error(f"Error reviewing file {index}: {e}")
                    continue

            # Сохраняем результаты ревью
            review_file = Path(self.pipeline.project_dir) / f"code_review_v{self.pipeline.architecture_version}.json"
            with open(review_file, 'w') as f:
                json.dump({"reviews": review_results}, f, indent=2)

            # Создаем маркер завершения ревью
            marker_file = Path(self.pipeline.project_dir) / f"review_complete_v{self.pipeline.architecture_version}.txt"
            marker_file.touch()

            LOGGER.info("Code review completed successfully")
            return True

        except Exception as e:
            LOGGER.error(f"Error in review process: {e}")
            return False

async def run_review_agent():
    pipeline = ProjectPipeline()
    try:
        if not await pipeline.initialize_database():
            raise Exception("Failed to initialize database")

        if not await pipeline.initialize_review_agent():
            raise Exception("Failed to initialize review agent")

        # Ждем завершения генерации кода
        if not await pipeline.wait_for_code_generation():
            raise Exception("Code generation not completed")

        # Запускаем процесс ревью
        review_handler = ReviewFileHandler(pipeline)
        if not await review_handler.process_reviews():
            raise Exception("Failed to complete code review")

    except Exception as e:
        LOGGER.error(f"Review agent error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(run_review_agent()) 