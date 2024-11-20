import os
import subprocess
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.logger import LOGGER
from src.file_monitor import check_architecture_file
from dotenv import load_dotenv

def ensure_directory_exists(path: str):
    """
    Создает директорию, если она не существует.
    
    Args:
        path (str): Путь к директории
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            LOGGER.info(f"Создана директория: {path}")
        except Exception as e:
            LOGGER.error(f"Ошибка при создании директории: {e}")
            sys.exit(1)

def main():
    load_dotenv(override=True)
    
    # Получаем путь к тестовой директории
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "test"
    project_path = os.path.join(root_dir, test_dir)
    
    # Создаем директорию, если она не существует
    ensure_directory_exists(project_path)
    
    LOGGER.info(f"Мониторинг директории: {project_path}")
    
    # Проверяем наличие файла архитектуры
    LOGGER.info("Начало проверки файла архитектуры")
    version = check_architecture_file(project_path, timeout=60)
    
    if not version:
        LOGGER.error("Файл архитектуры не найден")
        sys.exit(1)
    
    LOGGER.info(f"Найден файл архитектуры версии: {version}")
    
    # Сохраняем путь в переменную окружения
    os.environ["PROJECT_PATH"] = project_path
    
    # Запускаем FastAPI приложение
    try:
        LOGGER.info("Запуск агента")
        process = subprocess.Popen(
            ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"],
            env=os.environ.copy()
        )
        
        # Ждем завершения процесса
        process.wait()
        
        # Создаем файл завершения
        done_path = os.path.join(project_path, "code_done.txt")
        with open(done_path, "w") as f:
            f.write(f"version: {version}\n")
            f.write(f"status: completed\n")
        
        LOGGER.info("Процесс успешно завершен")
        
    except KeyboardInterrupt:
        LOGGER.info("Процесс прерван пользователем")
        process.terminate()
        sys.exit(0)
    except Exception as e:
        LOGGER.error(f"Ошибка при выполнении: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 