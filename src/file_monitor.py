import os
import time
import json
from typing import Optional
from src.logger import LOGGER
from pathlib import Path

def check_architecture_file(directory_path: str, interval: int = 5, timeout: Optional[int] = None) -> Optional[str]:
    """
    Проверяет наличие файла arch_services_*.json и возвращает его версию.
    
    Args:
        directory_path (str): Путь к директории для мониторинга
        interval (int): Интервал проверки в секундах
        timeout (Optional[int]): Таймаут в секундах
    """
    start_time = time.time()
    LOGGER.info(f"Начало мониторинга директории: {directory_path}")
    
    while True:
        if timeout and (time.time() - start_time > timeout):
            LOGGER.info("Превышено время ожидания")
            return None
        
        try:
            files = os.listdir(directory_path)
            arch_files = [f for f in files if f.startswith('arch_services_') and f.endswith('.json')]
            
            if arch_files:
                file_path = os.path.join(directory_path, arch_files[0])
                # Извлекаем версию из имени файла
                version = arch_files[0].replace('arch_services_', '').replace('.json', '')
                
                # Проверяем валидность JSON
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    LOGGER.info(f"Найден файл архитектуры версии: {version}")
                    return version
                except json.JSONDecodeError:
                    LOGGER.error(f"Файл {file_path} содержит невалидный JSON")
                    return None
            
            LOGGER.info(f"Файл архитектуры не найден, ожидание {interval} секунд...")
            time.sleep(interval)
            
        except Exception as e:
            LOGGER.error(f"Ошибка при проверке директории: {e}")
            return None
        
def create_empty_project_structure(version: str, base_dir: str = "test") -> bool:
    """
    Создает структуру проекта с пустыми файлами после проверки версии архитектуры
    
    Args:
        version: Версия архитектуры (arch_services_v*)
        base_dir: Базовая директория для создания проекта
    """
    try:
        # Проверка входных параметров
        if not version or not version.strip():
            LOGGER.error("Version cannot be empty")
            return False
            
        if not isinstance(base_dir, str) or not base_dir.strip():
            LOGGER.error("Base directory path must be a non-empty string")
            return False
            
        # Создаем базовую директорию, если её нет
        base_path = Path(base_dir)
        base_path.mkdir(exist_ok=True)
        LOGGER.info(f"Base directory created/verified: {base_dir}")

        # Проверка прав доступа
        try:
            test_file = base_path / ".test_write_access"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            LOGGER.error(f"No write permission in directory {base_dir}: {e}")
            return False

        # Создаем корневую директорию проекта
        project_path = base_path / f"project_{version}"
        project_path.mkdir(exist_ok=True)
        LOGGER.info(f"Project directory created: {project_path}")
        
        # Структура директорий и файлов
        directories = [
            "backend",
            "frontend/public",
            "frontend/src"
        ]
        
        files = [
            "backend/Dockerfile",
            "backend/main.py",
            "backend/requirements.txt",
            "backend/.env",
            "frontend/public/index.html",
            "frontend/src/App.js",
            "frontend/src/index.js",
            "frontend/.gitignore",
            "frontend/Dockerfile",
            "frontend/package.json",
            "frontend/.env",
            "docker-compose.yml",
            "README.md"
        ]

        # Создаем директории
        for directory in directories:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Created directory: {dir_path}")

        # Создаем пустые файлы
        for file in files:
            file_path = project_path / file
            file_path.parent.mkdir(parents=True, exist_ok=True)  # Создаем родительские директории
            file_path.touch()
            LOGGER.info(f"Created empty file: {file_path}")

        LOGGER.info(f"Project structure created successfully for version {version}")
        return True

    except Exception as e:
        LOGGER.error(f"Error creating project structure: {e}")
        return False