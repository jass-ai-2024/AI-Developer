import os
import time
import json
import shutil
from typing import Optional
from src.logger import LOGGER

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

def move_to_done(directory_path: str, version: str) -> bool:
    """
    Перемещает обработанный файл архитектуры в директорию code_done.
    
    Args:
        directory_path (str): Путь к директории с файлом
        version (str): Версия файла (например, 'v0')
    """
    try:
        output_directory = os.path.join(directory_path, "code_done")
        
        # Создаем директорию code_done, если её нет
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
        source_file = os.path.join(directory_path, f'arch_services_{version}.json')
        dest_file = os.path.join(output_directory, f'arch_services_{version}.json')
        
        if os.path.exists(source_file):
            shutil.move(source_file, dest_file)
            LOGGER.info(f"Файл перемещен в: {dest_file}")
            return True
        else:
            LOGGER.error(f"Исходный файл не найден: {source_file}")
            return False
            
    except Exception as e:
        LOGGER.error(f"Ошибка при перемещении файла: {e}")
        return False 