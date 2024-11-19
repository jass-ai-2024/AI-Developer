import glob
import os
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema.document import Document
from tqdm import tqdm

from src.logger import LOGGER

def read_py_file(file_path):
    """
    Читает содержимое Python-файла и возвращает его как одну строку.

    :param file_path: Путь к файлу.
    :return: Содержимое файла в виде строки.
    """
    LOGGER.info('START READ_PY_FILE')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Чтение содержимого файла и замена символов новой строки
            content = file.read()  # .replace('\n', ' ')
        return content
    except FileNotFoundError:
        return "Файл не найден."
    except Exception as e:
        return f"Произошла ошибка: {e}"

    
def get_all_files(abs_dir_path: str):

    documents_list = []

    pattern = os.path.join(abs_dir_path, "**", "*.py")
    files = glob.glob(pattern, recursive=True)

    for file in files:
        with open(file) as f:
            content = f.read()
            source = os.path.basename(file)
            doc = Document(page_content=content, metadata={"source": source})
            documents_list.append(doc)

    return documents_list
