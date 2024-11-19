import os
# from ast import literal_eval

from dotenv import load_dotenv
from src.documents import get_all_files_docs

# from documents import (
#     get_documentation_docs,
#     get_flowui_trainings_docs,
#     get_forum_docs,
#     get_ui_samples_docs,
# )
from src.logger import LOGGER
from pathlib import Path
from src.retrievers import create_vectorstore

load_dotenv()

RUN_IN_DOCKER = os.environ.get("RUN_IN_DOCKER", "").lower() in (
    "yes",
    "y",
    "true",
    "1",
)

if __name__ == "__main__":
    LOGGER.info('INIT_VECTORSTORES START')

    documentation_docs = get_all_files_docs(str(Path('./test_project')))
    LOGGER.info("get_documentation_docs done")
    vectorstore = create_vectorstore(
        retriever_name=os.getenv("RETRIEVER_DB"),
        user=os.getenv("RETRIEVER_USER"),
        password=os.getenv("RETRIEVER_PASSWORD"),
        port=5432 if RUN_IN_DOCKER else os.getenv("RETRIEVER_PORT"),
        db=os.getenv("RETRIEVER_DB"),
        add_docs=True,
        docs=documentation_docs,
    )
    LOGGER.info("INIT_VECTORSTORES DONE")
