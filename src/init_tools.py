import os
from pathlib import Path
from typing import Optional

import shutil

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.indexes import SQLRecordManager, index
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from src.logger import LOGGER
from src.retrievers import create_vectorstore
from src.documents import get_all_files_docs

load_dotenv()

RUN_IN_DOCKER = os.environ.get("RUN_IN_DOCKER", "").lower() in (
    "yes",
    "y",
    "on",
    "true",
    "1",
)

def init_vectorstore() -> VectorStore:
    """Initialize vector store with document indexing"""
    LOGGER.info("Starting vector store initialization")
    
    # Create vector store
    vectorstore = create_vectorstore(
        retriever_name=os.getenv("RETRIEVER_DB"),
        user=os.getenv("RETRIEVER_USER"),
        password=os.getenv("RETRIEVER_PASSWORD"),
        port=5432 if RUN_IN_DOCKER else os.getenv("RETRIEVER_PORT"),
        db=os.getenv("RETRIEVER_DB"),
        add_docs=False  # Не добавляем документы через vectorstore напрямую
    )
    
    # Create connection string for record manager
    connection_string = (
        f"postgresql+psycopg://{os.getenv('RETRIEVER_USER')}:{os.getenv('RETRIEVER_PASSWORD')}@"
        f"{'pgvector-docs' if RUN_IN_DOCKER else 'localhost'}:{5432 if RUN_IN_DOCKER else os.getenv('RETRIEVER_PORT')}/"
        f"{os.getenv('RETRIEVER_DB')}"
    )
    
    # Initialize record manager with namespace
    namespace = f"pgvector/{os.getenv('RETRIEVER_DB')}"
    record_manager = SQLRecordManager(
        namespace=namespace,
        db_url=connection_string
    )
    
    # Create schema for record manager
    record_manager.create_schema()
    
    # Get documents
    docs = get_all_files_docs(str(Path('./test_project')))
    LOGGER.info(f"Found {len(docs)} documents to index")
    
    # Index documents with incremental cleanup
    index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source"
    )
    
    LOGGER.info("Vector store initialization completed")
    return vectorstore

def create_multiquery_retriever(vectorstore: VectorStore) -> MultiQueryRetriever:
    llm = ChatOpenAI(temperature=0)
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )
    multiquery_retriever.include_original = True
    return multiquery_retriever

def update_vectorstore(vectorstore: VectorStore):
    """Update vector store with latest document changes"""
    LOGGER.info("Starting vector store update")
    
    # Create connection string for record manager
    connection_string = (
        f"postgresql+psycopg://{os.getenv('RETRIEVER_USER')}:{os.getenv('RETRIEVER_PASSWORD')}@"
        f"{'pgvector-docs' if RUN_IN_DOCKER else 'localhost'}:{5432 if RUN_IN_DOCKER else os.getenv('RETRIEVER_PORT')}/"
        f"{os.getenv('RETRIEVER_DB')}"
    )
    
    # Initialize record manager with namespace
    namespace = f"pgvector/{os.getenv('RETRIEVER_DB')}"
    record_manager = SQLRecordManager(
        namespace=namespace,
        db_url=connection_string
    )
    
    # Get latest documents
    docs = get_all_files_docs(str(Path('./test_project')))
    LOGGER.info(f"Found {len(docs)} documents to index")
    
    # Index documents with incremental cleanup
    result = index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",  # Using incremental mode to efficiently handle updates
        source_id_key="source"
    )
    
    LOGGER.info(
        f"Vector store update completed: "
        f"Added {result['num_added']}, "
        f"Updated {result['num_updated']}, "
        f"Deleted {result['num_deleted']}, "
        f"Skipped {result['num_skipped']} documents"
    )
    return vectorstore

# Initialize vector store and retrievers
LOGGER.info("Starting tools initialization")
retriever = init_vectorstore()
LOGGER.info("Vector store and retrievers initialized")

@tool("rag_search")
def rag_search(query: str, file_path: str):
    """Searches information related to query in particular file"""
    LOGGER.info(f"[TOOL CALL] rag_search | Input: query='{query}', file_path='{file_path}'")
    retrieved_docs = retriever.similarity_search(query, k=10)
    result = "".join(["\n```\n" + doc.page_content + "\n```\n" + f'Metadata: {doc.metadata}\n' for doc in retrieved_docs])
    LOGGER.info(f"[TOOL RESULT] rag_search | Found {len(retrieved_docs)} documents")
    return result if retrieved_docs else f"No results found. Params: query: {query}, file_path: {file_path}"

@tool('create_file')
def create_file(file_path, contents):
    """Create a file with the given name and contents."""
    LOGGER.info(f"[TOOL CALL] create_file | Input: file_path='{file_path}', content_length={len(contents)}")
    try:
        with open(file_path, "w") as file:
            file.write(contents)
        update_vectorstore(retriever)
        LOGGER.info(f"[TOOL RESULT] create_file | File created successfully: '{file_path}'")
        return f"File '{file_path}' created and filled with content."
    except Exception as e:
        LOGGER.error(f"[TOOL ERROR] create_file | {e}")
        return f'Something went wrong. Error: {e}'

@tool('project_structure')    
def project_structure(directory_path):
    """List the project structure of a given directory."""
    LOGGER.info(f"[TOOL CALL] project_structure | Input: directory_path='{directory_path}'")
    ans = ''
    for root, dirs, files in os.walk(directory_path):
        ans += f"Root: {root}\n"
        ans += f"Directories: {dirs}\n"
        ans += f"Files: {files}\n"
    LOGGER.info(f"[TOOL RESULT] project_structure | Scan completed for: '{directory_path}'")
    return ans

@tool('read_file')
def read_file(file_path):
    """Read and return the contents of a file."""
    LOGGER.info(f"[TOOL CALL] read_file | Input: file_path='{file_path}'")
    try:
        with open(file_path, "r") as file:
            contents = file.read()
        LOGGER.info(f"[TOOL RESULT] read_file | Successfully read file, content_length={len(contents)}")
        return contents
    except FileNotFoundError:
        LOGGER.error(f"[TOOL ERROR] read_file | File not found: '{file_path}'")
        return f"File '{file_path}' not found."
    except Exception as e:
        LOGGER.error(f"[TOOL ERROR] read_file | {e}")
        return f"An error occurred: {e}"

@tool('create_directory')
def create_directory(directory_path: str):
    """Create a new directory at the specified path."""
    LOGGER.info(f"[TOOL CALL] create_directory | Input: directory_path='{directory_path}'")
    try:
        os.makedirs(directory_path, exist_ok=True)
        LOGGER.info(f"[TOOL RESULT] create_directory | Directory created successfully")
        return f"Directory '{directory_path}' created successfully."
    except Exception as e:
        LOGGER.error(f"[TOOL ERROR] create_directory | {e}")
        return f"Failed to create directory. Error: {e}"

@tool('remove_directory')
def remove_directory(directory_path: str):
    """Remove a directory and all its contents at the specified path."""
    LOGGER.info(f"[TOOL CALL] remove_directory | Input: directory_path='{directory_path}'")
    try:
        shutil.rmtree(directory_path)
        LOGGER.info(f"[TOOL RESULT] remove_directory | Directory removed successfully")
        return f"Directory '{directory_path}' removed successfully."
    except Exception as e:
        LOGGER.error(f"[TOOL ERROR] remove_directory | {e}")
        return f"Failed to remove directory. Error: {e}"

@tool('remove_file')
def remove_file(file_path: str):
    """Remove a file at the specified path."""
    LOGGER.info(f"[TOOL CALL] remove_file | Input: file_path='{file_path}'")
    try:
        os.remove(file_path)
        update_vectorstore(retriever)
        LOGGER.info(f"[TOOL RESULT] remove_file | File removed successfully")
        return f"File '{file_path}' removed successfully."
    except Exception as e:
        LOGGER.error(f"[TOOL ERROR] remove_file | {e}")
        return f"Failed to remove file. Error: {e}"

tools = [
    rag_search,
    create_file,
    project_structure,
    read_file,
    create_directory,
    remove_directory,
    remove_file
]

