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

# Initialize vector store and retrievers
LOGGER.info("Starting tools initialization")
retriever = init_vectorstore()
LOGGER.info("Vector store and retrievers initialized")

@tool("rag_search")
def rag_search(query: str, file_path: str):
    """Searches information related to query in particular file"""
    retrieved_docs = retriever.similarity_search(
        query, k=10, 
        # filter={"id": {"$in": [file_path]}}
    )
    if len(retrieved_docs) == 0:
        return f"Something went wrong. Please try using another tool or rephrasing your query. Params: query: {query}, file_path: {file_path}"
    return "".join(["\n```\n" + doc.page_content + "\n```\n" + f'Metadata: {doc.metadata}\n' for doc in retrieved_docs])

@tool('create_file')
def create_file(file_path, contents):
    """Create a file with the given name and contents."""
    try:
        with open(file_path, "w") as file:
            file.write(contents)
        LOGGER.info(f"File '{file_path}' created and filled with content.")
        return f"File '{file_path}' created and filled with content."
    except Exception as e:
        return f'Something went wrong. Error: {e}'

@tool('project_structure')    
def project_structure(directory_path):
    """List the project structure of a given directory."""
    ans = ''
    for root, dirs, files in os.walk(directory_path):
        ans += f"Root: {root}\n"
        ans += f"Directories: {dirs}"
        ans += f"Files: {files}"
        
        LOGGER.info(f"Root: {root}")
        LOGGER.info(f"Directories: {dirs}")
        LOGGER.info(f"Files: {files}")
    return ans

# @tool('modify_file')
# def modify_file(file_path, diff_description):
#     """Modify file contents by launching the Diff Bot."""
#     try:
#         with open(file_path, "r") as file:
#             original_contents = file.read()

#         LOGGER.info("Launching Diff Bot...")
#         updated_contents = diff_chain.run(
#             {"original": original_contents, "diff_description": diff_description})
#         LOGGER.info(f"Updated contents:\n{updated_contents}")

#         # Simulate user approval step (this may require adaptation for a non-interactive flow)
#         approval = input(
#             "Do you want to save these changes to the file? (yes/no): ")
#         if approval.lower() == "yes":
#             with open(file_path, "w") as file:
#                 file.write(updated_contents)
#             LOGGER.info(f"File '{file_path}' updated successfully.")
#             return f"File '{file_path}' updated successfully."
#         else:
#             LOGGER.info("Changes discarded.")
#             return "Changes discarded."
#     except FileNotFoundError:
#         error_message = f"File '{file_path}' not found."
#         LOGGER.error(error_message)
#         return error_message
#     except Exception as e:
#         error_message = f"An error occurred: {e}"
#         LOGGER.error(error_message)
#         return error_message

@tool('read_file')
def read_file(file_path):
    """Read and return the contents of a file."""
    try:
        with open(file_path, "r") as file:
            contents = file.read()
        LOGGER.info(f"Contents of file '{file_path}':\n{contents}")
        return contents
    except FileNotFoundError:
        error_message = f"File '{file_path}' not found."
        LOGGER.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred: {e}"
        LOGGER.error(error_message)
        return error_message

# @tool('commit_changes')
# def commit_changes(commit_message):
#     """Stage and commit changes using git."""
#     try:
#         run_command_with_confirmation("git add .")
#         run_command_with_confirmation(f"git commit -m '{commit_message}'")
#         LOGGER.info(f"Changes committed with message: {commit_message}")
#         return f"Changes committed with message: '{commit_message}'"
#     except Exception as e:
#         error_message = f"An error occurred while committing changes: {e}"
#         LOGGER.error(error_message)
#         return error_message

# @tool('run_command_with_confirmation')
# def run_command_with_confirmation(command):
#     """Run a command after user confirmation."""
#     LOGGER.info(f"Suggested command:\n{command}")
#     approval = input("Do you want to execute this command? (yes/no): ")
#     if approval.lower() == "yes":
#         try:
#             result = subprocess.check_output(
#                 command, shell=True, stderr=subprocess.STDOUT)
#             LOGGER.info(result.decode())
#             return result.decode()
#         except subprocess.CalledProcessError as e:
#             error_message = f"Error executing command:\n{e.output.decode()}"
#             LOGGER.error(error_message)
#             return error_message
#     else:
#         LOGGER.info("Command not executed.")
#         return "Command not executed."

@tool('create_directory')
def create_directory(directory_path: str):
    """Create a new directory at the specified path."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        LOGGER.info(f"Directory '{directory_path}' created successfully.")
        return f"Directory '{directory_path}' created successfully."
    except Exception as e:
        return f"Failed to create directory. Error: {e}"

@tool('remove_directory')
def remove_directory(directory_path: str):
    """Remove a directory and all its contents at the specified path."""
    try:
        shutil.rmtree(directory_path)
        LOGGER.info(f"Directory '{directory_path}' removed successfully.")
        return f"Directory '{directory_path}' removed successfully."
    except Exception as e:
        return f"Failed to remove directory. Error: {e}"

@tool('remove_file')
def remove_file(file_path: str):
    """Remove a file at the specified path."""
    try:
        os.remove(file_path)
        LOGGER.info(f"File '{file_path}' removed successfully.")
        return f"File '{file_path}' removed successfully."
    except Exception as e:
        return f"Failed to remove file. Error: {e}"

tools = [
    rag_search,
    create_file,
    project_structure,
    read_file,
    # modify_file,
    # commit_changes,
    # run_command_with_confirmation,
    create_directory,
    remove_directory,
    remove_file
]

