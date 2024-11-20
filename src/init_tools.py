import os

from dotenv import load_dotenv
from langchain.agents import tool

# from langchain.agents import AgentType, initialize_agent, tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from src.logger import LOGGER
from src.retrievers import create_vectorstore
# from src.documents import read_py_file

load_dotenv()

RUN_IN_DOCKER = os.environ.get("RUN_IN_DOCKER", "").lower() in (
    "yes",
    "y",
    "on",
    "true",
    "1",
)


def create_multiquery_retriever(vectorstore: VectorStore) -> MultiQueryRetriever:
    llm = ChatOpenAI(temperature=0)
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )
    multiquery_retriever.include_original = True

    return multiquery_retriever


LOGGER.info("init start")
# TODO create retrievers

retriever = create_vectorstore(
    retriever_name=os.getenv("RETRIEVER_DB"),
    user=os.getenv("RETRIEVER_USER"),
    password=os.getenv("RETRIEVER_PASSWORD"),
    port=5432 if RUN_IN_DOCKER else os.getenv("RETRIEVER_PORT"),
    db=os.getenv("RETRIEVER_DB"),
    add_docs=False,
)
LOGGER.info("retrievers init done")

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

tools = [
    rag_search,
    create_file,
    project_structure
]
