import os

from dotenv import load_dotenv
from langchain.agents import tool

# from langchain.agents import AgentType, initialize_agent, tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from src.logger import LOGGER
from src.retrievers import create_vectorstore
from documents import read_py_file

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

@tool("search_in_file")
def search_in_file(query: str, file_path: str):
    retrieved_docs = retriever.similarity_search(
        query, k=10, 
        filter={"id": {"$in": [file_path]}}
    )
    if len(retrieved_docs) == 0:
        return "Something went wrong. Please try using another tool or rephrasing your query."
    return "".join(["\n```\n" + doc.page_content + "\n```\n" for doc in retrieved_docs])

tools = [
    search_in_file
]
