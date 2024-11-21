import os
from typing import List

from dotenv import load_dotenv

# from langchain.agents import AgentType, initialize_agent, tool
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()


def create_parent_document_retriever(
    docs: List[Document],
    retriever_name: str,
    user: str,
    password: str,
    port: int,
    db: str,
    k: int = 20,
    child_chunk_size: int = 500,
    parent_chunk_size: int = 3000,
    use_parent_splitter: bool = False,
) -> ParentDocumentRetriever:
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",  # text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    )

    if use_parent_splitter:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
    else:
        parent_splitter = None

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

    connection = f"postgresql+psycopg://{user}:{password}@{retriever_name}:{port}/{db}"  # Uses psycopg3!

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=retriever_name,
        connection=connection,
        use_jsonb=True,
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k},
    )

    retriever.add_documents(docs)
    return retriever


def create_vectorstore(
    retriever_name: str,
    user: str,
    password: str,
    port: int,
    db: str,
    add_docs: bool,
    splitter_chunk_size: int = 8000,  # "text-embedding-3-small" max context length is 8191 token
    docs: List[Document] = None,
) -> PGVector:
    if add_docs and docs is None:
        raise Exception("add_docs is True and docs is None")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",  # text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    )

    connection = f"postgresql+psycopg://{user}:{password}@{retriever_name}:{port}/{db}"  # Uses psycopg3!

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=retriever_name,
        connection=connection,
        use_jsonb=True,
    )

    if add_docs and docs is not None:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=splitter_chunk_size,
            chunk_overlap=100,
        )

        docs_processed = []
        for doc in docs:
            docs_processed += splitter.split_documents([doc])

        vectorstore.add_documents(docs)

    return vectorstore
