# import os
# import re
# import time
# from typing import List

# from bs4 import BeautifulSoup

# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options


def init_embeddings_model():
    """
    Initialize the embeddings model using the specified model name and keyword arguments.

    Returns:
        HuggingFaceBgeEmbeddings: The initialized embeddings model.
    """
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings
