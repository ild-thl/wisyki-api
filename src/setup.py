from langchain.vectorstores import Chroma
from chromadb.config import Settings
from chromadb import PersistentClient
from langchain.embeddings import HuggingFaceInstructEmbeddings
from FlagEmbedding import FlagReranker
from .models.DB import DB
import os


def load_embedding_functions():
    return {
        "instructor-large": HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            query_instruction="",
            embed_instruction="",
        ),
        "instructor-skillfit": HuggingFaceInstructEmbeddings(
            model_name="pascalhuerten/instructor-skillfit",
            query_instruction="Represent the learning outcome for retrieving relevant skills: ",
            embed_instruction="Represent the skill for retrieval: ",
        ),
    }


def load_escodb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/esco_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_dkzdb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/dkz_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_gretadb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/greta_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_reranker():
    return FlagReranker(
        "pascalhuerten/bge_reranker_skillfit", use_fp16=True
    )  # use fp16 can speed up computing


def setup():
    embedding_functions = load_embedding_functions()
    skilldbs = {
        "ESCO": load_escodb(embedding_functions["instructor-skillfit"]),
        "DKZ": load_dkzdb(embedding_functions["instructor-skillfit"]),
        "GRETA": load_gretadb(embedding_functions["instructor-skillfit"]),
    }
    reranker = load_reranker()

    # db = None
    db = DB()

    return embedding_functions, skilldbs, reranker, db
