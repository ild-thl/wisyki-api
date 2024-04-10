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


def load_skilldb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/skill_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_reranker():
    return FlagReranker(
        "pascalhuerten/bge_reranker_skillfit", use_fp16=True
    )  # use fp16 can speed up computing

def load_domains():
    with open(os.path.join("data", "domains", "languages_de.txt"), "r", encoding='utf-8') as file:
        languages = [line.lower().strip() for line in file.read().splitlines()]
    with open(os.path.join("data", "domains", "programminglanguages.txt"), "r", encoding='utf-8') as file:
        programminglanguages = [line.lower().strip() for line in file.read().splitlines()]
    with open(os.path.join("data", "domains", "otherdomains.txt"), "r", encoding='utf-8') as file:
        otherdomains = [line.lower().strip() for line in file.read().splitlines()]
    
    domains = set(languages) | set(programminglanguages) | set(otherdomains)
    return domains

def setup():
    embedding_functions = load_embedding_functions()
    skilldb = load_skilldb(embedding_functions["instructor-skillfit"])
    reranker = load_reranker()

    domains = load_domains()
    
    db = DB()

    return embedding_functions, skilldb, reranker, domains, db
