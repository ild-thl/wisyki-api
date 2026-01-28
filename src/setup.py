import os
import json
from pathlib import Path
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from FlagEmbedding import FlagReranker
from .models.DB import DB
from .embeddings import load_embedding_function
from .collection_manager import _populate_collection_from_import


def load_skilldb(embedding):
    """Load or create the skill vector store from external Chroma server"""
    chroma_host = os.getenv("CHROMA_HOST", "chroma")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_collection = os.getenv("CHROMA_COLLECTION", "wisyki-skills")
    chroma_database = os.getenv("CHROMA_DATABASE", "wisyki")
    chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")

    # Connect to remote Chroma server
    client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port,
        settings=Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=os.getenv("CHROMA_SERVER_AUTHN_CREDENTIALS"),
            chroma_auth_token_transport_header="Authorization",
        ),
        tenant=chroma_tenant,
    )

    # Get or create the collection for this service
    collection = client.get_or_create_collection(
        name=chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    # Check if collection is empty and populate from import directory if needed
    if collection.count() == 0:
        print(
            f"Collection '{chroma_collection}' is empty. Initializing from import directory..."
        )
        _populate_collection_from_import(
            client,
            collection,
            embedding,
            chroma_collection,
            force_recompute=False,
            save_embeddings=True,
        )

    # Test connection by retrieving document count and first document inclduing metadata
    doc_count = collection.count()
    print(
        f"Chroma collection '{chroma_collection}' connected. Document count: {doc_count}"
    )

    # Create Chroma vector store wrapper
    # Important: Pass collection_name explicitly to ensure Chroma reuses the existing collection
    # rather than creating a new random UUID collection
    skilldb = Chroma(
        client=client,
        collection_name=chroma_collection,
        embedding_function=embedding,
    )

    # Verify the collection is correctly set
    if skilldb._collection.name != chroma_collection:
        print(
            f"⚠ Warning: Collection name mismatch. Expected '{chroma_collection}', got '{skilldb._collection.name}'"
        )

    return skilldb


def load_reranker():
    return FlagReranker(
        "isy-thl/bge-reranker-base-course-skill-tuned", use_fp16=True
    )  # use fp16 can speed up computing


def load_domains():
    """Load domain keywords from files"""
    with open(
        os.path.join("data", "domains", "languages_de.txt"), "r", encoding="utf-8"
    ) as file:
        languages = [line.lower().strip() for line in file.read().splitlines()]
    with open(
        os.path.join("data", "domains", "programminglanguages.txt"),
        "r",
        encoding="utf-8",
    ) as file:
        programminglanguages = [
            line.lower().strip() for line in file.read().splitlines()
        ]
    with open(
        os.path.join("data", "domains", "otherdomains.txt"), "r", encoding="utf-8"
    ) as file:
        otherdomains = [line.lower().strip() for line in file.read().splitlines()]

    domains = set(languages) | set(programminglanguages) | set(otherdomains)
    return domains


def setup():
    """Initialize all components needed for the API"""
    embedding_function = load_embedding_function()
    skilldb = load_skilldb(embedding_function)
    if os.getenv("DISABLE_RERANKER", "0") == "1":
        print("Reranker is disabled via DISABLE_RERANKER environment variable.")
        reranker = None
    else:
        print("Loading reranker...")
        reranker = load_reranker()
    domains = load_domains()
    db = DB()

    return embedding_function, skilldb, reranker, domains, db
