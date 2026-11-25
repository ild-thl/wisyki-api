import os
import json
from pathlib import Path
from langchain_chroma import Chroma
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from .models.DB import DB
from .embeddings import HuggingFaceInferenceEmbeddings, load_embedding_function
from .collection_manager import populate_collection_from_import


def connect_skill_collection() -> tuple[chromadb.HttpClient, Collection]:
    """Connect to the configured Chroma collection without populating its data."""
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
    return client, collection


def load_skilldb(
    embedding: HuggingFaceInferenceEmbeddings,
) -> tuple[Chroma, Collection]:
    """Load or create the skill vector store from external Chroma server"""
    client, collection = connect_skill_collection()
    chroma_collection = collection.name

    # Check if collection is empty and populate from import directory if needed
    if collection.count() == 0:
        print(
            f"Collection '{chroma_collection}' is empty. Initializing from import directory..."
        )
        populate_collection_from_import(
            client,
            collection,
            embedding,
            chroma_collection,
            force_recompute=False,
            save_embeddings=True,
        )

    _ensure_isced_f_fields_of_education(
        client, collection, embedding, chroma_collection
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

    return skilldb, collection


def _ensure_isced_f_fields_of_education(
    client,
    collection: Collection,
    embedding: HuggingFaceInferenceEmbeddings,
    collection_name: str,
    import_dir: Path = Path("data/import"),
) -> None:
    """Refresh ISCED-F vectors when the imported vocabulary version changes."""
    vocabulary_file = import_dir / "isced-f-fields-of-education.jsonl"
    if not vocabulary_file.is_file():
        return

    expected: dict[str, tuple[str, str | None]] = {}
    with vocabulary_file.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            record_id = record.get("id")
            version = metadata.get("source_version")
            if (
                metadata.get("taxonomy") == "ISCED_F"
                and metadata.get("embedding") is not False
                and isinstance(record_id, str)
                and version
                and isinstance(record.get("document"), str)
                and record["document"].strip()
            ):
                expected[record_id] = (version, metadata.get("document_version"))

    current = collection.get(where={"taxonomy": "ISCED_F"}, include=["metadatas"])
    current_versions = {
        record_id: (
            metadata.get("source_version"),
            metadata.get("document_version"),
        )
        for record_id, metadata in zip(
            current.get("ids", []), current.get("metadatas", []) or []
        )
    }
    changed = set(current_versions) != set(expected) or any(
        current_versions.get(record_id) != version
        for record_id, version in expected.items()
    )

    if changed:
        collection.delete(where={"taxonomy": "ISCED_F"})
        populate_collection_from_import(
            client,
            collection,
            embedding,
            collection_name,
            force_recompute=False,
            save_embeddings=False,
            import_dir=import_dir,
        )


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


def setup() -> tuple[HuggingFaceInferenceEmbeddings, Chroma, set[str], DB, Collection]:
    """Initialize all components needed for the API"""
    embedding_function = load_embedding_function()
    skilldb, skill_collection = load_skilldb(embedding_function)
    domains = load_domains()
    db = DB()

    return embedding_function, skilldb, domains, db, skill_collection
