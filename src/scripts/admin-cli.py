#!/usr/bin/env python3
"""Command-line administration for the Chroma skill collection.

Usage:
    python src/scripts/admin-cli.py status
    python src/scripts/admin-cli.py reset-collection [--force-recompute]
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_environment() -> None:
    if load_dotenv is None:
        return
    for filename in (".env", ".env.dev"):
        environment_file = PROJECT_ROOT / filename
        if environment_file.exists():
            load_dotenv(environment_file, override=False)


def _chroma_settings():
    from chromadb.config import Settings

    return Settings(
        anonymized_telemetry=False,
        chroma_client_auth_provider=(
            "chromadb.auth.token_authn.TokenAuthClientProvider"
        ),
        chroma_client_auth_credentials=os.getenv("CHROMA_SERVER_AUTHN_CREDENTIALS"),
        chroma_auth_token_transport_header="Authorization",
    )


def _get_chroma_client():
    import chromadb

    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "chroma"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
        settings=_chroma_settings(),
        tenant=os.getenv("CHROMA_TENANT", "default_tenant"),
    )


def _collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", "wisyki-skills")


def _import_directory() -> Path:
    return PROJECT_ROOT / "data" / "import"


def reset_collection(force_recompute: bool = False) -> bool:
    collection_name = _collection_name()
    recompute_message = " (forcing embedding recomputation)" if force_recompute else ""

    print(f"Resetting Chroma collection '{collection_name}'{recompute_message}...")
    print("WARNING: This deletes all documents and reloads data/import/.")
    if input("Are you sure? (yes/no): ").strip().lower() != "yes":
        print("Cancelled.")
        return False

    try:
        from src.collection_manager import _populate_collection_from_import
        from src.embeddings import load_embedding_function

        client = _get_chroma_client()
        try:
            client.delete_collection(name=collection_name)
        except Exception as error:
            print(f"No existing collection deleted: {error}")

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        embedding_function = load_embedding_function()
        _populate_collection_from_import(
            client,
            collection,
            embedding_function,
            collection_name,
            force_recompute=force_recompute,
            save_embeddings=True,
            import_dir=_import_directory(),
        )

        print(f"Collection reset successfully: {collection.count()} documents")
        return True
    except Exception as error:
        print(f"Collection reset failed: {error}")
        return False


def get_status() -> bool:
    collection_name = _collection_name()
    try:
        client = _get_chroma_client()
        collection = client.get_collection(name=collection_name)
        print("Collection status:")
        print(
            f"  Chroma server: {os.getenv('CHROMA_HOST', 'chroma')}:{os.getenv('CHROMA_PORT', '8000')}"
        )
        print(f"  Collection: {collection_name}")
        print(f"  Documents: {collection.count()}")
        print("  Status: healthy")
        return True
    except Exception as error:
        print(f"Could not get collection status: {error}")
        return False


def main() -> int:
    _load_environment()

    parser = argparse.ArgumentParser(
        description="Manage the Chroma skill collection directly from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reset_parser = subparsers.add_parser(
        "reset-collection",
        help="Delete and repopulate the Chroma collection from data/import/.",
    )
    reset_parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute all embeddings.",
    )
    subparsers.add_parser(
        "status",
        help="Show the collection status directly from Chroma.",
    )

    args = parser.parse_args()
    if args.command == "reset-collection":
        return 0 if reset_collection(args.force_recompute) else 1
    return 0 if get_status() else 1


if __name__ == "__main__":
    sys.exit(main())
