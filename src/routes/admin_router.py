"""
Admin routes for system administrators to manage wisyki-api resources.

These endpoints require admin authentication (admin API key).
"""

import os
from fastapi import APIRouter, Header, HTTPException, status, Request
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])


class CollectionResetRequest(BaseModel):
    """Request body for collection reset"""

    confirm: bool = False
    force_recompute: bool = False


def _verify_admin_key(x_admin_key: Optional[str] = Header(None)) -> bool:
    """
    Verify admin API key from X-Admin-Key header.

    If ADMIN_API_KEY environment variable is not set, admin endpoints are disabled.
    If set, the provided header must match exactly.
    """
    admin_key = os.getenv("ADMIN_API_KEY")

    # If no admin key is configured, disable admin endpoints
    if not admin_key:
        return False

    # Verify the provided key matches
    return x_admin_key == admin_key


@router.post("/collection/reset")
async def reset_collection(
    request: CollectionResetRequest,
    x_admin_key: Optional[str] = Header(None),
    req: Request = None,
):
    """
    Reset and reinitialize the Chroma collection from import directory.
    
    This endpoint:
    1. Deletes the current collection
    2. Creates a new empty collection
    3. Repopulates it from data/import/
    
    **Requires:** X-Admin-Key header with valid admin API key
    
    **Parameters:**
    - `confirm`: Must be `true` to proceed (safety check)
    - `force_recompute`: If true, ignore precomputed embeddings and recompute all (default: false)
    
    **Returns:** Status and collection info
    
    **Example:**
    ```bash
    curl -X POST http://localhost:7680/admin/collection/reset \
      -H "X-Admin-Key: your-secret-key" \
      -H "Content-Type: application/json" \
      -d '{"confirm": true, "force_recompute": false}'
    ```
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header",
        )

    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm=true to proceed with collection reset",
        )

    try:
        from src.collection_manager import _populate_collection_from_import
        import chromadb
        from chromadb.config import Settings

        # Get app state
        app_state = req.app.state
        embedding_function = app_state.EMBEDDING_FUNCTION
        skilldb = app_state.SKILLDB

        # Get Chroma client and collection info from skilldb
        chroma_host = os.getenv("CHROMA_HOST", "chroma")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        chroma_collection = os.getenv("CHROMA_COLLECTION", "wisyki-skills")
        chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")

        # Create client
        client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=os.getenv(
                    "CHROMA_SERVER_AUTHN_CREDENTIALS"
                ),
                chroma_auth_token_transport_header="Authorization",
            ),
            tenant=chroma_tenant,
        )

        # Delete old collection
        try:
            client.delete_collection(name=chroma_collection)
        except Exception:
            pass  # Collection might not exist

        # Create new collection
        collection = client.create_collection(
            name=chroma_collection, metadata={"hnsw:space": "cosine"}
        )

        # Repopulate from import directory
        _populate_collection_from_import(
            client,
            collection,
            embedding_function,
            chroma_collection,
            force_recompute=request.force_recompute,
            save_embeddings=True,
        )

        final_count = collection.count()

        return {
            "status": "success",
            "message": "Collection reset and reinitialized",
            "collection": chroma_collection,
            "documents": final_count,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset collection: {str(e)}",
        )


@router.get("/collection/status")
async def collection_status(
    x_admin_key: Optional[str] = Header(None),
    req: Request = None,
):
    """
    Get current collection status and statistics.
    
    **Requires:** X-Admin-Key header with valid admin API key
    
    **Example:**
    ```bash
    curl http://localhost:7680/admin/collection/status \
      -H "X-Admin-Key: your-secret-key"
    ```
    """
    if not _verify_admin_key(x_admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header",
        )

    try:
        import chromadb
        from chromadb.config import Settings

        chroma_host = os.getenv("CHROMA_HOST", "chroma")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        chroma_collection = os.getenv("CHROMA_COLLECTION", "wisyki-skills")
        chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")

        client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=os.getenv(
                    "CHROMA_SERVER_AUTHN_CREDENTIALS"
                ),
                chroma_auth_token_transport_header="Authorization",
            ),
            tenant=chroma_tenant,
        )

        collection = client.get_collection(name=chroma_collection)

        return {
            "status": "healthy",
            "collection": chroma_collection,
            "documents": collection.count(),
            "chroma_host": chroma_host,
            "chroma_port": chroma_port,
            "chroma_tenant": chroma_tenant,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection status: {str(e)}",
        )
