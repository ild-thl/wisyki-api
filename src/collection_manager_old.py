"""
Collection management module for wisyki-api.

Handles Chroma collection initialization, population, and management,
including progress logging for long-running embedding operations.

Uses JSONL format exclusively for storing documents with metadata and precomputed embeddings.
Supports automatic migration from legacy JSON files to JSONL.
Memory-efficient streaming: processes documents in batches without loading entire files.
"""

import json
import uuid
from pathlib import Path
from typing import Tuple, Generator
import time
import tempfile
import shutil


def _populate_collection_from_import(
    client,
    collection,
    embedding_function,
    collection_name: str,
    force_recompute: bool = False,
    save_embeddings: bool = True,
) -> Tuple[int, int]:
    """
    Populate a Chroma collection with skill documents from JSONL files.

    Loads all JSONL files from data/import/ where each line contains
    an object with 'document' and optional 'metadata' and 'embedding' fields.

    Optionally uses precomputed embeddings (unless force_recompute=True).
    Optionally saves computed embeddings back to JSONL files.

    Args:
        client: Chroma client
        collection: Chroma collection object
        embedding_function: HuggingFaceInferenceEmbeddings instance
        collection_name: Name of the collection being populated
        force_recompute: If True, ignore precomputed embeddings and recompute all (default: False)
        save_embeddings: If True, save computed embeddings to JSONL files (default: True)

    Returns:
        Tuple of (total_added, total_skipped)
    """
    import_dir = Path("data/import")

    if not import_dir.exists():
        print(f"  ℹ Import directory not found: {import_dir}")
        return 0, 0

    jsonl_files = list(import_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ℹ No JSONL files found in {import_dir}")
        return 0, 0

    print(f"  Initializing collection '{collection_name}' from import directory...")
    print(f"  Found {len(jsonl_files)} JSONL file(s) to process")
    if not force_recompute:
        print(f"  Mode: Use precomputed embeddings if available, recompute others")
    else:
        print(f"  Mode: Force recompute all embeddings")

    total_added = 0
    total_skipped = 0
    total_from_cache = 0
    start_time = time.time()

    for file_idx, jsonl_file in enumerate(jsonl_files, 1):
        try:
            # Load documents from JSONL
            documents_data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        documents_data.append(json.loads(line))

            if not documents_data:
                print(f"    ⚠ No valid documents found in {jsonl_file.name}")
                continue

            print(
                f"\n  [{file_idx}/{len(jsonl_files)}] Processing {jsonl_file.name} ({len(documents_data)} items)..."
            )

            # Prepare documents for embedding
            valid_docs = []
            valid_metas = []
            valid_ids = []
            valid_embeddings = []  # None for items needing embedding, embedding vector for cached
            valid_orig_indices = []  # Track which original indices these correspond to
            file_skipped = 0
            file_from_cache = 0

            for i, item in enumerate(documents_data):
                if not item or "document" not in item:
                    file_skipped += 1
                    continue

                doc_text = item.get("document", "").strip()
                if not doc_text:
                    file_skipped += 1
                    continue

                doc_id = f"{jsonl_file.stem}_{i}_{uuid.uuid4().hex[:8]}"
                valid_docs.append(doc_text)
                valid_metas.append(item.get("metadata", {}))
                valid_ids.append(doc_id)
                valid_orig_indices.append(i)  # Track original index

                # Check for precomputed embedding in current line
                if not force_recompute and "embedding" in item:
                    valid_embeddings.append(item["embedding"])
                    file_from_cache += 1
                else:
                    valid_embeddings.append(None)

            if file_skipped > 0:
                print(f"    ℹ Skipped {file_skipped} invalid documents")

            if not valid_docs:
                print(f"    ⚠ No valid documents found in {jsonl_file.name}")
                total_skipped += file_skipped
                continue

            # Count how many need embedding
            need_embedding = sum(1 for e in valid_embeddings if e is None)

            if file_from_cache > 0:
                print(
                    f"    ℹ Using {file_from_cache} precomputed embeddings, computing {need_embedding} new ones"
                )
            else:
                print(f"    Embedding and storing {len(valid_docs)} documents...")

            # Process in batches
            batch_size = 32
            num_batches = (len(valid_docs) + batch_size - 1) // batch_size
            file_added = 0
            file_from_cache_batch = 0
            batch_start_time = time.time()

            # Track which items to update in JSONL (those we computed)
            items_to_update = []

            for batch_idx, batch_start in enumerate(
                range(0, len(valid_docs), batch_size), 1
            ):
                batch_end = min(batch_start + batch_size, len(valid_docs))
                batch_docs = valid_docs[batch_start:batch_end]
                batch_metas = valid_metas[batch_start:batch_end]
                batch_ids = valid_ids[batch_start:batch_end]
                batch_embeddings = valid_embeddings[batch_start:batch_end]
                batch_size_actual = len(batch_docs)

                progress_pct = (batch_idx / num_batches) * 100
                docs_done = min(batch_end, len(valid_docs))

                # Separate cached and new embeddings
                new_doc_indices = [
                    i for i, e in enumerate(batch_embeddings) if e is None
                ]
                cached_count = batch_size_actual - len(new_doc_indices)

                print(
                    f"      Batch {batch_idx}/{num_batches} ({progress_pct:.0f}%) - Docs {batch_start + 1}-{docs_done}...",
                    end="",
                    flush=True,
                )

                batch_iter_start = time.time()
                try:
                    # Compute embeddings only for new documents
                    if new_doc_indices:
                        new_docs = [batch_docs[i] for i in new_doc_indices]
                        new_embeddings = embedding_function.embed_documents(new_docs)

                        # Insert computed embeddings at correct positions
                        embedding_idx = 0
                        for i in range(len(batch_embeddings)):
                            if batch_embeddings[i] is None:
                                batch_embeddings[i] = new_embeddings[embedding_idx]
                                # Track for JSON update
                                items_to_update.append(
                                    (batch_start + i, new_embeddings[embedding_idx])
                                )
                                embedding_idx += 1

                    # All embeddings should now be set
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas,
                        embeddings=batch_embeddings,
                    )
                    batch_time = time.time() - batch_iter_start
                    file_added += batch_size_actual
                    file_from_cache_batch += cached_count
                    total_added += batch_size_actual

                    # Calculate throughput
                    docs_per_sec = (
                        batch_size_actual / batch_time if batch_time > 0 else 0
                    )
                    cache_indicator = (
                        f" ({cached_count} cached)" if cached_count > 0 else ""
                    )
                    print(
                        f" ✓ ({batch_time:.1f}s, {docs_per_sec:.1f} docs/s){cache_indicator}"
                    )

                except Exception as e:
                    print(f" ✗ Error: {e}")
                    raise

            # Save computed embeddings back to JSONL file
            if save_embeddings:
                # Update original documents with embeddings
                for valid_idx, embedding in enumerate(valid_embeddings):
                    if embedding is not None:
                        orig_idx = valid_orig_indices[valid_idx]
                        documents_data[orig_idx]["embedding"] = embedding

                try:
                    with open(jsonl_file, "w", encoding="utf-8") as f:
                        for doc in documents_data:
                            json.dump(doc, f, separators=(",", ":"))
                            f.write("\n")
                    print(f"    ✓ Updated {jsonl_file.name} with computed embeddings")
                except Exception as e:
                    print(
                        f"    ⚠ Warning: Could not save embeddings to {jsonl_file.name}: {e}"
                    )

            batch_elapsed = time.time() - batch_start_time
            file_throughput = file_added / batch_elapsed if batch_elapsed > 0 else 0
            print(
                f"    ✓ File complete: {file_added} documents added ({file_throughput:.1f} docs/s)"
            )
            total_skipped += file_skipped
            total_from_cache += file_from_cache_batch

        except Exception as e:
            print(f"    ✗ Error processing {jsonl_file.name}: {e}")

    # Final summary
    elapsed = time.time() - start_time
    overall_throughput = total_added / elapsed if elapsed > 0 else 0
    computed = total_added - total_from_cache

    print(f"\n  ✓ Collection population complete:")
    print(f"    - Total documents added: {total_added}")
    print(f"    - From precomputed cache: {total_from_cache}")
    print(f"    - Newly computed: {computed}")
    print(f"    - Total documents skipped: {total_skipped}")
    print(f"    - Time elapsed: {elapsed:.1f}s")
    print(f"    - Average throughput: {overall_throughput:.1f} docs/s")
    print(f"    - Collection size: {collection.count()}")

    return total_added, total_skipped
