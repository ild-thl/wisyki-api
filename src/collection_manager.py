"""
Collection management module for wisyki-api.

Handles Chroma collection initialization, population, and management,
including progress logging for long-running embedding operations.

Uses JSONL format exclusively for storing documents with metadata and precomputed embeddings.
Memory-efficient streaming: processes documents in fixed-size batches.
Functional design with focused methods for each task.
"""

import json
import uuid
import gc
from pathlib import Path
from typing import Tuple, Generator, List, Dict, Optional
import time
import shutil


def _read_batch_from_jsonl(
    file_path: Path, batch_size: int = 32
) -> Generator[List[Tuple[int, dict]], None, None]:
    """
    Stream JSONL file in batches of documents.
    Yields: List of (line_number, document) tuples for each batch.
    """
    batch = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        batch.append((line_num, doc))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        continue
        if batch:
            yield batch
    except Exception as e:
        print(f"    ⚠ Error reading {file_path.name}: {e}")


def _filter_valid_docs(
    batch: List[Tuple[int, dict]], force_recompute: bool = False
) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, dict]], int]:
    """
    Separate batch into documents needing embeddings and cached documents.
    Returns: (needs_embedding, has_embedding, skipped_count)
    """
    needs_embedding = []
    has_embedding = []
    skipped = 0

    for line_num, item in batch:
        if not item or "document" not in item:
            skipped += 1
            continue

        doc_text = item.get("document", "").strip()
        if not doc_text:
            skipped += 1
            continue

        if not force_recompute and "embedding" in item:
            has_embedding.append((line_num, item))
        else:
            needs_embedding.append((line_num, item))

    return needs_embedding, has_embedding, skipped


def _prepare_batch_for_upsert(
    file_stem: str, docs: List[Tuple[int, dict]]
) -> Tuple[List[str], List[str], List[dict], List[Optional[List[float]]]]:
    """
    Extract documents, metadata, ids, and embeddings from batch.
    Returns: (ids, documents, metadatas, embeddings)
    """
    ids = [f"{file_stem}_{line_num}_{uuid.uuid4().hex[:8]}" for line_num, _ in docs]
    documents = [item.get("document", "") for _, item in docs]
    metadatas = [item.get("metadata", {}) for _, item in docs]
    embeddings = [item.get("embedding") for _, item in docs]

    return ids, documents, metadatas, embeddings


def _compute_missing_embeddings(
    embedding_function, documents: List[str]
) -> List[List[float]]:
    """Compute embeddings for documents that don't have them."""
    if not documents:
        return []
    return embedding_function.embed_documents(documents)


def _upsert_batch(
    collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    embeddings: List[Optional[List[float]]],
) -> None:
    """Upsert batch to Chroma collection."""
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def _save_embeddings_to_jsonl(
    file_path: Path, embeddings_map: Dict[int, List[float]]
) -> None:
    """
    Rewrite JSONL file with computed embeddings.
    embeddings_map: {line_number: embedding_vector}
    """
    if not embeddings_map:
        return

    output_file = file_path.with_suffix(".jsonl.tmp")
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if line_num in embeddings_map:
                                item["embedding"] = embeddings_map[line_num]
                            json.dump(item, out_f, separators=(",", ":"))
                            out_f.write("\n")
                        except json.JSONDecodeError:
                            out_f.write(line)

        shutil.move(str(output_file), str(file_path))
    except Exception as e:
        print(f"    ⚠ Warning: Could not save embeddings: {e}")
        if output_file.exists():
            output_file.unlink()


def _save_embeddings_batch_to_jsonl(
    file_path: Path, line_nums: List[int], embeddings: List[List[float]]
) -> None:
    """
    Update specific lines in JSONL file with embeddings.
    Properly tracks line numbers including empty lines.
    """
    if not line_nums:
        return

    # Create mapping for this batch
    embeddings_map = dict(zip(line_nums, embeddings))

    output_file = file_path.with_suffix(".jsonl.tmp")
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f):
                    # Process every line, including empty ones
                    try:
                        if line.strip():  # Only parse non-empty lines
                            item = json.loads(line)
                            if line_num in embeddings_map:
                                item["embedding"] = embeddings_map[line_num]
                            json.dump(item, out_f, separators=(",", ":"))
                            out_f.write("\n")
                        else:
                            # Preserve empty lines as-is
                            out_f.write(line)
                    except json.JSONDecodeError:
                        # If line isn't valid JSON, preserve it as-is
                        out_f.write(line)

        shutil.move(str(output_file), str(file_path))
    except Exception as e:
        print(f"    ⚠ Warning: Could not save embeddings batch: {e}")
        if output_file.exists():
            output_file.unlink()


def _populate_collection_from_import(
    client,
    collection,
    embedding_function,
    collection_name: str,
    force_recompute: bool = False,
    save_embeddings: bool = True,
    batch_size: int = 32,
) -> Tuple[int, int]:
    """
    Populate a Chroma collection with skill documents from JSONL files.

    Streams JSONL files in fixed-size batches:
    1. Read batch from file
    2. Filter valid documents
    3. Compute missing embeddings
    4. Upsert to collection
    5. Save embeddings back to file

    Args:
        client: Chroma client
        collection: Chroma collection object
        embedding_function: HuggingFaceInferenceEmbeddings instance
        collection_name: Name of the collection being populated
        force_recompute: If True, recompute all embeddings (default: False)
        save_embeddings: If True, save computed embeddings to JSONL (default: True)
        batch_size: Documents per batch (default: 32)

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
            print(
                f"\n  [{file_idx}/{len(jsonl_files)}] Processing {jsonl_file.name}..."
            )

            file_added = 0
            file_from_cache = 0
            file_skipped = 0
            batch_start_time = time.time()
            batch_num = 0

            # Process file in batches
            for raw_batch in _read_batch_from_jsonl(jsonl_file, batch_size):
                batch_num += 1

                # Filter documents
                needs_embedding, has_embedding, skipped = _filter_valid_docs(
                    raw_batch, force_recompute
                )
                file_skipped += skipped

                # Upsert cached documents (with embeddings)
                if has_embedding:
                    ids, docs, metas, embeddings = _prepare_batch_for_upsert(
                        jsonl_file.stem, has_embedding
                    )
                    try:
                        _upsert_batch(collection, ids, docs, metas, embeddings)
                        file_from_cache += len(has_embedding)
                        total_added += len(has_embedding)
                    except Exception as e:
                        print(f"    ✗ Error upserting cached batch: {e}")
                        raise
                    finally:
                        del ids, docs, metas, embeddings

                # Compute and upsert new documents
                if needs_embedding:
                    docs_to_embed = [
                        item.get("document", "") for _, item in needs_embedding
                    ]

                    try:
                        new_embeddings = _compute_missing_embeddings(
                            embedding_function, docs_to_embed
                        )

                        ids, docs, metas, _ = _prepare_batch_for_upsert(
                            jsonl_file.stem, needs_embedding
                        )
                        _upsert_batch(collection, ids, docs, metas, new_embeddings)

                        file_added += len(needs_embedding)
                        total_added += len(needs_embedding)

                        # Save embeddings immediately after computing (not at end of file)
                        if save_embeddings:
                            batch_line_nums = [
                                line_num for line_num, _ in needs_embedding
                            ]
                            _save_embeddings_batch_to_jsonl(
                                jsonl_file, batch_line_nums, new_embeddings
                            )

                        # Progress
                        print(
                            f"      Batch {batch_num}: {len(needs_embedding)} docs computed, "
                            f"{len(has_embedding)} from cache"
                        )

                    except Exception as e:
                        print(f"    ✗ Error computing embeddings: {e}")
                        raise
                    finally:
                        del (
                            needs_embedding,
                            has_embedding,
                            raw_batch,
                            docs_to_embed,
                            new_embeddings,
                            ids,
                            docs,
                            metas,
                        )
                        gc.collect()

            # File summary
            batch_elapsed = time.time() - batch_start_time
            file_throughput = (
                (file_added + file_from_cache) / batch_elapsed
                if batch_elapsed > 0
                else 0
            )
            print(
                f"    ✓ File complete: {file_added} new + {file_from_cache} cached = {file_added + file_from_cache} total "
                f"({file_throughput:.1f} docs/s)"
            )
            total_skipped += file_skipped
            total_from_cache += file_from_cache

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
