#!/usr/bin/env python
"""
Generate local embeddings for markdown_chunks + discourse_chunks
and write them into a FAISS index (faiss.index) — no Internet needed.
"""

from __future__ import annotations
import sqlite3, pathlib, json, numpy as np
from fastembed import TextEmbedding
import faiss, tqdm, gc

# File paths and model
DB_PATH    = pathlib.Path("knowledge_base.db")
INDEX_PATH = pathlib.Path("faiss.index")
ID_MAP     = pathlib.Path("faiss_ids.json")
MODEL      = "BAAI/bge-small-en-v1.5"

# Batch sizes
EMBED_BATCH = 8         # Texts per embedding batch
EMBED_SUB_BATCH = 2     # Real embedding sub-batch

def load_rows():
    """Load rows from markdown_chunks and discourse_chunks where text is not NULL"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT id, content FROM markdown_chunks
        UNION ALL
        SELECT id, content FROM discourse_chunks
        WHERE content IS NOT NULL
    """
    try:
        cursor.execute(query)
        return list(cursor.fetchall())
    finally:
        conn.close()

def batch(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def main():
    print("📦 Loading rows from SQLite...")
    all_rows = load_rows()
    if not all_rows:
        print("❌ No rows found to embed. Exiting.")
        return

    ids, texts = zip(*all_rows)
    print(f"🔢 Total rows to embed: {len(texts)}")

    embedder = TextEmbedding(model_name=MODEL)
    vectors = []

    print("🧠 Generating embeddings...")
    for i, text_batch in enumerate(tqdm.tqdm(batch(texts, EMBED_BATCH), total=(len(texts) + EMBED_BATCH - 1)//EMBED_BATCH, desc="Embedding")):
        try:
            batch_vecs = list(embedder.embed(text_batch, batch_size=EMBED_SUB_BATCH))
            vectors.extend(batch_vecs)
        except Exception as e:
            print(f"[!] Failed at batch {i}: {e}")
            continue
        gc.collect()

    if not vectors:
        print("❌ No vectors generated. Exiting.")
        return

    print("🧊 Building FAISS index...")
    vecs = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✅ FAISS index saved to → {INDEX_PATH}")

    # Save mapping of FAISS position to DB id
    mapping = {i: ids[i] for i in range(len(ids))}
    ID_MAP.write_text(json.dumps(mapping, indent=2))
    print(f"✅ ID mapping saved to → {ID_MAP}")

if __name__ == "__main__":
    main()
