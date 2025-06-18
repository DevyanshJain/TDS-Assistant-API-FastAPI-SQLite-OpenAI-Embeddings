from __future__ import annotations

import base64
import json
import os
import sqlite3
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

# ─── Env & API config ──────────────────────────────────────────────────
load_dotenv()

AIPIPE_KEY   = os.getenv("AIPIPE_API_KEY")
BASE_URL     = os.getenv("AIPIPE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME   = os.getenv("CHAT_MODEL", "openai/gpt-4.1-nano")
DEBUG        = bool(int(os.getenv("RAG_DEBUG", "0")))

API_URL = f"{BASE_URL}/chat/completions"
HEADERS = {"Authorization": f"Bearer {AIPIPE_KEY}", "Content-Type": "application/json"}

# ─── Retrieval assets ──────────────────────────────────────────────────
DB_PATH     = Path("knowledge_base.db")
INDEX_BIN   = Path("faiss.index")
ID_MAP_JSON = Path("faiss_ids.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K       = 6

# ─── Init ──────────────────────────────────────────────────────────────
def init_rag() -> dict:
    if not (INDEX_BIN.exists() and ID_MAP_JSON.exists()):
        raise RuntimeError("FAISS index or ID map missing – run embed_local.py first.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    index    = faiss.read_index(str(INDEX_BIN))
    id_map   = json.loads(ID_MAP_JSON.read_text())
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    return {"db": conn, "index": index, "id_map": id_map, "embed": embedder}


# ─── Retrieval helper ──────────────────────────────────────────────────
def _retrieve(state: dict, query: str) -> List[sqlite3.Row]:
    q_vec = np.array(list(state["embed"].embed([query]))[0], dtype="float32")[None, :]
    _, I = state["index"].search(q_vec, TOP_K)

    if I is None or I[0].size == 0:
        return []

    ids = [state["id_map"][str(i)] for i in I[0]]

    placeholders = ','.join('?' * len(ids))

    cur = state["db"].execute(
        f"""
        SELECT id, content AS text, original_url AS source_url
        FROM markdown_chunks WHERE id IN ({placeholders})
        UNION ALL
        SELECT id, content AS text, url AS source_url
        FROM discourse_chunks WHERE id IN ({placeholders})
        """,
        ids * 2  # Provide parameters for both queries
    )

    return cur.fetchall()



# ─── Image helper ──────────────────────────────────────────────────────
def _handle_image(image_b64: str) -> str:
    """Decode and save the base‑64 image, return a note for the answer."""
    try:
        data = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as f:
            f.write(data)
            tmp_path = f.name
        if DEBUG:
            sys.stderr.write(f"📷  Saved uploaded image to {tmp_path}\n")
        return "Image received (current version does not analyze images)."
    except Exception as e:
        return f"Image could not be processed: {e}"


# ─── AIPipe call ───────────────────────────────────────────────────────
def _ask_ai_pipe(prompt: str) -> str:
    if not AIPIPE_KEY:
        raise RuntimeError("AIPIPE_API_KEY is missing")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful TA for IIT‑M’s Tools in Data Science course."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=25)
        if DEBUG:
            sys.stderr.write(f"Status: {resp.status_code}\n")
            sys.stderr.write(f"Raw: {resp.text[:800]}\n")
        resp.raise_for_status()
        out = resp.json()

        if "choices" in out and "message" in out["choices"][0]:
            return out["choices"][0]["message"]["content"].strip()
        if "result" in out:  # alternate schema
            return out["result"].strip()

        raise ValueError("Unexpected AIPipe JSON format.")

    except Exception as e:
        sys.stderr.write(f"AI Pipe error: {e}\n")
        raise


# ─── Public API function ───────────────────────────────────────────────
def answer_question(state: dict, question: str, image: str | None = None) -> dict:
    passages = _retrieve(state, question)

    # Construct link list with short snippets
    links = [
    {"url": p["source_url"], "text": textwrap.shorten(p["text"], width=120, placeholder="…")}
    for p in passages]


    if not passages:
        return {"answer": "I couldn't find any relevant documents.", "links": links}

    context = "\n\n".join(p["text"] for p in passages)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in 3–4 sentences. Cite passage numbers if helpful."
    )

    try:
        answer = _ask_ai_pipe(prompt)
    except Exception:
        # fallback to raw context if LLM call fails
        answer = ("Sorry, I had trouble generating a concise answer. "
                  "Here are relevant passages:\n\n---\n\n" +
                  context[:1500])

    # if an image was sent, acknowledge it
    if image:
        answer += "\n\n" + _handle_image(image)

    return {"answer": answer, "links": links}