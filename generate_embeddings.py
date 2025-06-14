import asyncio
import sqlite3
import os
import aiohttp
import json
import traceback
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

DB_PATH = "knowledge_base.db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_ENDPOINT = "https://aipipe.org/openai/v1/embeddings"
HEADERS = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}


async def get_embedding(text, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    EMBEDDING_ENDPOINT,
                    headers=HEADERS,
                    json={"model": EMBEDDING_MODEL, "input": text}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["data"][0]["embedding"]
                    elif response.status == 429:
                        print(f"Rate limit hit. Retrying ({retries+1})...")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        print(f"Error: {response.status} --> {error_text}")
                        return None
        except Exception as e:
            print(f"Exception during embedding: {e}")
            print(traceback.format_exc().encode('ascii', 'replace').decode())
            retries += 1
            await asyncio.sleep(3 * retries)
    return None


async def process_table(table_name, id_col="id"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT {id_col}, content FROM {table_name} WHERE embedding IS NULL")
    rows = cursor.fetchall()

    total = len(rows)
    print(f"\n {total} rows to embed in {table_name}...")

    for idx, (row_id, content) in enumerate(rows, 1):
        print(f"[{table_name} {idx}/{total}] Embedding row ID: {row_id} (length: {len(content)})")
        embedding = await get_embedding(content)
        if embedding:
            try:
                cursor.execute(
                    f"UPDATE {table_name} SET embedding = ? WHERE {id_col} = ?",
                    (json.dumps(embedding), row_id)
                )
                conn.commit()
            except Exception as db_error:
                print(f"Failed to update row {row_id}: {db_error}")
        else:
            print(f"Skipped row {row_id} due to embedding error.")

    conn.close()
    print(f"Done embedding {table_name}.\n")


async def main():
    if not API_KEY:
        print("API_KEY not found in .env")
        return

    await process_table("markdown_chunks")
    await process_table("discourse_chunks")


if __name__ == "__main__":
    asyncio.run(main())
