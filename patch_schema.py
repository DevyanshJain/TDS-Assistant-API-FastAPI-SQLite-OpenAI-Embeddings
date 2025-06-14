import sqlite3

conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# Define missing columns (add only if they don't exist)
columns_to_add = {
    "thread_id": "INTEGER",
    "post_id": "INTEGER",
    "author": "TEXT",
    "created_at": "TEXT",
    "url": "TEXT",
    "chunk_index": "INTEGER",
    "embedding": "BLOB"  # Add if missing or wrong type
}

# Fetch current columns in discourse_chunks
cursor.execute("PRAGMA table_info(discourse_chunks)")
existing_columns = {row[1] for row in cursor.fetchall()}

# Add missing columns
for column_name, column_type in columns_to_add.items():
    if column_name not in existing_columns:
        alter_query = f"ALTER TABLE discourse_chunks ADD COLUMN {column_name} {column_type}"
        print(f"Adding missing column: {column_name}")
        cursor.execute(alter_query)

conn.commit()
conn.close()
print("Table patched successfully.")
