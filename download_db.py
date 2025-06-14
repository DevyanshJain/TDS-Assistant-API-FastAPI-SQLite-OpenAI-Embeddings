import os
import requests

DB_URL = "https://drive.google.com/uc?export=download&id=1HZDBs6ka43tjebqwUyPy5LPf_sGBvqzy"
DB_PATH = "knowledge_base.db"

def download_if_needed():
    if not os.path.exists(DB_PATH):
        print("Downloading knowledge_base.db from Google Drive...")
        with requests.get(DB_URL, stream=True) as r:
            r.raise_for_status()
            with open(DB_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("DB downloaded.")
    else:
        print("DB already exists.")

# Run only if script is imported from elsewhere (like app.py)
if __name__ != "__main__":
    download_if_needed()
