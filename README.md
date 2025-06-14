# 📚 TDS Assistant API – FastAPI + SQLite + OpenAI Embeddings

This is an intelligent assistant built using FastAPI to answer student queries based on:

- ✅ Scraped course content (`markdown_chunks`)
- ✅ Scraped Discourse posts (`discourse_chunks`)
- ✅ Precomputed embeddings using OpenAI-compatible endpoint (`text-embedding-3-small` via [aipipe.org](https://aipipe.org))

---

## 🚀 Deployment on Railway

> ✨ Zero-config Python hosting for FastAPI + SQLite apps

---

This repo includes a script (discourseScrapper.py) that scrapes Discourse threads across a date range and saves them to JSON format.

---

### 🧑‍💻 1. Prerequisites

- GitHub account
- Railway account: [https://railway.app](https://railway.app)
- Your app pushed to a GitHub repo

---

### 🔧 2. Railway Setup

#### Step 1: Create a Railway Project

- Go to [https://railway.app](https://railway.app)
- Click **New Project** → **Deploy from GitHub Repo**
- Select your FastAPI project

#### Step 2: Add Environment Variable

In Railway → `Variables` tab:
Add API_KEY=your_aipipe_or_openai_key_here


#### Step 3: Define the Start Command

In Railway → `Deployments` → Set build and start commands:

```bash
Start command:
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 🧠 3. Embedding Data (One-Time)

Before deploying, make sure you've:

- Run generate_embeddings.py locally (it updates knowledge_base.db)

- Committed the final version of knowledge_base.db to GitHub

```bash
python generate_embeddings.py
```

### ✅ 4. After Deployment

Visit your public URL:
[https://your-app.up.railway.app/docs](https://your-app.up.railway.app/docs)

### KNOWLEDGE_BASE CAN BE FOUND ON THE BELOW LINK

[Knowledge_base.db](https://drive.google.com/file/d/1HZDBs6ka43tjebqwUyPy5LPf_sGBvqzy/view?usp=sharing)
