# ─── Imports ────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from mangum import Mangum
from rag import init_rag, answer_question

load_dotenv()  # still fine if you keep .env for local runs

app = FastAPI(title="TDS Virtual TA")

# ─── Startup (one‑time) ────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    global RAG_STATE
    try:
        RAG_STATE = init_rag()
        print("RAG index ready.")
    except Exception as e:
        print("Failed to initialise RAG:", e)
        raise


# ─── Schemas ───────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str
    image: str | None = None  # base64 string (optional)


# ─── Routes ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "TDS Virtual TA is running. Visit /docs for API usage."}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/")
async def ask(q: Question):
    try:
        return answer_question(RAG_STATE, q.question, q.image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Vercel entrypoint ─────────────────────────────────────────────────
# Vercel’s Python runtime looks for a top‑level variable named `app`
# or `vercel_app`. We export `vercel_app` so it can serve FastAPI
# in serverless mode.
handler = Mangum(app)
