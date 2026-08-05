import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import neurosleepnet as nsn
from openai import OpenAI

app = FastAPI(title="NeuroSleepNet Chat UI")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── API configuration (set via environment variables) ────────────────────────
# Supports both Ollama Cloud and OpenRouter.
# Ollama Cloud:  PROVIDER_BASE_URL=https://ollama.com/v1  PROVIDER_API_KEY=<ollama key>
# OpenRouter:    PROVIDER_BASE_URL=https://openrouter.ai/api/v1  PROVIDER_API_KEY=<or key>
PROVIDER_BASE_URL = os.environ.get("PROVIDER_BASE_URL", "https://ollama.com/v1")
PROVIDER_API_KEY  = os.environ.get("PROVIDER_API_KEY", "")
MODEL_NAME        = os.environ.get("MODEL_NAME", "nemotron-3-nano:30b")

if not PROVIDER_API_KEY:
    raise RuntimeError(
        "PROVIDER_API_KEY environment variable is not set. "
        "Export it before starting the server."
    )

print(f"Initializing client -> {PROVIDER_BASE_URL}  model={MODEL_NAME}")
base_client = OpenAI(base_url=PROVIDER_BASE_URL, api_key=PROVIDER_API_KEY)

# Initialize memory in a specific DB
db_path = "chat_memory.db"
client = nsn.init(base_client, namespace="nsn_chat_agent", db_path=db_path)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/generate", response_model=ChatResponse)
async def generate(request: ChatRequest):
    """Stateless generation — no NSN memory injection."""
    try:
        response = base_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.message}]
        )
        return ChatResponse(reply=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Memory-augmented chat — NSN automatically injects relevant past context."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.message}]
        )
        return ChatResponse(reply=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
