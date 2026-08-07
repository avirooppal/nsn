import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import neurosleepnet as nsn
from openai import OpenAI

app = FastAPI(title="NeuroSleepNet + Ollama Cloud Studio")

# Mount the static directory for the web frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── API Configuration Defaults ───────────────────────────────────────────────
DEFAULT_BASE_URL = os.environ.get("PROVIDER_BASE_URL", "https://ollama.com/v1")
DEFAULT_API_KEY  = os.environ.get("PROVIDER_API_KEY", os.environ.get("OLLAMA_API_KEY", ""))
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron-3-nano:30b")
DB_PATH = "chat_memory.db"

def get_clients(user_api_key: Optional[str] = None, user_base_url: Optional[str] = None):
    """Helper to instantiate base OpenAI-compatible client & NSN wrapper."""
    api_key = user_api_key.strip() if user_api_key and user_api_key.strip() else DEFAULT_API_KEY
    base_url = user_base_url.strip() if user_base_url and user_base_url.strip() else DEFAULT_BASE_URL
    
    base_client = OpenAI(base_url=base_url, api_key=api_key)
    nsn_client = nsn.init(base_client, namespace="ollama_web_agent", db_path=DB_PATH, recall_limit=5)
    return base_client, nsn_client


class ChatRequest(BaseModel):
    message: str
    mode: str = "nsn" # "vanilla" or "nsn"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    mode: str
    recalled_memories: List[dict] = []

class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    selected_model = request.model or DEFAULT_MODEL_NAME
    base_client, nsn_client = get_clients(request.api_key, request.base_url)
    
    if request.mode == "vanilla":
        # Pure Stateless Vanilla LLM Call — No Memory Layer
        try:
            response = base_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": request.message}]
            )
            reply_text = response.choices[0].message.content
            return ChatResponse(
                reply=reply_text,
                mode="vanilla",
                recalled_memories=[]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vanilla Model Error: {str(e)}")
            
    else:
        # NSN Memory Layer Augmented Call
        try:
            # 1. Check recalled memories prior to completion for UI display
            recalled = nsn_client.recall(request.message, limit=5)
            
            # 2. Let NSN execute memory-augmented completion
            response = nsn_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": request.message}]
            )
            reply_text = response.choices[0].message.content
            
            return ChatResponse(
                reply=reply_text,
                mode="nsn",
                recalled_memories=recalled
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NSN Model Error: {str(e)}")


@app.get("/api/memories")
async def get_memories():
    """Retrieve timeline of memories stored in NSN."""
    try:
        _, nsn_client = get_clients()
        memories = nsn_client.timeline(limit=50)
        return {"memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memories/search")
async def search_memories(req: MemorySearchRequest):
    """Search NSN memory database."""
    try:
        _, nsn_client = get_clients()
        results = nsn_client.recall(req.query, limit=req.limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_memory():
    """Reset NSN memory database for testing."""
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        # Re-initialize
        get_clients()
        return {"status": "success", "message": "Memory database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sleep")
async def trigger_sleep():
    """Trigger NSN NREM/REM memory consolidation."""
    try:
        _, nsn_client = get_clients()
        success = nsn_client.sleep()
        return {"status": "success" if success else "no_op", "message": "Sleep consolidation executed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models(x_api_key: Optional[str] = Header(None)):
    """List available models using default or user-provided API key."""
    try:
        base_client, _ = get_clients(user_api_key=x_api_key)
        models_data = base_client.models.list().data
        model_ids = [m.id for m in models_data]
        return {"models": model_ids}
    except Exception as e:
        return {"models": [DEFAULT_MODEL_NAME, "gemma4:31b", "qwen3.5:397b", "deepseek-v4-pro"]}
