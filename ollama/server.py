from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import json
from typing import Optional, Dict, Any, List
import logging
import uvicorn
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="GPT-OSS-20B API Server (MXFP4)", version="1.0.0")

# Request models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 50000
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    # NEW: pass-through reasoning effort
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"

# Get Ollama host from environment or default to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_API_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"
MODEL_NAME = "gpt-oss:20b"
MODEL_ID = "gpt-oss-20b"  # ID used in the API

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

def create_log_entry(
    request_id: str,
    prompt: str,
    response: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency: float,
    model: str,
    temperature: float,
    max_tokens: int,
    messages: List[Dict[str, str]],
    timestamp: datetime,
    status: str = "success",
    error: str = None,
    reasoning_effort: str = "low"  # NEW: log it
) -> Dict[str, Any]:
    """Create a structured log entry."""
    return {
        "request_id": request_id,
        "timestamp": timestamp.isoformat(),
        "model": model,
        "status": status,
        "latency_seconds": round(latency, 3),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens
        },
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort  # NEW
        },
        "messages": messages,
        "prompt": prompt,
        "model_response": response,
        "error": error
    }

def write_log_entry(log_entry: Dict[str, Any]):
    """Write log entry to timestamped file."""
    timestamp = datetime.now()
    log_filename = timestamp.strftime("%Y%m%d_%H%M%S") + ".log"
    log_filepath = LOGS_DIR / log_filename
    try:
        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, indent=2, ensure_ascii=False))
            f.write("\n" + "="*80 + "\n\n")
    except Exception as e:
        logger.error(f"Failed to write log: {str(e)}")

def print_request_summary(
    request_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency: float,
    status: str = "success"
):
    """Print minimal summary to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | in: {prompt_tokens} | out: {completion_tokens} | latency: {latency:.3f}s")

def wait_for_ollama(timeout=60):
    """Wait for Ollama service to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = http.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama service")
                return True
        except requests.RequestException as e:
            logger.warning(f"Waiting for Ollama service... ({str(e)})")
        time.sleep(2)
    return False

def messages_to_prompt(messages: List[Dict[str, str]]) -> tuple[str, str]:
    """Convert OpenAI messages format to Ollama prompt format."""
    system_prompt = ""
    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            conversation.append(f"User: {content}")
        elif role == "assistant":
            conversation.append(f"Assistant: {content}")
    prompt = "\n\n".join(conversation)
    if conversation:
        prompt += "\n\nAssistant:"
    return prompt, system_prompt

def normalize_reasoning_effort(val: Optional[str]) -> str:
    """Validate and normalize reasoning_effort, default to 'low'."""
    if val is None:
        return "low"
    v = str(val).strip().lower()
    if v not in {"low", "medium", "high"}:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Invalid reasoning_effort. Use 'low', 'medium', or 'high'.",
                              "type": "invalid_request_error", "code": 400}}
        )
    return v

@app.on_event("startup")
async def startup_event():
    logger.info(f"Checking Ollama service at {OLLAMA_API_BASE}")
    logger.info(f"Logs will be written to: {LOGS_DIR.absolute()}")
    if not wait_for_ollama():
        logger.error("Failed to connect to Ollama service")
        logger.error(f"Please ensure Ollama is installed and running at {OLLAMA_API_BASE}")
        logger.error("Installation instructions:")
        logger.error("1. curl -fsSL https://ollama.com/install.sh | sh")
        logger.error("2. systemctl start ollama (or 'ollama serve' if not using systemd)")
        logger.error(f"3. ollama pull {MODEL_NAME}")

@app.get("/")
async def index():
    return {
        "name": "GPT-OSS-20B API Server (MXFP4)",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions (OpenAI compatible)",
            "/v1/models": "GET - List models",
            "/health": "GET - Health check",
            "/": "GET - This page"
        },
        "example": {
            "url": "/v1/chat/completions",
            "method": "POST",
            "body": {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
                "temperature": 0.7,
                "reasoning_effort": "high"
            }
        },
        "logs_directory": str(LOGS_DIR.absolute())
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    timestamp = datetime.now()

    try:
        # Validate and normalize reasoning effort
        reasoning_effort = normalize_reasoning_effort(request.reasoning_effort)

        # Convert messages
        messages_dict = [msg.dict() for msg in request.messages]
        prompt, system_prompt = messages_to_prompt(messages_dict)

        # Build Ollama payload
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "stop": request.stop or [],
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                # NEW: pass through to downstream; Ollama will ignore if unknown
                "reasoning_effort": reasoning_effort
            },
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = http.post(
            f"{OLLAMA_API_BASE}/generate",
            json=payload,
            timeout=1000
        )
        response.raise_for_status()

        latency = time.time() - start_time

        ollama_response = response.json()
        generated_text = ollama_response.get("response", "")

        prompt_tokens = int(ollama_response.get("prompt_eval_count", len(prompt.split()) * 1.3))
        completion_tokens = int(ollama_response.get("eval_count", len(generated_text.split()) * 1.3))
        total_tokens = prompt_tokens + completion_tokens

        log_entry = create_log_entry(
            request_id=request_id,
            prompt=prompt,
            response=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
            model=MODEL_ID,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            messages=messages_dict,
            timestamp=timestamp,
            status="success",
            reasoning_effort=reasoning_effort
        )
        background_tasks.add_task(write_log_entry, log_entry)
        background_tasks.add_task(
            print_request_summary, request_id, prompt_tokens, completion_tokens, latency, "success"
        )

        created_timestamp = int(timestamp.timestamp())
        openai_response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            # NEW: echo back for clients expecting this metadata
            "reasoning": {"effort": reasoning_effort}
        }
        return openai_response

    except requests.RequestException as e:
        latency = time.time() - start_time
        error_msg = f"Ollama API error: {str(e)}"
        logger.error(f"Request {request_id}: {error_msg}")

        messages_dict = [msg.dict() for msg in request.messages]
        prompt, _ = messages_to_prompt(messages_dict)

        log_entry = create_log_entry(
            request_id=request_id,
            prompt=prompt,
            response="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency=latency,
            model=MODEL_ID,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            messages=messages_dict,
            timestamp=datetime.now(),
            status="error",
            error=error_msg,
            reasoning_effort=normalize_reasoning_effort(request.reasoning_effort)
        )
        background_tasks.add_task(write_log_entry, log_entry)
        background_tasks.add_task(print_request_summary, request_id, 0, 0, latency, "error")

        return JSONResponse(
            status_code=500,
            content={"error": {"message": error_msg, "type": "internal_error", "code": 500}}
        )

    except HTTPException as e:
        # Invalid reasoning_effort or similar validation error
        return JSONResponse(status_code=e.status_code, content=e.detail)

    except Exception as e:
        latency = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Request {request_id}: {error_msg}")

        messages_dict = [msg.dict() for msg in request.messages]
        prompt, _ = messages_to_prompt(messages_dict)

        log_entry = create_log_entry(
            request_id=request_id,
            prompt=prompt,
            response="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency=latency,
            model=MODEL_ID,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            messages=messages_dict,
            timestamp=datetime.now(),
            status="error",
            error=error_msg,
            reasoning_effort=normalize_reasoning_effort(request.reasoning_effort)
        )
        background_tasks.add_task(write_log_entry, log_entry)
        background_tasks.add_task(print_request_summary, request_id, 0, 0, latency, "error")

        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error", "code": 500}}
        )

@app.get("/v1/models")
async def list_models():
    try:
        response = http.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        ollama_models = response.json().get("models", [])
        model_available = any(m.get("name") == MODEL_NAME for m in ollama_models)
        models_list = []
        if model_available:
            models_list.append({
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "permission": [],
                "root": MODEL_ID,
                "parent": None
            })
        return {"object": "list", "data": models_list}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Error listing models: {str(e)}",
                               "type": "internal_error", "code": 500}}
        )

@app.get("/health")
async def health_check():
    try:
        response = http.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
        model_loaded = any(m.get("name") == MODEL_NAME for m in models)

        status_info = {
            "status": "healthy" if (ollama_status == "healthy" and model_loaded) else "unhealthy",
            "ollama_service": {"status": ollama_status, "url": OLLAMA_API_BASE},
            "model": {"name": MODEL_NAME, "id": MODEL_ID, "loaded": model_loaded},
            "logs_directory": str(LOGS_DIR.absolute())
        }
        if not model_loaded:
            status_info["help"] = f"Model {MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}"
        return status_info
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "ollama_service": {"status": "unhealthy", "url": OLLAMA_API_BASE},
            "logs_directory": str(LOGS_DIR.absolute()),
            "help": "Ensure Ollama is installed and running:\n"
                    "1. curl -fsSL https://ollama.com/install.sh | sh\n"
                    "2. systemctl start ollama (or 'ollama serve' if not using systemd)"
        }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info"
    )
