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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-OSS-20B API Server (MXFP4)", version="1.0.0")

# Request models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 10000
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0

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
    
    # Join conversation and add prompt for assistant
    prompt = "\n\n".join(conversation)
    if conversation:
        prompt += "\n\nAssistant:"
    
    return prompt, system_prompt

@app.on_event("startup")
async def startup_event():
    """Check Ollama availability on startup."""
    logger.info(f"Checking Ollama service at {OLLAMA_API_BASE}")
    if not wait_for_ollama():
        logger.error("Failed to connect to Ollama service")
        logger.error(f"Please ensure Ollama is installed and running at {OLLAMA_API_BASE}")
        logger.error("Installation instructions:")
        logger.error("1. curl -fsSL https://ollama.com/install.sh | sh")
        logger.error("2. systemctl start ollama (or 'ollama serve' if not using systemd)")
        logger.error(f"3. ollama pull {MODEL_NAME}")

@app.get("/")
async def index():
    """Root endpoint with API info"""
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
                "temperature": 0.7
            }
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """Handle chat completion requests"""
    try:
        # Convert messages to Ollama format
        messages_dict = [msg.dict() for msg in request.messages]
        prompt, system_prompt = messages_to_prompt(messages_dict)
        
        # Prepare the request payload for Ollama
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
                "frequency_penalty": request.frequency_penalty
            },
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        logger.info(f"Sending request to Ollama API for model {MODEL_NAME}")
        
        # Make request to Ollama API
        response = http.post(
            f"{OLLAMA_API_BASE}/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        # Parse response from Ollama
        ollama_response = response.json()
        generated_text = ollama_response.get("response", "")
        
        # Create OpenAI-compatible response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())
        
        # Calculate token counts (approximate if not provided)
        prompt_tokens = ollama_response.get("prompt_eval_count", len(prompt.split()) * 1.3)
        completion_tokens = ollama_response.get("eval_count", len(generated_text.split()) * 1.3)
        total_tokens = int(prompt_tokens + completion_tokens)
        
        # Format response in OpenAI format
        openai_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens
            }
        }
        
        # Log completion for monitoring
        background_tasks.add_task(
            logger.info,
            f"Generated {completion_tokens} tokens for completion {completion_id}"
        )
        
        return openai_response
        
    except requests.RequestException as e:
        logger.error(f"Ollama API error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Error communicating with Ollama API: {str(e)}",
                    "type": "internal_error",
                    "code": 500
                }
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500
                }
            }
        )

@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI format"""
    try:
        response = http.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        ollama_models = response.json().get("models", [])
        
        # Check if our model is available
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
        
        return {
            "object": "list",
            "data": models_list
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Error listing models: {str(e)}",
                    "type": "internal_error",
                    "code": 500
                }
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and model status."""
    try:
        response = http.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
        model_loaded = any(m.get("name") == MODEL_NAME for m in models)
        
        status_info = {
            "status": "healthy" if (ollama_status == "healthy" and model_loaded) else "unhealthy",
            "ollama_service": {
                "status": ollama_status,
                "url": OLLAMA_API_BASE
            },
            "model": {
                "name": MODEL_NAME,
                "id": MODEL_ID,
                "loaded": model_loaded
            }
        }
        
        if not model_loaded:
            status_info["help"] = f"Model {MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}"
            
        return status_info
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "ollama_service": {
                "status": "unhealthy",
                "url": OLLAMA_API_BASE
            },
            "help": "Ensure Ollama is installed and running:\n"
                   "1. curl -fsSL https://ollama.com/install.sh | sh\n"
                   "2. systemctl start ollama (or 'ollama serve' if not using systemd)"
        }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # Single worker for GPU optimization
        log_level="info"
    )