from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import json
from typing import Optional, Dict, Any
import logging
import uvicorn
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ollama GPU Server", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    stop: Optional[list] = None

# Get Ollama host from environment or default to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_API_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"
MODEL_NAME = "gpt-oss:20b"

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

def format_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the response from Ollama to match expected output structure."""
    return {
        "text": response_data.get("response", ""),
        "model": response_data.get("model", MODEL_NAME),
        "usage": {
            "prompt_tokens": response_data.get("prompt_eval_count", 0),
            "completion_tokens": response_data.get("eval_count", 0),
            "total_tokens": response_data.get("total_eval_count", 0)
        }
    }

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
        # Don't exit - allow the server to start anyway but endpoints will fail
        # This allows the health check endpoint to provide status

@app.post("/v1/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate text using the Ollama API with optimized settings for gpt-oss-20b.
    """
    try:
        # Prepare the request payload with optimized settings
        payload = {
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repeat_penalty": request.repeat_penalty,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "stop": request.stop or []
            },
            "stream": False
        }

        # Add system prompt if provided
        if request.system_prompt:
            payload["system"] = request.system_prompt

        logger.info(f"Sending request to Ollama API for model {MODEL_NAME}")
        
        # Make request to Ollama API
        response = http.post(
            f"{OLLAMA_API_BASE}/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse and format response
        response_data = response.json()
        formatted_response = format_response(response_data)
        
        # Log completion for monitoring
        background_tasks.add_task(
            logger.info,
            f"Generated {formatted_response['usage']['completion_tokens']} tokens"
        )
        
        return formatted_response

    except requests.RequestException as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Error communicating with Ollama API: {str(e)}\n"
                f"Please ensure Ollama is running at {OLLAMA_API_BASE}\n"
                "Installation instructions:\n"
                "1. curl -fsSL https://ollama.com/install.sh | sh\n"
                "2. systemctl start ollama (or 'ollama serve' if not using systemd)\n"
                f"3. ollama pull {MODEL_NAME}"
            )
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
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