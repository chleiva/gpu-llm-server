from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import json
from typing import Optional, Dict, Any
import logging
import uvicorn

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

OLLAMA_API_BASE = "http://localhost:11434/api"
MODEL_NAME = "gpt-oss:20b"

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
        response = requests.post(
            f"{OLLAMA_API_BASE}/generate",
            json=payload,
            timeout=60  # Add timeout to prevent hanging
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
            status_code=500,
            detail=f"Error communicating with Ollama API: {str(e)}"
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
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return {
            "status": "healthy",
            "model_loaded": any(m.get("name") == MODEL_NAME for m in models)
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU optimization
        log_level="info"
    )
