import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama
import logging
from typing import Optional

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class
class ModelConfig(BaseModel):
    model_path: str = "./models/mistral-small-3.2-24b-instruct-2506-q8_0.gguf"
    n_gpu_layers: int = -1  # Use all GPU layers
    n_ctx: int = 8192  # Context window
    n_batch: int = 512  # Batch size
    verbose: bool = False

def load_config(config_path: str = "model_config.json") -> ModelConfig:
    """Load configuration from JSON file with fallbacks"""
    config_file = Path(config_path)
    
    if config_file.exists():
        logger.info(f"Loading config from {config_path}")
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return ModelConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}. Using defaults.")
            return ModelConfig()
    else:
        logger.info(f"Config file {config_path} not found. Using defaults.")
        return ModelConfig()

# Load configuration
config = load_config()

# Request/Response models (keeping your existing structure)
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.5, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

class InferenceResponse(BaseModel):
    response: str
    model_info: dict

class ModelInfo(BaseModel):
    model_name: str
    quantized: bool
    device_map: str
    local_files_only: bool

# Global model variables
llm = None
model_info = {}

def load_model_once():
    """Load GGUF model once at startup"""
    global llm, model_info
    
    if llm is not None:
        return
    
    model_path = config.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please download it first.")
    
    logger.info(f"Loading GGUF model from: {model_path}")
    logger.info(f"GPU layers: {config.n_gpu_layers}")
    logger.info(f"Context window: {config.n_ctx}")
    logger.info(f"Batch size: {config.n_batch}")
    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            verbose=config.verbose
        )
        
        # Store model info
        model_info = {
            "model_name": os.path.basename(model_path),
            "model_type": "GGUF",
            "quantization": "Q8_0",
            "quantized": True,
            "device_map": "GPU" if config.n_gpu_layers != 0 else "CPU",
            "local_files_only": True,
            "gpu_layers": config.n_gpu_layers,
            "context_window": config.n_ctx
        }
        
        logger.info("GGUF model loaded successfully!")
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# FastAPI app
app = FastAPI(
    title="GGUF LLM Inference API",
    description="Inference server for GGUF format models with GPU acceleration",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_once()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": llm is not None,
        "model_info": model_info
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_info["model_name"],
        quantized=model_info["quantized"],
        device_map=model_info["device_map"],
        local_files_only=model_info["local_files_only"]
    )

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Inference endpoint"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt for Mistral (adjust if needed)
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Generate response
        output = llm(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repeat_penalty=request.repetition_penalty,
            echo=False,
            stop=["</s>", "[/INST]", "<|endoftext|>"]
        )
        
        response_text = output['choices'][0]['text'].strip()
        
        return InferenceResponse(
            response=response_text,
            model_info=model_info
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)