import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from typing import Optional, Dict, Any

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class
class ModelConfig(BaseModel):
    model_name_or_path: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    model_cache_dir: Optional[str] = None
    use_local_files_only: bool = False
    use_quantization: bool = True
    quantization_bits: int = Field(default=8, ge=4, le=8)
    device_map: str = "auto"

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

# Request/Response models
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
model = None
tokenizer = None
model_info = {}

def get_quantization_config():
    """Create quantization config based on settings"""
    if not config.use_quantization:
        return None
    
    if config.quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
    elif config.quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        logger.warning(f"Unsupported quantization bits: {config.quantization_bits}. Disabling quantization.")
        return None

def load_model_once():
    """Load model once at startup"""
    global model, tokenizer, model_info
    
    if model is not None:
        return
    
    logger.info(f"Loading model from: {config.model_name_or_path}")
    logger.info(f"Local files only: {config.use_local_files_only}")
    logger.info(f"Cache directory: {config.model_cache_dir}")
    logger.info(f"Quantization: {config.use_quantization} ({config.quantization_bits}bit)")
    
    try:
        # Get quantization config
        quantization_config = get_quantization_config()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.model_cache_dir,
            local_files_only=config.use_local_files_only,
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=config.device_map,
            quantization_config=quantization_config,
            cache_dir=config.model_cache_dir,
            local_files_only=config.use_local_files_only,
            torch_dtype=torch.bfloat16 if quantization_config is None else None,
        )
        
        # Store model info
        model_info = {
            "model_name": config.model_name_or_path,
            "quantized": config.use_quantization,
            "quantization_bits": config.quantization_bits if config.use_quantization else None,
            "device_map": config.device_map,
            "local_files_only": config.use_local_files_only,
            "cache_dir": config.model_cache_dir or "default"
        }
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# FastAPI app
app = FastAPI(
    title="Generic LLM Inference API",
    description="Flexible inference server for HuggingFace LLMs with quantization support",
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
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_info)

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Inference endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response (skip the input prompt)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = full_response[len(request.prompt):].strip()
        
        return InferenceResponse(
            response=response_only,
            model_info=model_info
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)