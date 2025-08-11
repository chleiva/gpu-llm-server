import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import logging
from typing import Optional

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables (loaded once at startup)
model = None
tokenizer = None
text_generation_pipeline = None
model_info = {}

def load_model_once():
    """Load model once at startup - using customer's exact approach"""
    global model, tokenizer, text_generation_pipeline, model_info
    
    if model is not None:
        logger.info("Model already loaded, skipping...")
        return
    
    logger.info("Starting model loading...")
    logger.info("Using customer's proven configuration")
    
    try:
        # Quantization config (exactly as customer's)
        use_8bit = True
        bnb_8bit_compute_dtype = torch.bfloat16
        bnb_8bit_quant_type = "nf4"
        use_nested_quant = True
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            bnb_8bit_quant_type=bnb_8bit_quant_type,
            bnb_8bit_compute_dtype=bnb_8bit_compute_dtype,
            bnb_8bit_use_double_quant=use_nested_quant,
        )
        
        logger.info("Loading model: mistralai/Mistral-Small-24B-Instruct-2501")
        logger.info("Quantization: 8-bit with nf4, bfloat16")
        
        # Load model (customer's exact approach)
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-Small-24B-Instruct-2501",
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501")
        
        # Create pipeline (customer's exact setup, but without streamer for API use)
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.1,
            top_p=0.5,
            repetition_penalty=1.1,
            return_full_text=False,  # Changed to False for API responses
            max_new_tokens=1000, 
            do_sample=True,
            # Note: Removed streamer for API use
        )
        
        # Store model info
        model_info = {
            "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
            "quantized": True,
            "quantization_type": "8-bit nf4",
            "device_map": "auto",
            "local_files_only": False,
            "compute_dtype": str(bnb_8bit_compute_dtype)
        }
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# FastAPI app
app = FastAPI(
    title="Mistral LLM Inference API",
    description="FastAPI service using customer's proven Transformers approach",
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
    
    return ModelInfo(
        model_name=model_info["model_name"],
        quantized=model_info["quantized"],
        device_map=model_info["device_map"],
        local_files_only=model_info["local_files_only"]
    )

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Inference endpoint using customer's pipeline approach"""
    if text_generation_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing inference request: {request.prompt[:50]}...")
        
        # Update pipeline parameters for this request
        text_generation_pipeline.task_kwargs.update({
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
        })
        
        # Generate response using customer's pipeline
        outputs = text_generation_pipeline(request.prompt)
        
        # Extract the generated text
        if isinstance(outputs, list) and len(outputs) > 0:
            response_text = outputs[0]["generated_text"].strip()
        else:
            response_text = str(outputs).strip()
        
        logger.info(f"Generated response length: {len(response_text)} chars")
        
        return InferenceResponse(
            response=response_text,
            model_info=model_info
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Mistral LLM Inference API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/health", "/model-info", "/inference"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)