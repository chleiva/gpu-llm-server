#!/usr/bin/env python3
"""
OpenAI-compatible API server for gpt-oss-20b with MXFP4
"""

from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import json
import time
import uuid
import argparse

app = Flask(__name__)

# Global pipeline
pipe = None
model_id = "openai/gpt-oss-20b"

def load_model():
    """Load the model with MXFP4"""
    global pipe
    if pipe is not None:
        return  # Already loaded
    print(f"Loading {model_id} with MXFP4...")
    pipe = pipeline("text-generation", model=model_id, torch_dtype="auto", device_map="auto")
    print(f"✅ Model loaded!")
    print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# Load model immediately when module is imported (for gunicorn)
load_model()

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        }]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completion requests"""
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 10000)
        temperature = data.get('temperature', 0.1)
        
        # Add reasoning level if specified
        reasoning = data.get('reasoning_level', 'medium')
        if not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {"role": "system", "content": f"Reasoning: {reasoning}"})
        
        # Generate with pipeline
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            return_full_text=False
        )
        
        # Extract response
        response_text = outputs[0]["generated_text"]
        if isinstance(response_text, list) and len(response_text) > 0:
            response_text = response_text[-1].get("content", "")
        elif isinstance(response_text, dict):
            response_text = response_text.get("content", "")
        
        # Clean up harmony format if present
        if "final" in response_text:
            response_text = response_text.split("final", 1)[1].strip()
        elif "assistant" in response_text:
            response_text = response_text.split("assistant", 1)[1].strip()
        
        # Format OpenAI-style response
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(messages)),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(str(messages)) + len(response_text.split())
            }
        })
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": 500
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = "ready" if pipe is not None else "loading"
    gpu_memory = torch.cuda.memory_allocated(0)/1024**3 if torch.cuda.is_available() else 0
    return jsonify({
        "status": status,
        "model": model_id,
        "gpu_memory_gb": round(gpu_memory, 2)
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
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
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
                "temperature": 0.7
            }
        }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-OSS-20B API Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Load model before starting server
    load_model()
    
    # Start Flask server
    print(f"\n🚀 Server running at http://{args.host}:{args.port}")
    print(f"📡 API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    
    app.run(host=args.host, port=args.port, threaded=True)