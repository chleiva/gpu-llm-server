#!/usr/bin/env python3
"""
OpenAI-compatible API server for gpt-oss-20b with MXFP4
"""

from flask import Flask, request, jsonify, Response
from transformers import pipeline, AutoTokenizer
import torch
import json
import time
import uuid
import argparse
from threading import Thread
from queue import Queue

app = Flask(__name__)

# Global pipeline
pipe = None
model_id = "openai/gpt-oss-20b"

def load_model():
    """Load the model with MXFP4"""
    global pipe
    print(f"Loading {model_id} with MXFP4...")
    
    # Load with left padding for decoder-only models
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # Fix the padding warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        tokenizer=tokenizer,
        torch_dtype="auto", 
        device_map="auto"
    )
    print(f"✅ Model loaded!")
    print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

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
        max_tokens = data.get('max_tokens', 256)
        temperature = data.get('temperature', 0.7)
        stream = data.get('stream', False)
        
        # Add reasoning level if specified
        reasoning = data.get('reasoning_level', 'medium')
        if not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {"role": "system", "content": f"Reasoning: {reasoning}"})
        
        if stream:
            return handle_streaming(messages, max_tokens, temperature)
        else:
            return handle_non_streaming(messages, max_tokens, temperature)
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": 500
            }
        }), 500

def handle_non_streaming(messages, max_tokens, temperature):
    """Handle non-streaming responses"""
    # Use the correct autocast syntax for newer PyTorch
    if torch.cuda.is_available():
        with torch.amp.autocast('cuda', dtype=torch.float16):  # Fixed: Updated syntax
            outputs = pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id  # Ensure proper padding
            )
    else:
        # CPU fallback without autocast
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
    
    # Extract response
    response_text = outputs[0]["generated_text"]
    
    # Handle the response format properly
    if isinstance(response_text, str):
        # Direct string response
        pass
    elif isinstance(response_text, list) and len(response_text) > 0:
        # List of messages
        if isinstance(response_text[-1], dict):
            response_text = response_text[-1].get("content", "")
        else:
            response_text = str(response_text[-1])
    elif isinstance(response_text, dict):
        # Single message dict
        response_text = response_text.get("content", "")
    else:
        response_text = str(response_text)
    
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

def handle_streaming(messages, max_tokens, temperature):
    """Handle streaming responses"""
    def generate():
        # Use the correct autocast syntax
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.float16):  # Fixed: Updated syntax
                outputs = pipe(
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    return_full_text=False,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
        else:
            outputs = pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
        
        response_text = outputs[0]["generated_text"]
        
        # Handle the response format properly
        if isinstance(response_text, str):
            pass
        elif isinstance(response_text, list) and len(response_text) > 0:
            if isinstance(response_text[-1], dict):
                response_text = response_text[-1].get("content", "")
            else:
                response_text = str(response_text[-1])
        elif isinstance(response_text, dict):
            response_text = response_text.get("content", "")
        else:
            response_text = str(response_text)
        
        # Clean up
        if "final" in response_text:
            response_text = response_text.split("final", 1)[1].strip()
        elif "assistant" in response_text:
            response_text = response_text.split("assistant", 1)[1].strip()
        
        # Stream in chunks
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Initial chunk
        yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'delta': {'role': 'assistant'}}]})}\n\n"
        
        # Content chunks
        words = response_text.split()
        chunk_size = 5  # words per chunk
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                chunk_text += " "
            
            yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'delta': {'content': chunk_text}}]})}\n\n"
            time.sleep(0.01)  # Small delay for streaming effect
        
        # Final chunk
        yield f"data: {json.dumps({'id': chunk_id, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

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
                "temperature": 0.7,
                "stream": False,
                "reasoning_level": "medium"
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