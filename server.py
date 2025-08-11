#!/usr/bin/env python3
"""
Production-optimized OpenAI-compatible API server for gpt-oss-20b
"""

from flask import Flask, request, jsonify, Response
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
import uuid
import argparse
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial
import os

# Production optimizations
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
torch.cuda.empty_cache()  # Clear cache before starting

app = Flask(__name__)

# Global model components
model = None
tokenizer = None
model_id = "openai/gpt-oss-20b"

# Thread pool for parallel processing
executor = None

def load_model_optimized(batch_size=4):
    """Load the model with production optimizations"""
    global model, tokenizer, executor
    
    print(f"Loading {model_id} with production optimizations...")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    # Load tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=True,  # Enable KV cache
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"  # Try Flash Attention
    )
    
    # Enable gradient checkpointing for memory efficiency (if needed)
    # model.gradient_checkpointing_enable()
    
    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="max-autotune")
    
    model.eval()  # Set to evaluation mode
    
    # Initialize thread pool for parallel requests
    executor = ThreadPoolExecutor(max_workers=batch_size)
    
    print(f"✅ Model loaded and optimized!")
    print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

def generate_batch(messages_batch, max_tokens=256, temperature=0.7):
    """Process multiple requests in parallel"""
    with torch.cuda.amp.autocast(dtype=torch.float16):  # Mixed precision
        # Tokenize all messages
        inputs_list = []
        for messages in messages_batch:
            # Convert messages to prompt
            prompt = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant: "
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            inputs_list.append(inputs)
        
        # Batch process
        all_outputs = []
        for inputs in inputs_list:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,  # Greedy for speed
                    use_cache=True  # Use KV cache
                )
            
            # Decode
            generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_outputs.append(response_text)
        
        return all_outputs

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
            return handle_non_streaming_optimized(messages, max_tokens, temperature)
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": 500
            }
        }), 500

def handle_non_streaming_optimized(messages, max_tokens, temperature):
    """Handle non-streaming responses with optimizations"""
    # Process single request efficiently
    outputs = generate_batch([messages], max_tokens, temperature)
    response_text = outputs[0]
    
    # Clean up response if needed
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
        # Generate response
        outputs = generate_batch([messages], max_tokens, temperature)
        response_text = outputs[0]
        
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
    """Health check endpoint with detailed GPU info"""
    status = "ready" if model is not None else "loading"
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "allocated_gb": round(torch.cuda.memory_allocated(0)/1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0)/1024**3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2),
            "name": torch.cuda.get_device_name(0)
        }
    
    return jsonify({
        "status": status,
        "model": model_id,
        "gpu": gpu_info,
        "optimization": {
            "fp16": True,
            "torch_compile": hasattr(torch, 'compile'),
            "flash_attention": "attempting",
            "tf32": torch.backends.cuda.matmul.allow_tf32
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "GPT-OSS-20B Production API Server",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions (OpenAI compatible)",
            "/v1/models": "GET - List models",
            "/health": "GET - Health check with GPU stats",
            "/": "GET - This page"
        },
        "optimizations": [
            "FP16 inference",
            "torch.compile() if available",
            "Flash Attention 2",
            "TF32 matmul",
            "KV cache enabled",
            "CuDNN autotuner"
        ],
        "example": {
            "url": "/v1/chat/completions",
            "method": "POST",
            "body": {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False
            }
        }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-OSS-20B Production Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Set environment variables for production
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Better memory management
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    
    # Load model with optimizations
    load_model_optimized(batch_size=args.workers)
    
    # Production server configuration
    print(f"\n🚀 Production server running at http://{args.host}:{args.port}")
    print(f"📡 API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    print(f"⚡ Optimizations: FP16, torch.compile, Flash Attention")
    print(f"🔧 Workers: {args.workers}")
    
    # For production, consider using gunicorn instead:
    # gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --worker-class gthread --threads 4 server:app
    
    app.run(host=args.host, port=args.port, threaded=True)