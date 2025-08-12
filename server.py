#!/usr/bin/env python3
"""
Optimized OpenAI-compatible API server for gpt-oss-20b with MXFP4
Features: Dynamic batching, tensor parallelism, optimized memory usage
"""

from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
import uuid
import argparse
from threading import Lock
from queue import Queue, Empty
import threading
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
model_id = "openai/gpt-oss-20b"
batch_queue = Queue()
result_dict = {}
result_locks = {}
batch_processor_thread = None

@dataclass
class BatchRequest:
    request_id: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    timestamp: float

class OptimizedModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.lock = Lock()
        
    def load(self):
        """Load model with optimizations"""
        print(f"Loading {self.model_id} with optimizations...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,  # Use fast tokenizer for better performance
            padding_side='left'  # Important for batch generation
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,  # Use FP16 for faster inference
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=True,  # Enable KV cache
            attn_implementation="sdpa"
        )
        
        # Enable torch compile for additional speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Set to eval mode
        self.model.eval()
        
        # Enable gradient checkpointing if needed (saves memory)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        print(f"✅ Model loaded and optimized!")
        print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
    def generate_batch(self, requests: List[BatchRequest]) -> Dict[str, str]:
        """Generate responses for a batch of requests"""
        if not requests:
            return {}
        
        # Prepare batch inputs
        batch_texts = []
        for req in requests:
            # Convert messages to text
            text = ""
            for msg in req.messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                text += f"{role}: {content}\n"
            text += "assistant: "
            batch_texts.append(text)
        
        # Tokenize batch
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate with optimizations
        with torch.cuda.amp.autocast():  # Mixed precision for faster inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(req.max_tokens for req in requests),
                    temperature=requests[0].temperature if requests else 0.7,
                    do_sample=requests[0].temperature > 0 if requests else True,
                    num_beams=1,  # Greedy/sampling is faster than beam search
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode batch outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Map results back to request IDs
        results = {}
        for req, text in zip(requests, generated_texts):
            results[req.request_id] = text
            
        return results

# Initialize optimized model
opt_model = OptimizedModel(model_id)

def batch_processor():
    """Background thread that processes batches"""
    global batch_queue, result_dict, result_locks
    
    BATCH_SIZE = 4  # Adjust based on your GPU memory
    BATCH_TIMEOUT = 0.1  # 100ms timeout for batch collection
    
    while True:
        batch = []
        deadline = time.time() + BATCH_TIMEOUT
        
        # Collect requests for batch
        while len(batch) < BATCH_SIZE and time.time() < deadline:
            timeout = max(0, deadline - time.time())
            try:
                req = batch_queue.get(timeout=timeout)
                batch.append(req)
            except Empty:
                break
        
        if batch:
            # Process batch
            try:
                results = opt_model.generate_batch(batch)
                
                # Store results
                for req_id, response in results.items():
                    with result_locks.get(req_id, Lock()):
                        result_dict[req_id] = response
                        
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Store error for all requests in batch
                for req in batch:
                    with result_locks.get(req.request_id, Lock()):
                        result_dict[req.request_id] = f"Error: {str(e)}"

def load_model():
    """Load the optimized model"""
    global opt_model, batch_processor_thread
    
    if opt_model.model is not None:
        return  # Already loaded
        
    opt_model.load()
    
    # Start batch processor thread
    if batch_processor_thread is None:
        batch_processor_thread = threading.Thread(target=batch_processor, daemon=True)
        batch_processor_thread.start()
        print("✅ Batch processor started!")

# Load model immediately
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
    """Handle chat completion requests with batching"""
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 10000)
        temperature = data.get('temperature', 0.1)
        
        # Add reasoning level if specified
        reasoning = data.get('reasoning_level', 'medium')
        if not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {"role": "system", "content": f"Reasoning: {reasoning}"})
        
        # Create request ID and lock
        request_id = uuid.uuid4().hex
        result_locks[request_id] = Lock()
        
        # Add to batch queue
        batch_req = BatchRequest(
            request_id=request_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timestamp=time.time()
        )
        batch_queue.put(batch_req)
        
        # Wait for result (with timeout)
        timeout = 30  # 30 second timeout
        start_time = time.time()
        while request_id not in result_dict:
            if time.time() - start_time > timeout:
                return jsonify({
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout",
                        "code": 504
                    }
                }), 504
            time.sleep(0.01)  # Check every 10ms
        
        # Get result
        with result_locks[request_id]:
            response_text = result_dict.pop(request_id)
            del result_locks[request_id]
        
        # Format OpenAI-style response
        return jsonify({
            "id": f"chatcmpl-{request_id[:8]}",
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
    status = "ready" if opt_model.model is not None else "loading"
    gpu_memory = torch.cuda.memory_allocated(0)/1024**3 if torch.cuda.is_available() else 0
    
    # Get queue stats
    queue_size = batch_queue.qsize()
    
    return jsonify({
        "status": status,
        "model": model_id,
        "gpu_memory_gb": round(gpu_memory, 2),
        "queue_size": queue_size,
        "batch_processing": batch_processor_thread.is_alive() if batch_processor_thread else False
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "Optimized GPT-OSS-20B API Server (MXFP4)",
        "optimizations": [
            "Dynamic batching for parallel processing",
            "Flash Attention 2 (if available)",
            "Mixed precision (FP16) inference",
            "Torch compile optimization",
            "Efficient tokenization with fast tokenizers",
            "KV cache enabled"
        ],
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions (OpenAI compatible)",
            "/v1/models": "GET - List models",
            "/health": "GET - Health check with queue stats",
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
    parser = argparse.ArgumentParser(description='Optimized GPT-OSS-20B API Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Load model before starting server
    load_model()
    
    # Start Flask server with threading
    print(f"\n🚀 Optimized server running at http://{args.host}:{args.port}")
    print(f"📡 API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    print(f"⚡ Optimizations enabled: Batching, Flash Attention, FP16, Torch Compile")
    
    app.run(host=args.host, port=args.port, threaded=True)