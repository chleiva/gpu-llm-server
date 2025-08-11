#!/usr/bin/env python3
"""
OpenAI-compatible API server for gpt-oss-20b with MXFP4 and request batching
"""

from flask import Flask, request, jsonify, Response
from transformers import pipeline
import torch
import json
import time
import uuid
import argparse
from threading import Thread, Event
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, Any, Optional

app = Flask(__name__)

# Global pipeline
pipe = None
model_id = "openai/gpt-oss-20b"

# Batching configuration
BATCH_SIZE = 8  # Maximum requests per batch
BATCH_TIMEOUT = 0.1  # Maximum wait time in seconds
MIN_BATCH_SIZE = 2  # Minimum batch size to trigger processing

# Request queuing system
request_queue = Queue()
response_store = {}

@dataclass
class BatchRequest:
    """Container for batch request data"""
    request_id: str
    messages: list
    max_tokens: int
    temperature: float
    stream: bool
    event: Event
    
class BatchProcessor:
    """Handles batching and processing of requests"""
    
    def __init__(self, pipeline, batch_size=BATCH_SIZE, timeout=BATCH_TIMEOUT):
        self.pipe = pipeline
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = True
        
    def process_batch(self, batch_requests):
        """Process a batch of requests together"""
        if not batch_requests:
            return
            
        try:
            # Prepare batch inputs
            batch_messages = [req.messages for req in batch_requests]
            
            # Use common parameters from first request (could be improved)
            # In production, you'd want to group by similar parameters
            max_tokens = batch_requests[0].max_tokens
            temperature = batch_requests[0].temperature
            
            print(f"Processing batch of {len(batch_requests)} requests...")
            
            # Process entire batch at once
            outputs = self.pipe(
                batch_messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                return_full_text=False,
                batch_size=len(batch_messages)  # Explicit batch size
            )
            
            # Store results for each request
            for req, output in zip(batch_requests, outputs):
                # Extract response text
                response_text = output["generated_text"]
                if isinstance(response_text, list) and len(response_text) > 0:
                    response_text = response_text[-1].get("content", "")
                elif isinstance(response_text, dict):
                    response_text = response_text.get("content", "")
                
                # Clean up harmony format if present
                if "final" in response_text:
                    response_text = response_text.split("final", 1)[1].strip()
                elif "assistant" in response_text:
                    response_text = response_text.split("assistant", 1)[1].strip()
                
                # Store response
                response_store[req.request_id] = {
                    "success": True,
                    "response": response_text
                }
                
                # Signal that response is ready
                req.event.set()
                
        except Exception as e:
            print(f"Batch processing error: {e}")
            # Mark all requests in batch as failed
            for req in batch_requests:
                response_store[req.request_id] = {
                    "success": False,
                    "error": str(e)
                }
                req.event.set()
    
    def run(self):
        """Main loop for batch processor"""
        while self.running:
            batch = []
            deadline = time.time() + self.timeout
            
            # Collect requests up to batch_size or until timeout
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout_remaining = max(0, deadline - time.time())
                    req = request_queue.get(timeout=timeout_remaining)
                    batch.append(req)
                    
                    # Process immediately if we have a full batch
                    if len(batch) >= self.batch_size:
                        break
                        
                except Empty:
                    # Timeout reached
                    break
            
            # Process batch if we have minimum requests or timeout reached
            if len(batch) >= MIN_BATCH_SIZE or (batch and time.time() >= deadline):
                self.process_batch(batch)
            elif batch:
                # If we have requests but not enough for a batch, process anyway
                # to avoid keeping users waiting too long
                if time.time() >= deadline:
                    self.process_batch(batch)
                else:
                    # Put requests back in queue if not ready to process
                    for req in batch:
                        request_queue.put(req)

# Global batch processor
batch_processor = None

def load_model():
    """Load the model with MXFP4"""
    global pipe, batch_processor
    print(f"Loading {model_id} with MXFP4...")
    pipe = pipeline("text-generation", model=model_id, torch_dtype="auto", device_map="auto")
    print(f"✅ Model loaded!")
    print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    # Start batch processor
    batch_processor = BatchProcessor(pipe)
    processor_thread = Thread(target=batch_processor.run, daemon=True)
    processor_thread.start()
    print(f"✅ Batch processor started (batch_size={BATCH_SIZE}, timeout={BATCH_TIMEOUT}s)")

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
        max_tokens = data.get('max_tokens', 256)
        temperature = data.get('temperature', 0.7)
        stream = data.get('stream', False)
        
        # Add reasoning level if specified
        reasoning = data.get('reasoning_level', 'medium')
        if not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {"role": "system", "content": f"Reasoning: {reasoning}"})
        
        if stream:
            # Streaming still processes individually for now
            return handle_streaming(messages, max_tokens, temperature)
        else:
            # Use batched processing for non-streaming requests
            return handle_batched_request(messages, max_tokens, temperature)
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": 500
            }
        }), 500

def handle_batched_request(messages, max_tokens, temperature):
    """Handle non-streaming requests with batching"""
    # Create unique request ID
    request_id = uuid.uuid4().hex
    
    # Create event to wait for response
    response_event = Event()
    
    # Create batch request
    batch_request = BatchRequest(
        request_id=request_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        event=response_event
    )
    
    # Add to queue
    request_queue.put(batch_request)
    
    # Wait for response (with timeout)
    if response_event.wait(timeout=30):  # 30 second timeout
        # Get response from store
        result = response_store.pop(request_id)
        
        if result["success"]:
            response_text = result["response"]
            
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
        else:
            return jsonify({
                "error": {
                    "message": result["error"],
                    "type": "processing_error",
                    "code": 500
                }
            }), 500
    else:
        # Timeout occurred
        return jsonify({
            "error": {
                "message": "Request timeout",
                "type": "timeout_error",
                "code": 504
            }
        }), 504

def handle_streaming(messages, max_tokens, temperature):
    """Handle streaming responses (not batched for now)"""
    def generate():
        # Generate full response first (simplified streaming)
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            return_full_text=False
        )
        
        response_text = outputs[0]["generated_text"]
        if isinstance(response_text, list) and len(response_text) > 0:
            response_text = response_text[-1].get("content", "")
        elif isinstance(response_text, dict):
            response_text = response_text.get("content", "")
        
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
    """Health check endpoint with queue stats"""
    status = "ready" if pipe is not None else "loading"
    gpu_memory = torch.cuda.memory_allocated(0)/1024**3 if torch.cuda.is_available() else 0
    
    return jsonify({
        "status": status,
        "model": model_id,
        "gpu_memory_gb": round(gpu_memory, 2),
        "queue_size": request_queue.qsize(),
        "batch_config": {
            "batch_size": BATCH_SIZE,
            "timeout_ms": int(BATCH_TIMEOUT * 1000),
            "min_batch_size": MIN_BATCH_SIZE
        }
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get batching statistics"""
    return jsonify({
        "queue_size": request_queue.qsize(),
        "pending_responses": len(response_store),
        "batch_config": {
            "max_batch_size": BATCH_SIZE,
            "batch_timeout_ms": int(BATCH_TIMEOUT * 1000),
            "min_batch_size": MIN_BATCH_SIZE
        },
        "gpu_memory": {
            "allocated_gb": round(torch.cuda.memory_allocated(0)/1024**3, 2) if torch.cuda.is_available() else 0,
            "reserved_gb": round(torch.cuda.memory_reserved(0)/1024**3, 2) if torch.cuda.is_available() else 0
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "GPT-OSS-20B API Server (MXFP4 with Batching)",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions (OpenAI compatible)",
            "/v1/models": "GET - List models",
            "/health": "GET - Health check",
            "/stats": "GET - Batching statistics",
            "/": "GET - This page"
        },
        "batching": {
            "enabled": True,
            "batch_size": BATCH_SIZE,
            "timeout_ms": int(BATCH_TIMEOUT * 1000),
            "info": "Requests are automatically batched for better GPU utilization"
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
    parser = argparse.ArgumentParser(description='GPT-OSS-20B API Server with Batching')
    parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--batch-size', type=int, default=8, help='Maximum batch size')
    parser.add_argument('--batch-timeout', type=float, default=0.1, help='Batch timeout in seconds')
    
    args = parser.parse_args()
    
    # Update batch configuration from arguments
    BATCH_SIZE = args.batch_size
    BATCH_TIMEOUT = args.batch_timeout
    
    # Load model before starting server
    load_model()
    
    # Start Flask server
    print(f"\n🚀 Server running at http://{args.host}:{args.port}")
    print(f"📡 API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"📊 Stats endpoint: http://{args.host}:{args.port}/stats")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    print(f"⚡ Batching: size={BATCH_SIZE}, timeout={BATCH_TIMEOUT}s")
    
    app.run(host=args.host, port=args.port, threaded=True)