from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, time, uuid
torch.cuda.empty_cache()
torch.cuda.synchronize()


app = Flask(__name__)
model = None
tokenizer = None
model_id = "openai/gpt-oss-20b"

def load_model():
    global model, tokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.eval()
    print("Ready!")

@app.route('/v1/chat/completions', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 256)
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    return jsonify({
        "choices": [{"message": {"role": "assistant", "content": response_text}}],
        "model": model_id
    })

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    load_model()
    print(f"Server running on port {port}")
    app.run(host='0.0.0.0', port=port)
EOF