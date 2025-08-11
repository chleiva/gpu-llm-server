#!/usr/bin/env python3
"""
Script to download and quantize Mistral-Small-3.1-24B-Base-2503 model to 8-bit
Compatible with Hugging Face Transformers library
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from pathlib import Path

def check_transformers_version():
    """Check and report transformers version"""
    import transformers
    print(f"Current transformers version: {transformers.__version__}")
    return transformers.__version__

def download_and_quantize_model(
    model_id="mistralai/Mistral-Small-3.1-24B-Base-2503",
    output_dir="./mistral-small-8bit",
    cache_dir=None,
    use_auth_token=None
):
    """
    Download and quantize the Mistral model to 8-bit format
    
    Args:
        model_id: Hugging Face model ID
        output_dir: Directory to save the quantized model
        cache_dir: Optional cache directory for downloads
        use_auth_token: Optional Hugging Face auth token
    """
    
    print(f"Starting download and quantization of {model_id}")
    print(f"Output directory: {output_dir}")
    
    # Check transformers version
    check_transformers_version()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True
    )
    
    print("\n1. Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            trust_remote_code=True  # Essential for Mistral 3 models
        )
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        print("\nTrying with legacy option...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            use_fast=True
        )
    
    print("\n2. Downloading and quantizing model to 8-bit...")
    print("Note: This will take time and requires significant RAM during the process")
    print("Using trust_remote_code=True for Mistral 3 architecture support")
    
    try:
        # First attempt: Load with trust_remote_code for Mistral 3 support
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            trust_remote_code=True,  # This is crucial for Mistral 3
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
    except ValueError as ve:
        print(f"error: {ve}")


        if "Mistral3Config" in str(ve):
            print("\n⚠️  Mistral 3 architecture not supported in current transformers version")
            print("Attempting alternative loading method...")
            
            # Alternative: Try loading as a generic model
            try:
                from transformers import AutoConfig
                
                # Download config first
                config = AutoConfig.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    trust_remote_code=True
                )
                
                # Try to load as MistralForCausalLM (fallback to Mistral v1/v2 architecture)
                from transformers import MistralForCausalLM
                
                model = MistralForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True
                )
                
            except Exception as e2:
                print(f"\n✗ Alternative loading failed: {str(e2)}")
                print("\n" + "="*60)
                print("SOLUTION OPTIONS:")
                print("="*60)
                print("\n1. Update transformers to the latest version:")
                print("   pip install --upgrade transformers")
                print("\n2. Install from source for latest Mistral 3 support:")
                print("   pip install git+https://github.com/huggingface/transformers.git")
                print("\n3. Use the model with remote code execution:")
                print("   The model requires trust_remote_code=True which we've already set.")
                print("\n4. Alternative: Use an older Mistral model that's fully supported:")
                print("   - mistralai/Mistral-7B-v0.1")
                print("   - mistralai/Mixtral-8x7B-v0.1")
                print("\nPlease upgrade transformers and try again.")
                sys.exit(1)
        else:
            raise ve
    
    print("\n3. Saving quantized model and tokenizer...")
    
    # Save the quantized model
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save a config file to indicate this is a quantized model
    import json
    quantization_info = {
        "quantization": "8bit",
        "original_model": model_id,
        "quantization_config": {
            "load_in_8bit": True,
            "bnb_8bit_compute_dtype": "float16",
            "bnb_8bit_quant_type": "nf8",
            "bnb_8bit_use_double_quant": True
        }
    }
    
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump(quantization_info, f, indent=2)
    
    print(f"\n✓ Successfully quantized and saved model to {output_dir}")
    
    # Print model size information
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    ) / (1024**3)
    
    print(f"Total size of quantized model: {total_size:.2f} GB")
    
    return model, tokenizer

def test_quantized_model(model_dir="./mistral-small-8bit"):
    """
    Test loading and using the quantized model with transformers
    """
    print("\n" + "="*50)
    print("Testing quantized model loading...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        # For loading the saved 8-bit model
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Test inference
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest prompt: {test_prompt}")
        print(f"Model response: {response}")
        print("\n✓ Model loaded and working correctly!")
        
    except Exception as e:
        print(f"\n✗ Error testing model: {str(e)}")
        raise

def check_requirements():
    """Check system requirements and installed packages"""
    print("Checking system requirements...")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check installed packages
    packages = {
        "torch": None,
        "transformers": None,
        "accelerate": None,
        "bitsandbytes": None
    }
    
    for package in packages:
        try:
            module = __import__(package)
            if hasattr(module, "__version__"):
                packages[package] = module.__version__
                print(f"{package}: {module.__version__}")
            else:
                packages[package] = "installed"
                print(f"{package}: installed")
        except ImportError:
            packages[package] = "NOT INSTALLED"
            print(f"{package}: NOT INSTALLED ⚠️")
    
    # Check CUDA availability
    if packages["torch"] != "NOT INSTALLED":
        import torch
        if torch.cuda.is_available():
            print(f"\nCUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\nCUDA available: No ⚠️")
    
    print("="*50)
    
    # Check for required packages
    missing = [pkg for pkg, ver in packages.items() if ver == "NOT INSTALLED"]
    if missing:
        print("\n⚠️  Missing required packages:", ", ".join(missing))
        print("\nInstall with:")
        print("pip install torch transformers accelerate bitsandbytes")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download and quantize Mistral-Small-3.1-24B model to 8-bit"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mistral-small-8bit",
        help="Directory to save the quantized model"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face authentication token (if needed)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the quantized model after saving"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test an existing quantized model"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system requirements"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        check_requirements()
        sys.exit(0)
    
    # Always check requirements first
    if not check_requirements():
        sys.exit(1)
    
    if args.test_only:
        test_quantized_model(args.output_dir)
    else:
        # Download and quantize
        model, tokenizer = download_and_quantize_model(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            use_auth_token=args.hf_token
        )
        
        # Test if requested
        if args.test:
            test_quantized_model(args.output_dir)

if __name__ == "__main__":
    main()