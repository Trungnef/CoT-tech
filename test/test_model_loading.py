"""
Script để kiểm tra việc tải mô hình.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_memory():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("🧹 Memory cache cleared")

def test_load_model(model_name, model_path, tokenizer_path):
    """Test loading a model."""
    print(f"\n=== Kiểm tra tải mô hình {model_name} ===")
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    
    clear_memory()
    
    print("1. Tải tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
    
    print("\n2. Tải mô hình...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3. Kiểm tra mô hình...")
    try:
        input_text = "Hãy giải bài toán sau: 5 + 7 = ?"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {input_text}")
        print(f"Output: {generated_text}")
        print("✅ Model test completed successfully")
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Get model paths from environment variables
    llama_model_path = os.getenv("LLAMA_MODEL_PATH")
    llama_tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")
    
    qwen_model_path = os.getenv("QWEN_MODEL_PATH")
    qwen_tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH")
    
    # Check GPU information
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total - allocated
            print(f"    Total: {total:.2f} GB")
            print(f"    Used: {allocated:.2f} GB")
            print(f"    Free: {free:.2f} GB")
    else:
        print("No GPU available. Using CPU only.")
    
    # Test loading models
    test_load_model("Llama", llama_model_path, llama_tokenizer_path)
    test_load_model("Qwen", qwen_model_path, qwen_tokenizer_path)

if __name__ == "__main__":
    main() 