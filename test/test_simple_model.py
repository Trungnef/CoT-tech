"""
Script để kiểm tra việc tải mô hình nhỏ từ Hugging Face.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_small_model():
    """Test loading a small model from Hugging Face."""
    print("=== Kiểm tra tải mô hình nhỏ từ Hugging Face ===")
    
    # Mô hình nhỏ để kiểm tra (khoảng 1-3GB)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Tải mô hình và tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Mô hình và tokenizer đã được tải thành công")
        
        # Kiểm tra mô hình với một đầu vào đơn giản
        input_text = "<|system|>\nBạn là trợ lý AI hữu ích.\n<|user|>\nTính 12 + 23\n<|assistant|>"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput: {input_text}")
        print(f"Output: {result}")
        print("✅ Kiểm tra mô hình thành công")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_small_model() 