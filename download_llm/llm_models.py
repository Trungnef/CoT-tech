from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Hoặc load_in_8bit=True nếu muốn 8-bit
    bnb_4bit_quant_type="nf4",  # Loại quantization phù hợp
    bnb_4bit_compute_dtype=torch.bfloat16  # Chuyển về float16 để phù hợp với RTX 6000
)

# List of models to download
models = [
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-70B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "deepseek-ai/DeepSeek-R1",
    # "deepseek-ai/DeepSeek-V3",
    # "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    # "perplexity-ai/r1-1776",
]

# Base cache directory
cache_base_dir = "./cache"

# Kiểm tra xem GPU có khả dụng không
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    print(f"🚀 GPU Detected: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
else:
    device = "cpu"
    print("⚠️ No GPU found! Using CPU instead.")

# Loop through each model
for model_checkpoint in models:
    model_name = model_checkpoint.replace("/", "_")  # Convert model name for directory
    tokenizer_cache_dir = os.path.join(cache_base_dir, "tokenizer", model_name)
    model_cache_dir = os.path.join(cache_base_dir, "model", model_name)

    # Ensure directories exist
    os.makedirs(tokenizer_cache_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    try:
        # Download tokenizer and save to cache
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, cache_dir=tokenizer_cache_dir, trust_remote_code=True
        )
        print(f"✅ Tokenizer for {model_checkpoint} downloaded successfully.")

        #Download model and save to cache
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            cache_dir=model_cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Dùng bfloat16 nếu GPU hỗ trợ
            trust_remote_code=True
        )
        

        # Kiểm tra thiết bị mà model đang sử dụng
        model_device = next(model.parameters()).device
        print(f"✅ Model {model_checkpoint} loaded successfully on {model_device}.")

    except Exception as e:
        print(f"❌ Error downloading {model_checkpoint}: {e}")

print("✅ All models processed.")

