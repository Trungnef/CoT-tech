"""
Model manager for loading and optimizing LLMs with efficient multi-GPU usage.
"""

import os
from dotenv import load_dotenv
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
import google.generativeai as genai
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Model paths from environment variables
qwen_model_path = os.getenv("QWEN_MODEL_PATH")
qwen_tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH")

llama_model_path = os.getenv("LLAMA_MODEL_PATH")
llama_tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")

# GPU configuration from environment variables
MAX_GPU_MEMORY = float(os.getenv("MAX_GPU_MEMORY_GB", 47.5))
SYSTEM_RESERVE = float(os.getenv("SYSTEM_RESERVE_MEMORY_GB", 2.5))
CPU_OFFLOAD = float(os.getenv("CPU_OFFLOAD_GB", 24))

# Configure API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Thread-local storage for models
thread_local = threading.local()

def clear_memory():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Memory cache cleared")

def check_gpu_memory():
    """Check and display GPU memory information."""
    print("\n💾 GPU Memory Information:")
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total - allocated
        total_memory += total
        
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Total: {total:.2f} GB")
        print(f"  - Allocated: {allocated:.2f} GB")
        print(f"  - Reserved: {reserved:.2f} GB")
        print(f"  - Free: {free:.2f} GB")
    print(f"\nTotal GPU Memory: {total_memory:.2f} GB")
    print()

def create_optimal_device_map(model_size_gb=70):
    """
    Create an optimal device map for model distribution across GPUs.
    
    Args:
        model_size_gb (float): Approximate model size in GB
        
    Returns:
        dict: Device map configuration
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"
    
    # Calculate available memory per GPU
    gpu_memories = []
    for i in range(num_gpus):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        available = total - allocated - SYSTEM_RESERVE  # Reserve memory from env config
        gpu_memories.append(available)
    
    total_available = sum(gpu_memories)
    
    if total_available < model_size_gb:
        print(f"⚠️ Warning: Total available GPU memory ({total_available:.2f}GB) is less than model size ({model_size_gb}GB)")
        print("Will use CPU offloading")
        return "auto"
    
    # Create memory map
    memory_map = {}
    for i in range(num_gpus):
        memory_map[f"cuda:{i}"] = f"{int(gpu_memories[i])}GiB"
    memory_map["cpu"] = f"{CPU_OFFLOAD}GiB"  # CPU offload from env config
    
    return memory_map

def load_model_optimized(model_path, tokenizer_path, model_type="llama", dtype=torch.bfloat16, use_4bit=True, model_size_gb=70):
    """
    Load and optimize a model with memory-efficient configuration.
    
    Args:
        model_path: Path to the model files
        tokenizer_path: Path to the tokenizer files
        model_type: Type of model ('llama', 'qwen', etc.)
        dtype: Torch data type (bfloat16, float16, etc.)
        use_4bit: Whether to use 4-bit quantization
        model_size_gb: Approximate model size in GB
        
    Returns:
        tuple: (tokenizer, model)
    """
    clear_memory()
    
    print(f"🔄 Loading {model_type} model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")
        
        # Configure quantization
        if use_4bit:
            print("⚙️ Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Create device map
        device_map = create_optimal_device_map(model_size_gb)
        print(f"📊 Using device map: {device_map}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
            offload_folder="offload"
        )
        
        print("✅ Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def get_thread_local_model(model_type):
    """Get or create thread-local model instance."""
    if not hasattr(thread_local, model_type):
        if model_type == "llama":
            tokenizer, model = load_model_optimized(llama_model_path, llama_tokenizer_path)
        elif model_type == "qwen":
            tokenizer, model = load_model_optimized(qwen_model_path, qwen_tokenizer_path)
        elif model_type == "gemini":
            model = load_gemini_model()
            tokenizer = None
        setattr(thread_local, f"{model_type}_tokenizer", tokenizer)
        setattr(thread_local, f"{model_type}_model", model)
    
    return (
        getattr(thread_local, f"{model_type}_tokenizer", None),
        getattr(thread_local, f"{model_type}_model", None)
    )

def load_gemini_model(model_name="gemini-1.5-pro"):
    """Load Gemini model."""
    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini model '{model_name}' loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading Gemini: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_text_with_model(prompt, model_type, max_tokens=1024, temperature=0.7):
    """
    Generate text using specified model type with automatic model management.
    
    Args:
        prompt: Input prompt text
        model_type: Type of model to use ('llama', 'qwen', 'gemini')
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated text
    """
    try:
        tokenizer, model = get_thread_local_model(model_type)
        
        if model_type in ["llama", "qwen"]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
            
        elif model_type == "gemini":
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95
                }
            ).text
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        return f"Error: {str(e)}"

def parallel_generate(prompts, model_type, max_workers=3):
    """
    Generate responses for multiple prompts in parallel.
    
    Args:
        prompts: List of prompts
        model_type: Type of model to use
        max_workers: Maximum number of parallel workers
        
    Returns:
        list: Generated responses
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_text_with_model, prompt, model_type)
            for prompt in prompts
        ]
        responses = [f.result() for f in futures]
    return responses 