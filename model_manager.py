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
import time
import psutil
import pickle
import tempfile
import random
from pathlib import Path

# Load environment variables
load_dotenv()

# Debug: Print environment variables
print("🔍 Debug: Environment variables:")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"Current working directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")

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
gemini_api_key = os.getenv("GEMINI_API_KEY")
print(f"🔑 Using Gemini API key: {gemini_api_key}")

if not gemini_api_key:
    raise ValueError("Gemini API key not found in environment variables")

genai.configure(api_key=gemini_api_key)

# Thread-local storage for models
thread_local = threading.local()

# Thêm biến toàn cục để lưu trữ mô hình đã tải
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_LAST_USED = {}
_CACHE_DIR = Path("./model_cache")
_CACHE_DIR.mkdir(exist_ok=True)

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
    # Use cached model if available
    cache_key = f"{model_type}_{model_path}"
    if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
        print(f"✅ Using cached {model_type} model")
        _LAST_USED[cache_key] = time.time()
        return _TOKENIZER_CACHE[cache_key], _MODEL_CACHE[cache_key]
        
    clear_memory()
    
    print(f"🔄 Loading {model_type} model from {model_path}")
    
    try:
        # Load tokenizer first - this is fast
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=_CACHE_DIR
        )
        print("✅ Tokenizer loaded")
        
        # Configure quantization
        if use_4bit:
            print("⚙️ Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Sử dụng bfloat16 thay vì float16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_device_map=None,  # Allow auto compute map
                llm_int8_enable_fp32_cpu_offload=False  # Disable CPU offloading trong quá trình load
            )
        else:
            quantization_config = None
            
        # Optimize GPU memory first
        if torch.cuda.is_available():
            print(f"💾 Available GPU memory before loading:")
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - allocated
                print(f"  GPU {i}: {free:.2f}GB free / {total:.2f}GB total")
            
            # Làm sạch memory trước khi load
            torch.cuda.empty_cache()
            gc.collect()
        
        # Setup device map và max memory
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus > 1:
            print(f"🔧 Using custom device map for {num_gpus} GPUs")
            # Tạo max_memory config cho từng GPU
            max_memory = {}
            for i in range(num_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                # Để lại một chút memory để tránh OOM
                usable_memory = int(gpu_memory * 0.85)
                max_memory[i] = f"{usable_memory}GiB"
            
            # Thêm CPU memory vào cấu hình
            max_memory["cpu"] = "32GiB"
        else:
            max_memory = None
        
        # Start loading with progress indication
        print("🔄 Starting model loading with progress tracking...")
        
        # Prepare base model kwargs
        base_model_kwargs = {
            "device_map": "auto",  # Sử dụng auto device map
            "max_memory": max_memory,  # Cấu hình memory cho từng device
            "torch_dtype": dtype,
            "quantization_config": quantization_config if use_4bit else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "cache_dir": _CACHE_DIR,
            "offload_folder": "offload",  # Thêm thư mục offload
        }
        
        # Add model-specific parameters based on model type
        model_kwargs = base_model_kwargs.copy()
        
        # Llama models may not support sliding_window and some attention implementations
        if model_type.lower() == "llama":
            # Remove parameters not supported by Llama
            model_kwargs.pop("sliding_window", None)  # Ensure sliding_window is not passed
            model_kwargs["attn_implementation"] = "eager"  # Use eager attention for Llama
        else:
            # For other models, we can use the full set of parameters
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["use_flash_attention_2"] = False
            model_kwargs["sliding_window"] = None
        
        # Add loading progress
        loading_start = time.time()
        
        # Load model with progress update thread
        progress_step = 0
        progress_done = False
        
        def show_progress():
            nonlocal progress_step
            progress_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            while not progress_done:
                elapsed = time.time() - loading_start
                char = progress_chars[progress_step % len(progress_chars)]
                print(f"\r{char} Loading model... (elapsed: {elapsed:.1f}s)", end="")
                progress_step += 1
                time.sleep(0.1)
            print()  # New line after progress is done
        
        # Start progress thread
        import threading
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Tạo thư mục offload nếu cần
            os.makedirs("offload", exist_ok=True)
            
            # Load the model với các tùy chọn đã cấu hình
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Store in memory cache
            _MODEL_CACHE[cache_key] = model
            _TOKENIZER_CACHE[cache_key] = tokenizer
            _LAST_USED[cache_key] = time.time()
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            # Thử lại với cấu hình khác nếu gặp lỗi
            try:
                print("🔄 Retrying with simpler configuration...")
                # Cấu hình đơn giản hơn, không sử dụng 4-bit
                retry_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "cache_dir": _CACHE_DIR,
                }
                
                # Nếu là mô hình Llama, không thêm các tham số không được hỗ trợ
                if model_type.lower() == "llama":
                    retry_kwargs["attn_implementation"] = "eager"
                else:
                    retry_kwargs["attn_implementation"] = "eager"
                    retry_kwargs["sliding_window"] = None
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **retry_kwargs
                )
                
                # Lưu vào cache nếu thành công
                _MODEL_CACHE[cache_key] = model
                _TOKENIZER_CACHE[cache_key] = tokenizer
                _LAST_USED[cache_key] = time.time()
                
            except Exception as retry_error:
                print(f"\n❌ Error loading model with simpler config: {retry_error}")
                # Thử lại lần cuối với cấu hình tối giản nhất
                try:
                    print("🔄 Retrying with minimal configuration...")
                    minimal_kwargs = {
                        "device_map": "auto",
                        "torch_dtype": torch.float16,  # Thử với float16
                        "trust_remote_code": True,
                    }
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **minimal_kwargs
                    )
                    
                    # Lưu vào cache nếu thành công
                    _MODEL_CACHE[cache_key] = model
                    _TOKENIZER_CACHE[cache_key] = tokenizer
                    _LAST_USED[cache_key] = time.time()
                    
                except Exception as final_error:
                    print(f"\n❌ All loading attempts failed: {final_error}")
                    return tokenizer, None
        finally:
            # Stop progress thread
            progress_done = True
            progress_thread.join(timeout=1.0)
        
        loading_time = time.time() - loading_start
        print(f"✅ Model loaded successfully in {loading_time:.2f} seconds")
        
        # Print GPU memory after loading
        if torch.cuda.is_available():
            print(f"💾 GPU memory after loading:")
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = total - allocated
                print(f"  GPU {i}: {free:.2f}GB free / {total:.2f}GB total (Reserved: {reserved:.2f}GB)")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        
        return tokenizer, None

def get_thread_local_model(model_type):
    """Get or create thread-local model instance."""
    # Debug output
    print(f"🔍 Debug: get_thread_local_model called with model_type: '{model_type}'")
    
    # Normalize model type to lowercase
    model_type_lower = model_type.lower()
    
    # Check if model type is valid
    valid_models = ["llama", "qwen", "gemini"]
    if model_type_lower not in valid_models:
        print(f"❌ Error: Invalid model type '{model_type}'. Valid options are: {valid_models}")
        return None, None
    
    if not hasattr(thread_local, model_type_lower):
        print(f"🔍 Debug: Loading model '{model_type_lower}' for the first time in this thread")
        if model_type_lower == "llama":
            tokenizer, model = load_model_optimized(llama_model_path, llama_tokenizer_path)
        elif model_type_lower == "qwen":
            tokenizer, model = load_model_optimized(qwen_model_path, qwen_tokenizer_path)
        elif model_type_lower == "gemini":
            model = load_gemini_model()
            tokenizer = None
        setattr(thread_local, f"{model_type_lower}_tokenizer", tokenizer)
        setattr(thread_local, f"{model_type_lower}_model", model)
    else:
        print(f"🔍 Debug: Using cached model '{model_type_lower}' from thread local storage")
    
    return (
        getattr(thread_local, f"{model_type_lower}_tokenizer", None),
        getattr(thread_local, f"{model_type_lower}_model", None)
    )

def load_gemini_model(model_name="gemini-1.5-flash"):
    """Load Gemini model."""
    try:
        # Validate API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found in environment variables")
        
        # Set API key again before creating model
        genai.configure(api_key=api_key)
        
        # Use the 'flash' version which is smaller and has higher quota limits
        model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini model '{model_name}' loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading Gemini: {e}")
        print("⚠️ A valid API key from Google AI Studio is required (https://ai.google.dev/)")
        print("⚠️ If you have a valid API key, your quota may be exhausted. Try again later or create a new key.")
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
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
        
        # Check if model failed to load
        if model is None:
            return f"[Error: Could not load {model_type} model. Please check GPU memory and model paths.]"
        
        if model_type in ["llama", "qwen"]:
            # Ensure tokenizer and model are available
            if tokenizer is None:
                return f"[Error: No tokenizer found for {model_type}.]"
                
            try:
                # Đặt padding token nếu chưa có
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Encode với padding và tạo attention mask
                encoding = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    return_attention_mask=True
                )
                
                # Chuyển cả input_ids và attention_mask sang device của model
                input_ids = encoding['input_ids'].to(model.device)
                attention_mask = encoding['attention_mask'].to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,  # Cung cấp attention mask
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Lấy text đầu ra từ vị trí cuối cùng của input
                input_length = input_ids.size(1)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(tokenizer.decode(input_ids[0][:input_length], skip_special_tokens=True)):]
                
                # Check if response is empty
                if not response or len(response.strip()) == 0:
                    return "[Error: Model generated empty response. Try with different parameters.]"
            except Exception as e:
                print(f"❌ Error during model generation: {e}")
                return f"[Error generating text with {model_type}: {str(e)}]"
            
        elif model_type == "gemini":
            try:
                # Add delay between requests to avoid rate limits
                time.sleep(1)
                
                # Check if model is None (failed to load)
                if model is None:
                    return "[Error: Could not initialize Gemini API. Check your API key.]"
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 0.95
                    }
                ).text
                
                # Validate response
                if not response or len(response.strip()) == 0:
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(err in error_str for err in ["429", "quota", "rate limit", "resource exhausted"]):
                    print(f"⚠️ Gemini API quota/rate limit exceeded. Waiting before retry... ({e})")
                    # Add longer delay for rate limit errors
                    time.sleep(5)
                    raise e  # Trigger retry with exponential backoff
                elif "500" in error_str or "503" in error_str:
                    print(f"⚠️ Gemini API server error. Retrying... ({e})")
                    time.sleep(2)
                    raise e  # Trigger retry
                else:
                    print(f"❌ Unexpected Gemini API error: {e}")
                    return f"[Error: Could not generate response due to API error: {str(e)}]"
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        # Provide a fallback response that won't break the evaluation
        return f"[Error: Could not generate response: {str(e)}]"

def parallel_generate(prompts, model_type, max_workers=3):
    """
    Generate responses for multiple prompts in parallel.
    
    Args:
        prompts: List of prompts
        model_type: Type of model to use ('llama', 'qwen', 'gemini')
        max_workers: Maximum number of parallel workers
        
    Returns:
        list: Generated responses
    """
    # Check if model_type is valid
    valid_models = ["llama", "qwen", "gemini"]
    if model_type not in valid_models:
        print(f"❌ Error: Invalid model type '{model_type}'. Valid options are: {valid_models}")
        return ["[Error: Invalid model type]" for _ in prompts]
    
    # For API-based models, process sequentially to avoid rate limits
    if model_type == "gemini":
        print(f"📝 Using sequential processing for {model_type} to avoid rate limits")
        return [generate_text_with_model(prompt, model_type) for prompt in prompts]
    
    # For local models, use parallel processing with the specified number of workers
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(generate_text_with_model, prompt, model_type)
                for prompt in prompts
            ]
            responses = [f.result() for f in futures]
        return responses
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to sequential processing
        print("⚠️ Falling back to sequential processing")
        return [generate_text_with_model(prompt, model_type) for prompt in prompts]

def clear_gpu_memory():
    """Clear GPU memory comprehensively."""
    gc.collect()
    torch.cuda.empty_cache()
    # More aggressive clearing
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def get_system_memory_info():
    """Get information about system memory."""
    vm = psutil.virtual_memory()
    return {
        "total": vm.total / (1024 ** 3),  # GB
        "available": vm.available / (1024 ** 3),  # GB
        "percent_used": vm.percent,
    }

def get_gpu_memory_info():
    """Get information about GPU memory."""
    if not torch.cuda.is_available():
        return None
    
    result = []
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        mem_total = info.total_memory / (1024 ** 3)  # GB
        mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        mem_free = mem_total - mem_reserved
        
        result.append({
            "device": i,
            "name": info.name,
            "total": mem_total,
            "reserved": mem_reserved,
            "allocated": mem_allocated,
            "free": mem_free
        })
    
    return result

def should_offload_models(threshold_mb=4000):
    """Check if we should offload unused models to save memory."""
    vm = psutil.virtual_memory()
    if vm.available < threshold_mb * 1024 * 1024:  # Convert from MB to bytes
        return True
    
    # Check GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
            if free_memory < threshold_mb * 1024 * 1024:
                return True
    
    return False

def serialize_model(model, model_name):
    """Nén và lưu trạng thái mô hình để tải lại nhanh hơn."""
    model_path = _CACHE_DIR / f"{model_name}_state.pkl"
    
    # Lưu state_dict thay vì toàn bộ mô hình
    state_dict = model.state_dict()
    torch.save(state_dict, model_path)
    
    return model_path

def deserialize_model(model_class, model_name, config):
    """Tải mô hình từ trạng thái đã lưu."""
    model_path = _CACHE_DIR / f"{model_name}_state.pkl"
    
    if not model_path.exists():
        return None
    
    # Tạo mô hình mới và tải state_dict
    try:
        # Tạo mô hình với config nhưng không tải trọng số
        model = model_class.from_config(config)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình từ cache: {e}")
        # Xóa file cache lỗi
        model_path.unlink(missing_ok=True)
        return None

def get_max_memory_config():
    """
    Get optimal memory configuration for model loading based on available GPU and system memory.
    
    Returns:
        dict: A dictionary mapping device IDs to memory limits
    """
    memory_config = {}
    
    # Check if GPU is available
    if torch.cuda.is_available():
        # Set memory limit for each GPU
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            # Reserve memory for system operations
            available_memory = total_memory - SYSTEM_RESERVE
            # Limit to MAX_GPU_MEMORY if specified
            if available_memory > MAX_GPU_MEMORY:
                available_memory = MAX_GPU_MEMORY
            
            memory_config[f"cuda:{i}"] = f"{int(available_memory)}GiB"
    
    # Add CPU memory configuration
    memory_config["cpu"] = f"{CPU_OFFLOAD}GiB"
    
    return memory_config

def load_model_with_caching(model_name, model_path, tokenizer_path, use_4bit=True, max_memory=None):
    """Load model with caching and intelligent memory management."""
    # Check if model is already cached
    if model_name in _MODEL_CACHE and model_name in _TOKENIZER_CACHE:
        # Update last used time
        _LAST_USED[model_name] = time.time()
        print(f"👍 Using {model_name} model from cache")
        return _TOKENIZER_CACHE[model_name], _MODEL_CACHE[model_name]
    
    # If memory is low, offload least recently used models
    if should_offload_models():
        offload_least_used_models()
    
    # Clear GPU memory before loading new model
    clear_gpu_memory()
    
    print(f"⏳ Loading {model_name} model...")
    start_time = time.time()
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Configure memory for model - use integers for GPU indices instead of 'cuda:0'
        if max_memory is None:
            # Create a simpler device map that works with newer transformers versions
            max_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Use integer device IDs instead of 'cuda:X'
                    max_memory[i] = f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 - 2)}GiB"
            max_memory["cpu"] = f"{CPU_OFFLOAD}GiB"
            print(f"🔧 Using device map: {max_memory}")
        
        # Load model with quantization, using "auto" device_map
        base_model_kwargs = {
            "device_map": "auto",
            "max_memory": max_memory,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        # Add model-specific parameters based on model type
        model_kwargs = base_model_kwargs.copy()
        
        # Llama models may not support sliding_window and some attention implementations
        if model_name.lower() == "llama":
            # Use only parameters supported by Llama
            model_kwargs["attn_implementation"] = "eager"
        else:
            # For other models, we can use additional parameters
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["use_flash_attention_2"] = False
            model_kwargs["sliding_window"] = None
        
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_enable_fp32_cpu_offload=True  # Enable CPU offloading
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("⚠️ Cannot use 4-bit quantization, bitsandbytes library not available")
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"❌ Error loading model with default config: {e}")
            print("🔄 Retrying with simpler configuration...")
            
            # Try a simpler configuration
            simple_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            }
            
            # Llama models may not support certain parameters
            if model_name.lower() == "llama":
                simple_kwargs["attn_implementation"] = "eager"
            else:
                simple_kwargs["attn_implementation"] = "eager"
                simple_kwargs["sliding_window"] = None
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **simple_kwargs
                )
            except Exception as e2:
                print(f"❌ Error loading with simplified config: {e2}")
                print("🔄 Final attempt with minimal configuration...")
                
                # Last attempt with minimal configuration
                minimal_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                }
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **minimal_kwargs
                )
        
        # Store in cache
        _MODEL_CACHE[model_name] = model
        _TOKENIZER_CACHE[model_name] = tokenizer
        _LAST_USED[model_name] = time.time()
        
        # Save model state for faster reloading later
        serialize_model(model, model_name)
        
        load_time = time.time() - start_time
        print(f"✅ Loaded {model_name} model in {load_time:.2f} seconds")
        print_memory_usage()
        
        return tokenizer, model
    
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        return None, None

def print_memory_usage():
    """Print information about memory usage."""
    sys_memory = get_system_memory_info()
    gpu_memory = get_gpu_memory_info()
    
    print(f"\n--- System Memory Information ---")
    print(f"RAM: {sys_memory['available']:.1f}GB free / {sys_memory['total']:.1f}GB total ({sys_memory['percent_used']}% used)")
    
    if gpu_memory:
        print("\n--- GPU Memory Information ---")
        for idx, gpu in enumerate(gpu_memory):
            print(f"GPU {idx} - {gpu['name']}: {gpu['free']:.1f}GB free / {gpu['total']:.1f}GB total")

def offload_least_used_models(keep_newest=1):
    """Offload least recently used models to save memory."""
    if len(_MODEL_CACHE) <= keep_newest:
        return
    
    # Sort models by recent usage time
    sorted_models = sorted(_LAST_USED.items(), key=lambda x: x[1], reverse=True)
    
    # Keep newest models, offload the rest
    models_to_keep = [model[0] for model in sorted_models[:keep_newest]]
    models_to_offload = [model[0] for model in sorted_models[keep_newest:]]
    
    for model_name in models_to_offload:
        if model_name in _MODEL_CACHE:
            print(f"🔄 Offloading {model_name} model to save memory")
            # Ensure model state dict is saved before offloading
            serialize_model(_MODEL_CACHE[model_name], model_name)
            # Remove model from cache
            del _MODEL_CACHE[model_name]
            # Keep tokenizer as it's much smaller
            gc.collect()
            torch.cuda.empty_cache()

def load_model(model_name, use_4bit=True):
    """Load large language model with intelligent caching."""
    # Debug output
    print(f"🔍 Debug: Attempting to load model: '{model_name}'")
    
    # Normalize model name to lowercase
    model_name_lower = model_name.lower()
    
    # Get model path from environment variables
    if model_name_lower == "llama":
        model_path = os.getenv("LLAMA_MODEL_PATH")
        tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")
        print(f"🔍 Debug: Using paths for llama model: {model_path}")
    elif model_name_lower == "qwen":
        model_path = os.getenv("QWEN_MODEL_PATH")
        tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH")
        print(f"🔍 Debug: Using paths for qwen model: {model_path}")
    else:
        print(f"❌ Model not supported: {model_name}")
        return None, None
    
    # Check if path is valid
    if not model_path or not tokenizer_path:
        print(f"❌ Path not found for {model_name} model in .env")
        return None, None
    
    return load_model_with_caching(model_name, model_path, tokenizer_path, use_4bit)

# Thêm hàm quản lý bộ nhớ model cache tốt hơn
def manage_model_cache(max_models=2):
    """
    Quản lý bộ nhớ cache mô hình, giữ chỉ số lượng mô hình được chỉ định 
    và giải phóng bộ nhớ cho các mô hình ít được sử dụng.
    
    Args:
        max_models (int): Số lượng mô hình tối đa để giữ trong bộ nhớ
    """
    if len(_MODEL_CACHE) <= max_models:
        return
    
    # Sắp xếp các mô hình theo thời gian sử dụng gần đây nhất
    sorted_models = sorted(_LAST_USED.items(), key=lambda x: x[1], reverse=True)
    
    # Giữ lại những mô hình được sử dụng gần đây nhất
    models_to_keep = [model[0] for model in sorted_models[:max_models]]
    
    # Giải phóng bộ nhớ cho các mô hình ít được sử dụng
    for model_key in list(_MODEL_CACHE.keys()):
        if model_key not in models_to_keep:
            print(f"🧹 Offloading model {model_key} to save memory")
            # Lưu trạng thái mô hình vào cache nếu cần
            # serialize_model(_MODEL_CACHE[model_key], model_key)
            # Xóa khỏi bộ nhớ
            del _MODEL_CACHE[model_key]
            # Giữ lại tokenizer vì nó nhỏ hơn nhiều
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
# Thêm hàm mới để giám sát và báo cáo hiệu suất mô hình
def monitor_generation_stats(model_name, start_time, end_time, input_length, output_length):
    """
    Theo dõi hiệu suất sinh văn bản của từng mô hình.
    
    Args:
        model_name (str): Tên mô hình
        start_time (float): Thời gian bắt đầu
        end_time (float): Thời gian kết thúc
        input_length (int): Độ dài đầu vào
        output_length (int): Độ dài đầu ra
    
    Returns:
        dict: Thống kê hiệu suất
    """
    elapsed = end_time - start_time
    tokens_per_second = output_length / elapsed if elapsed > 0 else 0
    
    stats = {
        "model_name": model_name,
        "elapsed_time": elapsed,
        "input_length": input_length,
        "output_length": output_length,
        "tokens_per_second": tokens_per_second,
        "timestamp": time.time()
    }
    
    # Log thống kê
    print(f"📊 {model_name} generation stats:")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - Output length: {output_length} tokens")
    print(f"  - Speed: {tokens_per_second:.2f} tokens/second")
    
    return stats

# Cải tiến hàm generate_text_with_model
def generate_text_with_model(model_name, prompt, max_tokens=1024, temperature=0.7):
    """
    Generate text using specified model with improved performance monitoring.
    
    Args:
        model_name: Type of model to use ('llama', 'qwen', 'gemini')
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated text and performance statistics
    """
    # Manage model cache to prevent OOM errors
    manage_model_cache(max_models=2)
    
    # Check input validity
    if not prompt or len(prompt.strip()) == 0:
        return "[Error: Empty prompt]"
    
    model_name_lower = model_name.lower()
    
    # Validate model name
    valid_models = ["llama", "qwen", "gemini"]
    if model_name_lower not in valid_models:
        return f"[Error: Unsupported model '{model_name}'. Valid options are: {valid_models}]"
    
    # Record start time
    start_time = time.time()
    
    try:
        if model_name_lower in ["llama", "qwen"]:
            # Get model path from environment variables
            if model_name_lower == "llama":
                model_path = os.getenv("LLAMA_MODEL_PATH")
                tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")
            else:  # qwen
                model_path = os.getenv("QWEN_MODEL_PATH")
                tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH")
            
            # Check if paths are valid
            if not model_path or not tokenizer_path:
                return f"[Error: Path not found for {model_name} model in .env]"
            
            # Use cache or load model
            cache_key = f"{model_name_lower}_{model_path}"
            
            if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
                tokenizer = _TOKENIZER_CACHE[cache_key]
                model = _MODEL_CACHE[cache_key]
                _LAST_USED[cache_key] = time.time()
                print(f"👍 Using {model_name} model from cache")
            else:
                tokenizer, model = load_model_optimized(model_path, tokenizer_path, model_name_lower)
            
            # Check if model loaded successfully
            if model is None:
                return f"[Error: Failed to load {model_name} model]"
            
            # Generate text with improved error handling
            try:
                # Calculate input length for statistics
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                input_length = input_ids.size(1)
                
                # Generate with proper error handling
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
                # Get only the generated part by removing the prompt
                if full_output.startswith(prompt_decoded):
                    response = full_output[len(prompt_decoded):]
                else:
                    response = full_output
                
                # Measure performance
                end_time = time.time()
                output_length = len(response.split())
                monitor_generation_stats(model_name, start_time, end_time, input_length, output_length)
                
                return response.strip()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ GPU OUT OF MEMORY: {e}")
                    # Attempt to recover by clearing memory
                    clear_memory()
                    return f"[Error: GPU out of memory when generating with {model_name}. Try a smaller max_tokens value.]"
                else:
                    print(f"❌ Runtime error: {e}")
                    return f"[Error generating with {model_name}: {str(e)}]"
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
                return f"[Error generating with {model_name}: {str(e)}]"
                
        elif model_name_lower == "gemini":
            # Gemini API-based generation
            try:
                # Get API key from environment variables
                api_key = os.getenv("GEMINI_API_KEY")
                
                if not api_key:
                    return "[Error: GEMINI_API_KEY not found in environment variables]"
                
                # Ensure API is configured
                genai.configure(api_key=api_key)
                
                # Use gemini-1.5-flash by default for better performance/cost ratio
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Calculate input length for statistics
                input_length = len(prompt.split())
                
                # Generate content with proper error handling
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": 0.95
                    }
                )
                
                # Ensure response is valid
                if hasattr(response, 'text') and response.text:
                    result = response.text
                elif hasattr(response, 'parts'):
                    # Fallback for some API versions
                    result = ''.join(part.text for part in response.parts)
                else:
                    return "[Error: Gemini API returned an empty response]"
                
                # Measure performance
                end_time = time.time()
                output_length = len(result.split())
                monitor_generation_stats("gemini", start_time, end_time, input_length, output_length)
                
                return result.strip()
                
            except Exception as e:
                print(f"❌ Gemini API error: {e}")
                
                # Specific error handling for common Gemini API issues
                error_str = str(e).lower()
                if any(term in error_str for term in ["quota", "rate limit", "429"]):
                    return "[Error: Gemini API quota exceeded or rate limited. Please try again later.]"
                elif any(term in error_str for term in ["500", "503", "internal"]):
                    return "[Error: Gemini API server error. Please try again later.]"
                else:
                    return f"[Error with Gemini API: {str(e)}]"
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"[Error: {str(e)}]" 