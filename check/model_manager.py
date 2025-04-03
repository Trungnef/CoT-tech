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
from functools import lru_cache
import logging
import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_loading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelManager")

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
MAX_GPU_MEMORY = float(os.getenv("MAX_GPU_MEMORY_GB", 140))
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

# Cache configuration
_CACHE_SIZE = 3  # Keep 3 models in cache
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_LAST_USED = {}
_CACHE_DIR = Path("./model_cache")
_CACHE_DIR.mkdir(exist_ok=True)

# Model configuration defaults
_MODEL_CONFIGS = {
    "llama": {
        "attn_implementation": "eager",
        "dtype": torch.bfloat16,
        "use_4bit": True,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    },
    "qwen": {
        "attn_implementation": "eager",
        "use_flash_attention_2": False,
        "sliding_window": None,
        "dtype": torch.bfloat16,
        "use_4bit": True,
        "max_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    },
    "gemini": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }
}

def get_model_config(model_name, custom_config=None):
    """
    Get configuration for a specific model, with optional custom overrides.
    
    Args:
        model_name (str): Name of the model ("llama", "qwen", "gemini")
        custom_config (dict): Optional custom configuration to override defaults
        
    Returns:
        dict: Configuration dictionary for the model
    """
    model_name = model_name.lower()
    
    # Check if model is supported
    if model_name not in _MODEL_CONFIGS:
        print(f"⚠️ Warning: No default configuration for model '{model_name}'")
        return custom_config or {}
    
    # Get default configuration
    config = _MODEL_CONFIGS[model_name].copy()
    
    # Apply custom overrides if provided
    if custom_config:
        for key, value in custom_config.items():
            config[key] = value
    
    return config

def update_model_config(model_name, new_config):
    """
    Update the default configuration for a specific model.
    
    Args:
        model_name (str): Name of the model ("llama", "qwen", "gemini")
        new_config (dict): New configuration parameters
    """
    model_name = model_name.lower()
    
    # Initialize if model doesn't exist in configs yet
    if model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = {}
    
    # Update configuration
    for key, value in new_config.items():
        _MODEL_CONFIGS[model_name][key] = value
    
    print(f"✅ Updated configuration for {model_name}")

def optimize_memory_config():
    """Optimize memory configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True
    )

@lru_cache(maxsize=_CACHE_SIZE)
def get_cached_model(model_key):
    """Get model from cache with LRU policy."""
    return _MODEL_CACHE.get(model_key)

def optimize_batch_processing(questions, prompt_fn, max_workers=None):
    """Optimize batch processing with parallel execution."""
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prompts = list(executor.map(lambda q: prompt_fn(q, "classical_problem"), questions))
    return prompts

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

def get_max_memory_config():
    """
    Tạo cấu hình bộ nhớ tối ưu cho các GPU và CPU.
    
    Returns:
        dict: Cấu hình bộ nhớ tối đa cho mỗi thiết bị
    """
    max_memory = {}
    
    try:
        # Xử lý bộ nhớ GPU nếu có
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                # Lấy tổng bộ nhớ GPU và tính toán bộ nhớ khả dụng (để lại 10% cho hệ thống)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                usable_memory = int(total_memory * 0.9)  # Để lại 10% cho hệ thống
                
                # Lưu cấu hình bộ nhớ tối đa cho GPU này
                max_memory[i] = f"{usable_memory}GiB"
                
            # Thêm cấu hình bộ nhớ CPU (thường dùng cho offload)
            max_memory["cpu"] = "32GiB"  # Giá trị mặc định cho CPU offload
        else:
            # Nếu không có GPU, sử dụng CPU để xử lý mô hình
            max_memory["cpu"] = "32GiB"
            
        # In thông tin cấu hình bộ nhớ
        print(f"Cấu hình bộ nhớ: {max_memory}")
            
    except Exception as e:
        print(f"Không thể tạo cấu hình bộ nhớ tối ưu: {e}")
        # Trả về None để sử dụng cấu hình mặc định
        return None
    
    return max_memory

def load_model_optimized(model_path, tokenizer_path, model_type="llama", dtype=torch.bfloat16, use_4bit=True, model_size_gb=70):
    """
    Load model and tokenizer with optimized memory usage and efficient configuration.
    
    Args:
        model_path: Path to model
        tokenizer_path: Path to tokenizer
        model_type: Type of model (llama, qwen, etc.)
        dtype: Data type for model weights
        use_4bit: Whether to use 4-bit quantization
        model_size_gb: Estimated model size in GB
        
    Returns:
        tuple: (tokenizer, model)
    """
    try:
        # Xóa bộ nhớ để giảm thiểu OOM errors
        clear_memory()
        
        # Thiết lập giá trị mặc định cho model
        device_map = "auto"
        attn_implementation = "eager"
        
        # Thiết lập cấu hình cho từng loại model
        if model_type.lower() == "qwen":
            print(f"📝 Configuring Qwen model with specific settings")
            attn_implementation = "eager"  # Changed from "sdpa" to "eager"
            device_map = "auto"
        elif model_type.lower() == "llama":
            print(f"📝 Configuring Llama model with specific settings")
            attn_implementation = "eager"  # Flash attention không được sử dụng cho Llama
            device_map = "auto"  # Auto device map tốt nhất cho Llama
        
        # Cấu hình memory cho mô hình
        max_memory = get_max_memory_config()
        
        # Load tokenizer trước
        print(f"⏳ Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        # Xử lý pad_token đặc biệt để tránh cảnh báo
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"ℹ️ Set pad_token=eos_token ({tokenizer.pad_token})")
            else:
                tokenizer.pad_token = tokenizer.unk_token
                print(f"ℹ️ Set pad_token=unk_token ({tokenizer.pad_token})")
        
        # Đảm bảo tất cả special tokens đều được thiết lập đúng
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        if not tokenizer.bos_token or not tokenizer.bos_token_id:
            # Sử dụng token đầu tiên làm BOS nếu không có
            if tokenizer.eos_token:
                tokenizer.bos_token = tokenizer.eos_token
                tokenizer.bos_token_id = tokenizer.eos_token_id
        
        # Cấu hình lượng tử hóa 4-bit nếu được bật
        print(f"⏳ Setting up model configuration")
        quantization_config = None
        
        # Thiết lập lượng tử hóa nếu được yêu cầu
        if use_4bit:
            print(f"⏳ Configuring 4-bit quantization")
            quantization_config = optimize_memory_config()
            
        # Load model với cấu hình tối ưu
        print(f"⏳ Loading model from {model_path}")
        
        # Tạo cấu hình model tùy theo loại model
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "quantization_config": quantization_config,
            "max_memory": max_memory,
            "trust_remote_code": True
        }
        
        # Thêm cấu hình riêng cho từng loại model
        if model_type.lower() == "llama":
            model_kwargs["attn_implementation"] = "eager"
            # Không thêm use_flash_attention vì gây lỗi
        else:
            model_kwargs["attn_implementation"] = attn_implementation
            model_kwargs["sliding_window"] = None  # Explicitly disable sliding window
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Thiết lập cấu hình generation mặc định
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.95
        model.generation_config.top_k = 40
        model.generation_config.num_beams = 1  # Greedy decoding with temperature
        model.generation_config.max_new_tokens = 1024
        model.generation_config.use_cache = True
        
        # Lưu vào cache
        cache_key = f"{model_type}_{model_path}"
        _MODEL_CACHE[cache_key] = model
        _TOKENIZER_CACHE[cache_key] = tokenizer
        _LAST_USED[cache_key] = time.time()
        
        # Ghi log bộ nhớ GPU
        print_memory_usage()
        
        print(f"✅ Successfully loaded {model_type} model and tokenizer")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()  # Xóa bộ nhớ sau lỗi
        return None, None

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
def generate_text_with_model(prompt, model_type="local", model_name=None, custom_config=None, prompt_type=None, **kwargs):
    """
    Generate text using specified model with improved performance monitoring.
    
    Args:
        prompt: Input prompt text
        model_type: Type of model interface to use ('local', 'gemini', etc.)
        model_name: Specific model name to use ('llama', 'qwen', etc.)
        custom_config: Custom configuration to override defaults
        prompt_type: Type of prompt being used (for logging)
        **kwargs: Additional parameters to override configuration
        
    Returns:
        str: Generated text
    """
    # Manage model cache to prevent OOM errors
    manage_model_cache(max_models=2)
    
    # Check input validity
    if not prompt or len(prompt.strip()) == 0:
        return "[Error: Empty prompt]"
    
    # Get the specific model name if not specified
    if model_name is None and model_type == "local":
        model_name = "llama"  # Default to llama if not specified
    elif model_type == "gemini":
        model_name = "gemini"
    
    prompt_info = f"(prompt: {prompt_type})" if prompt_type else ""
    print(f"🔄 Generating text with model: {model_name} {prompt_info}")
    
    # Get configuration for the model
    config = get_model_config(model_name, custom_config)
    
    # Override with kwargs
    for key, value in kwargs.items():
        config[key] = value
    
    # Record start time
    start_time = time.time()
    
    try:
        if model_type == "local":
            # Get model path from environment variables
            if model_name.lower() == "llama":
                model_path = os.getenv("LLAMA_MODEL_PATH")
                tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")
            elif model_name.lower() == "qwen":
                model_path = os.getenv("QWEN_MODEL_PATH")
                tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH")
            else:
                return f"[Error: Unsupported local model '{model_name}'. Valid options are: llama, qwen]"
            
            # Check if paths are valid
            if not model_path or not tokenizer_path:
                return f"[Error: Path not found for {model_name} model in .env]"
            
            # Use cache or load model
            cache_key = f"{model_name}_{model_path}"
            
            if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
                tokenizer = _TOKENIZER_CACHE[cache_key]
                model = _MODEL_CACHE[cache_key]
                _LAST_USED[cache_key] = time.time()
                print(f"👍 Using {model_name} model from cache")
            else:
                # Load model with configuration
                use_4bit = config.get("use_4bit", True)
                dtype = config.get("dtype", torch.bfloat16)
                tokenizer, model = load_model_optimized(model_path, tokenizer_path, model_name, dtype=dtype, use_4bit=use_4bit)
            
            # Check if model loaded successfully
            if model is None:
                return f"[Error: Failed to load {model_name} model]"
            
            # Generate text with improved error handling
            try:
                # Calculate input length for statistics
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                input_length = input_ids.size(1)
                
                # Get generation parameters from config
                max_new_tokens = config.get("max_tokens", 512)  # Use 512 as default if not specified
                temperature = config.get("temperature", 0.7)
                top_p = config.get("top_p", 0.95)
                top_k = config.get("top_k", 40)
                repetition_penalty = config.get("repetition_penalty", 1.1)
                
                # Log the generation parameters being used
                print(f"📊 Generation parameters for {model_name} (prompt: {prompt_type}):")
                print(f"  - max_new_tokens: {max_new_tokens}")
                print(f"  - temperature: {temperature}")
                print(f"  - top_p: {top_p}")
                print(f"  - repetition_penalty: {repetition_penalty}")
                
                # Generate with proper error handling and use the specified parameters
                with torch.inference_mode():
                    # Create attention mask explicitly since pad_token_id = eos_token_id
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                    
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,  # Thêm attention mask rõ ràng
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
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
                monitor_generation_stats(model_name, start_time, end_time, input_length, output_length, prompt_type)
                
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
                
        elif model_type == "gemini":
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
                
                # Get generation parameters from config
                max_output_tokens = config.get("max_tokens", 1024)
                temperature = config.get("temperature", 0.7)
                top_p = config.get("top_p", 0.95)
                top_k = config.get("top_k", 40)
                
                # Log the generation parameters being used
                print(f"📊 Generation parameters for Gemini (prompt: {prompt_type}):")
                print(f"  - max_output_tokens: {max_output_tokens}")
                print(f"  - temperature: {temperature}")
                print(f"  - top_p: {top_p}")
                print(f"  - top_k: {top_k}")
                
                # Generate content with proper error handling and use the specified max_tokens
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "top_p": top_p,
                        "top_k": top_k
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
                monitor_generation_stats("gemini", start_time, end_time, input_length, output_length, prompt_type)
                
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

def load_model_with_caching(model_name, model_path, tokenizer_path, use_4bit=True, max_memory=None):
    """Load model with caching and intelligent memory management."""
    # Check if model is already cached
    if model_name in _MODEL_CACHE and model_name in _TOKENIZER_CACHE:
        # Update last used time
        _LAST_USED[model_name] = time.time()
        logger.info(f"👍 Using {model_name} model from cache")
        return _TOKENIZER_CACHE[model_name], _MODEL_CACHE[model_name]
    
    # Log start time
    start_time = time.time()
    load_times = {"tokenizer": 0, "model_init": 0, "model_load": 0, "total": 0}
    logger.info(f"⏳ Loading {model_name} model from {model_path}")
    
    # If memory is low, offload least recently used models
    if should_offload_models():
        offload_least_used_models()
    
    # Clear GPU memory before loading new model
    clear_gpu_memory()
    
    try:
        # Load tokenizer first
        tokenizer_start = time.time()
        logger.info(f"⏳ Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side='left'  # Fix for decoder-only models
        )
        tokenizer_end = time.time()
        load_times["tokenizer"] = tokenizer_end - tokenizer_start
        logger.info(f"✅ Tokenizer loaded in {load_times['tokenizer']:.2f}s")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.pad_token = tokenizer.eos_token_id if tokenizer.eos_token_id else 0
                logger.info(f"Set pad_token to eos_token_id: {tokenizer.pad_token}")
        
        # Configure memory for model - use integers for GPU indices instead of 'cuda:0'
        if max_memory is None:
            # Create a simpler device map that works with newer transformers versions
            max_memory_start = time.time()
            max_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Use integer device IDs instead of 'cuda:X'
                    mem_size = int(torch.cuda.get_device_properties(i).total_memory / 1024**3 - 2)
                    max_memory[i] = f"{mem_size}GiB"
                    logger.info(f"GPU {i}: Allocated {mem_size}GB for model")
            max_memory["cpu"] = f"{CPU_OFFLOAD}GiB"
            logger.info(f"CPU: Allocated {CPU_OFFLOAD}GB for model offloading")
            logger.info(f"🔧 Memory configuration: {max_memory}")
            max_memory_end = time.time()
            logger.info(f"⏳ Memory configuration created in {max_memory_end - max_memory_start:.2f}s")
        
        # Configure model loading parameters
        model_config_start = time.time()
        # Load model with quantization, using "auto" device_map
        base_model_kwargs = {
            "device_map": "auto",
            "max_memory": max_memory,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "sliding_window": None,  # Ensure sliding window is disabled by default for eager
        }
        
        # Add model-specific parameters based on model type
        model_kwargs = base_model_kwargs.copy()
        
        # Llama models may not support sliding_window and some attention implementations
        if model_name.lower() == "llama":
            # Use only parameters supported by Llama
            model_kwargs["attn_implementation"] = "eager"
            # Do not add flash attention parameters that cause errors
            logger.info(f"Using eager attention implementation for {model_name}")
        else:
            # For other models like qwen, we also use eager attention
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["use_flash_attention_2"] = False
            model_kwargs["sliding_window"] = None  # Explicitly disable sliding window
            logger.info(f"Using eager attention implementation for {model_name} with sliding window disabled")
        
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                logger.info("Configuring 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_enable_fp32_cpu_offload=True  # Enable CPU offloading
                )
                model_kwargs["quantization_config"] = quantization_config
                logger.info("✅ 4-bit quantization config created")
            except ImportError:
                logger.warning("⚠️ Cannot use 4-bit quantization, bitsandbytes library not available")
        model_config_end = time.time()
        load_times["model_init"] = model_config_end - model_config_start
        logger.info(f"⏳ Model configuration initialized in {load_times['model_init']:.2f}s")
        
        # Log GPU memory before model loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - allocated
                logger.info(f"GPU {i} before loading: {free:.2f}GB free / {total:.2f}GB total")
        
        # Load model
        model_load_start = time.time()
        logger.info(f"⏳ Loading model weights from {model_path}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info("✅ Model loaded successfully with primary configuration")
        except Exception as e:
            logger.warning(f"❌ Error loading model with default config: {e}")
            logger.info("🔄 Retrying with simpler configuration...")
            
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
                simple_kwargs["attn_implementation"] = "eager"  # Changed from "sdpa" to "eager"
                simple_kwargs["sliding_window"] = None
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **simple_kwargs
                )
                logger.info("✅ Model loaded successfully with simplified configuration")
            except Exception as e2:
                logger.warning(f"❌ Error loading with simplified config: {e2}")
                logger.info("🔄 Final attempt with minimal configuration...")
                
                # Last attempt with minimal configuration
                minimal_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                }
                
                # Add attn_implementation with caution
                if model_name.lower() != "llama":
                    minimal_kwargs["attn_implementation"] = "eager"
                    minimal_kwargs["sliding_window"] = None
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **minimal_kwargs
                )
                logger.info("✅ Model loaded successfully with minimal configuration")
        
        model_load_end = time.time()
        load_times["model_load"] = model_load_end - model_load_start
        logger.info(f"✅ Model weights loaded in {load_times['model_load']:.2f}s")
        
        # Log GPU memory after model loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - allocated
                logger.info(f"GPU {i} after loading: {free:.2f}GB free / {total:.2f}GB total")
                logger.info(f"GPU {i} memory usage: {allocated:.2f}GB")
        
        # Store in cache
        _MODEL_CACHE[model_name] = model
        _TOKENIZER_CACHE[model_name] = tokenizer
        _LAST_USED[model_name] = time.time()
        
        # Save model state for faster reloading later
        cache_start = time.time()
        try:
            serialize_model(model, model_name)
            logger.info(f"✅ Model state serialized for future use")
        except Exception as e:
            logger.warning(f"⚠️ Could not serialize model state: {e}")
        cache_end = time.time()
        logger.info(f"⏳ Cache operations completed in {cache_end - cache_start:.2f}s")
        
        # Calculate total loading time
        end_time = time.time()
        load_times["total"] = end_time - start_time
        
        # Log detailed timing breakdown
        logger.info(f"📊 Model loading time breakdown:")
        logger.info(f"  - Tokenizer: {load_times['tokenizer']:.2f}s ({(load_times['tokenizer']/load_times['total']*100):.1f}%)")
        logger.info(f"  - Model config: {load_times['model_init']:.2f}s ({(load_times['model_init']/load_times['total']*100):.1f}%)")
        logger.info(f"  - Model weights: {load_times['model_load']:.2f}s ({(load_times['model_load']/load_times['total']*100):.1f}%)")
        logger.info(f"  - Other operations: {load_times['total'] - load_times['tokenizer'] - load_times['model_init'] - load_times['model_load']:.2f}s")
        logger.info(f"  - Total: {load_times['total']:.2f}s (100%)")
        
        print_memory_usage()
        
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"❌ Error loading {model_name} model: {e}")
        # Print traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def print_memory_usage():
    """In thông tin về bộ nhớ GPU và CPU hiện tại."""
    try:
        print("\n--- Thông tin bộ nhớ hiện tại ---")
        
        # Thông tin CPU
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"CPU Memory: {mem_info.rss / (1024**3):.2f} GB")
        
        # Thông tin GPU nếu có
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
    except Exception as e:
        print(f"Không thể in thông tin bộ nhớ: {e}")
    print("-----------------------------------\n")

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

def load_model(model_name, custom_config=None):
    """
    Load large language model with intelligent caching and custom configuration.
    
    Args:
        model_name (str): Name of model to load ('llama', 'qwen')
        custom_config (dict): Optional custom configuration parameters
        
    Returns:
        tuple: (tokenizer, model) for the requested model
    """
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
    
    # Get configuration for the model
    config = get_model_config(model_name_lower, custom_config)
    
    # Get parameters from config
    use_4bit = config.get("use_4bit", True)
    
    return load_model_with_caching(model_name_lower, model_path, tokenizer_path, use_4bit=use_4bit)

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
def monitor_generation_stats(model_name, start_time, end_time, input_length, output_length, prompt_type=None):
    """
    Theo dõi hiệu suất sinh văn bản của từng mô hình.
    
    Args:
        model_name (str): Tên mô hình
        start_time (float): Thời gian bắt đầu
        end_time (float): Thời gian kết thúc
        input_length (int): Độ dài đầu vào
        output_length (int): Độ dài đầu ra
        prompt_type (str): Loại prompt được sử dụng
    
    Returns:
        dict: Thống kê hiệu suất
    """
    elapsed = end_time - start_time
    tokens_per_second = output_length / elapsed if elapsed > 0 else 0
    
    stats = {
        "model_name": model_name,
        "prompt_type": prompt_type,
        "elapsed_time": elapsed,
        "input_length": input_length,
        "output_length": output_length,
        "tokens_per_second": tokens_per_second,
        "timestamp": time.time()
    }
    
    # Log thống kê
    prompt_info = f" (prompt: {prompt_type})" if prompt_type else ""
    print(f"📊 {model_name}{prompt_info} generation stats:")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - Output length: {output_length} tokens")
    print(f"  - Speed: {tokens_per_second:.2f} tokens/second")
    
    return stats 