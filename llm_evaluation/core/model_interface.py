"""
Model Interface quản lý tương tác với các mô hình LLM.
Hỗ trợ cả model local (Llama, Qwen) và API (Gemini, Groq).
Cung cấp interface thống nhất, caching và xử lý lỗi.
"""

import os
import sys
import time
import logging
from pathlib import Path
import torch
import gc
from functools import lru_cache
import datetime
import threading
import traceback
import hashlib
import pickle
import shutil
import json

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parents[1].absolute()))

# Import các module cần thiết
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import huggingface_hub

# API clients
import google.generativeai as genai

# Nếu Groq được sử dụng
try:
    import groq
except ImportError:
    pass

logger = logging.getLogger("model_interface")

# Thread-local storage cho model và tokenizer
thread_local = threading.local()

# Cache cho model và tokenizer
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_LAST_USED = {}  # Theo dõi lần cuối cùng model được sử dụng
_API_CLIENTS = {}  # Cache cho API clients
_RATE_LIMITERS = {}
_DISK_CACHE_INDEX = {}  # Lưu trữ thông tin về model cache trên disk

class ModelInterface:
    """
    Interface thống nhất cho việc tương tác với các model LLM,
    bao gồm cả model local (Llama, Qwen) và API (Gemini, Groq).
    """
    
    def __init__(self, use_cache=True, use_disk_cache=None):
        """
        Khởi tạo ModelInterface.
        
        Args:
            use_cache (bool): Có sử dụng cache model không
            use_disk_cache (bool): Có sử dụng disk cache không, mặc định theo cấu hình
        """
        self.use_cache = use_cache
        self.use_disk_cache = config.DISK_CACHE_CONFIG["enabled"] if use_disk_cache is None else use_disk_cache
        self._setup_rate_limiters()
        self._init_disk_cache()
        
    def _init_disk_cache(self):
        """Khởi tạo disk cache nếu được bật."""
        if not self.use_disk_cache:
            return
            
        global _DISK_CACHE_INDEX
        
        # Tạo thư mục cache nếu chưa tồn tại
        cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Tải index cache nếu tồn tại
        index_path = cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    _DISK_CACHE_INDEX = json.load(f)
                logger.info(f"Đã tải index cache với {len(_DISK_CACHE_INDEX)} model")
            except Exception as e:
                logger.error(f"Lỗi khi tải index cache: {e}")
                _DISK_CACHE_INDEX = {}
        
        # Dọn dẹp cache cũ nếu được cấu hình
        if config.DISK_CACHE_CONFIG.get("cleanup_on_startup", False):
            self._cleanup_disk_cache()
            
    def _setup_rate_limiters(self):
        """Thiết lập rate limiters cho các API."""
        global _RATE_LIMITERS
        
        for api_name, api_config in config.API_CONFIGS.items():
            requests_per_minute = api_config.get("requests_per_minute", 60)
            min_interval = 60.0 / requests_per_minute  # Khoảng thời gian tối thiểu giữa các request (giây)
            
            if api_name not in _RATE_LIMITERS:
                _RATE_LIMITERS[api_name] = {
                    "min_interval": min_interval,
                    "last_request_time": 0,
                    "lock": threading.Lock()
                }
        
        logger.debug("Đã thiết lập rate limiters cho các API.")
    
    def generate_text(self, model_name, prompt, config=None):
        """
        Sinh văn bản từ model được chỉ định với prompt và cấu hình cho trước.
        
        Args:
            model_name (str): Tên của model (llama, qwen, gemini, groq)
            prompt (str): Prompt input
            config (dict): Cấu hình generation (temperature, max_tokens, etc.)
            
        Returns:
            tuple: (text, stats)
                - text (str): Văn bản được sinh
                - stats (dict): Thống kê về quá trình sinh văn bản
        """
        logger.debug(f"Sinh văn bản cho {model_name} với prompt dài {len(prompt)} ký tự")
        
        # Chuẩn bị cấu hình mặc định từ config module
        import config as app_config
        model_config = app_config.MODEL_CONFIGS.get(model_name, {}).copy()
        
        # Ghi đè cấu hình nếu được cung cấp
        if config:
            model_config.update(config)
        
        # Xác định loại model và gọi phương thức phù hợp
        if model_name.lower() in ["llama", "qwen"]:
            return self._generate_with_local_model(model_name, prompt, model_config)
        elif model_name.lower() == "gemini":
            return self._generate_with_gemini(prompt, model_config)
        elif model_name.lower() == "groq":
            return self._generate_with_groq(prompt, model_config)
        else:
            error_msg = f"Model không được hỗ trợ: {model_name}"
            logger.error(error_msg)
            return f"[Error: {error_msg}]", {"has_error": True, "error_message": error_msg}
    
    def _generate_with_local_model(self, model_name, prompt, gen_config):
        """
        Sinh văn bản sử dụng model local (Llama, Qwen).
        
        Args:
            model_name (str): Tên model ('llama' hoặc 'qwen')
            prompt (str): Prompt đầu vào
            gen_config (dict): Cấu hình generation
            
        Returns:
            tuple: (text, stats)
        """
        start_time = time.time()
        
        try:
            # Tải tokenizer và model
            tokenizer, model = self._load_model(model_name)
            
            if tokenizer is None or model is None:
                error_msg = f"Không thể tải model {model_name}"
                logger.error(error_msg)
                return f"[Error: {error_msg}]", {
                    "has_error": True, 
                    "error_message": error_msg,
                    "elapsed_time": time.time() - start_time
                }
            
            # Mã hóa prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            input_length = input_ids.size(1)
            
            # Lấy tham số generation
            max_tokens = gen_config.get("max_tokens", 512)
            temperature = gen_config.get("temperature", 0.7)
            top_p = gen_config.get("top_p", 0.95)
            top_k = gen_config.get("top_k", 40)
            repetition_penalty = gen_config.get("repetition_penalty", 1.1)
            
            # Log tham số
            logger.debug(f"Tham số sinh cho {model_name}: max_tokens={max_tokens}, "
                        f"temp={temperature}, top_p={top_p}, top_k={top_k}")
            
            # Sinh văn bản
            start_generate = time.time()
            with torch.inference_mode():
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Giải mã output
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Chỉ lấy phần được sinh
            if full_output.startswith(prompt_decoded):
                generated_text = full_output[len(prompt_decoded):]
            else:
                generated_text = full_output
            
            # Tính thời gian và tốc độ
            end_time = time.time()
            generation_time = end_time - start_time
            decoding_time = end_time - start_generate
            output_length = len(generated_text.split())
            
            tokens_per_second = output_length / decoding_time if decoding_time > 0 else 0
            
            # Trả về văn bản và thống kê
            stats = {
                "token_count": output_length,
                "elapsed_time": generation_time,
                "decoding_time": decoding_time,
                "tokens_per_second": tokens_per_second,
                "has_error": False
            }
            
            return generated_text.strip(), stats
            
        except Exception as e:
            error_msg = f"Lỗi khi sinh văn bản với model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return f"[Error: {error_msg}]", {
                "has_error": True,
                "error_message": error_msg,
                "elapsed_time": time.time() - start_time
            }
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,))
    )
    def _generate_with_gemini(self, prompt, gen_config):
        """
        Sinh văn bản sử dụng Gemini API.
        
        Args:
            prompt (str): Prompt đầu vào
            gen_config (dict): Cấu hình generation
            
        Returns:
            tuple: (text, stats)
        """
        start_time = time.time()
        
        try:
            # Áp dụng rate limiting
            self._apply_rate_limiting("gemini")
            
            # Lấy Gemini client
            genai_client = self._get_gemini_client()
            if genai_client is None:
                error_msg = "Không thể kết nối tới Gemini API. Vui lòng kiểm tra API key."
                logger.error(error_msg)
                return f"[Error: {error_msg}]", {
                    "has_error": True,
                    "error_message": error_msg,
                    "error_type": "API_CONNECTION_ERROR",
                    "elapsed_time": time.time() - start_time
                }
            
            # Lấy tham số generation
            max_tokens = gen_config.get("max_tokens", 1024)
            temperature = gen_config.get("temperature", 0.7)
            top_p = gen_config.get("top_p", 0.95)
            top_k = gen_config.get("top_k", 40)
            
            # Get model to use
            model_name = gen_config.get("model", "gemini-1.5-flash")
            
            # Log tham số
            logger.debug(f"Tham số sinh cho Gemini: model={model_name}, max_tokens={max_tokens}, "
                        f"temp={temperature}, top_p={top_p}, top_k={top_k}")
            
            # Cấu hình generation
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
            
            # Tạo model instance từ genai client
            model = genai_client.GenerativeModel(model_name=model_name)
            
            # Sinh văn bản
            start_generate = time.time()
            
            # Set a timeout
            timeout = gen_config.get("timeout", 30)  # 30 seconds default
            response = model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            generated_text = response.text
            
            # Tính thời gian và tốc độ
            end_time = time.time()
            generation_time = end_time - start_time
            decoding_time = end_time - start_generate
            output_length = len(generated_text.split())
            
            tokens_per_second = output_length / decoding_time if decoding_time > 0 else 0
            
            # Trả về văn bản và thống kê
            stats = {
                "token_count": output_length,
                "elapsed_time": generation_time,
                "decoding_time": decoding_time,
                "tokens_per_second": tokens_per_second,
                "has_error": False,
                "model": model_name
            }
            
            return generated_text.strip(), stats
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Lỗi khi sinh văn bản với Gemini API: {str(e)}"
            logger.error(error_msg)
            if hasattr(e, 'status_code'):
                logger.error(f"Status code: {e.status_code}")
            logger.debug(traceback.format_exc())
            
            # Customized error message based on error type
            if 'quota' in str(e).lower() or 'rate limit' in str(e).lower():
                detail_msg = "Đã vượt quá giới hạn tỷ lệ của API. Đang thử lại với backoff..."
                error_type = "RATE_LIMIT_ERROR"
            elif 'invalid api key' in str(e).lower():
                detail_msg = "API key không hợp lệ. Vui lòng kiểm tra GEMINI_API_KEY trong .env"
                error_type = "INVALID_API_KEY_ERROR"
            elif 'timeout' in str(e).lower():
                detail_msg = "Request bị timeout. Đang thử lại..."
                error_type = "TIMEOUT_ERROR"
            elif 'block' in str(e).lower() or 'content filter' in str(e).lower():
                detail_msg = "Nội dung bị chặn bởi bộ lọc nội dung của Gemini. Thử điều chỉnh prompt."
                error_type = "CONTENT_FILTER_ERROR"
            else:
                detail_msg = "Lỗi không rõ. Đang thử lại..."
                error_type = "UNKNOWN_ERROR"
            
            logger.warning(f"Gemini API error: {detail_msg}")
            
            # Raise để retry được kích hoạt
            raise Exception(f"{error_type}: {detail_msg}. Original error: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(8),  # Tăng số lần thử từ 5 lên 8
        wait=wait_exponential(multiplier=1, min=4, max=120),  # Tăng thời gian chờ tối đa từ 60s lên 120s và min từ 2s lên 4s
        retry=retry_if_exception_type((Exception,))
    )
    def _generate_with_groq(self, prompt, gen_config):
        """
        Sinh văn bản sử dụng Groq API.
        
        Args:
            prompt (str): Prompt đầu vào
            gen_config (dict): Cấu hình generation
            
        Returns:
            tuple: (text, stats)
        """
        start_time = time.time()
        
        try:
            # Áp dụng rate limiting
            self._apply_rate_limiting("groq")
            
            # Lấy Groq client
            client = self._get_groq_client()
            if client is None:
                error_msg = "Không thể kết nối tới Groq API. Vui lòng kiểm tra API key."
                logger.error(error_msg)
                return f"[Error: {error_msg}]", {
                    "has_error": True,
                    "error_message": error_msg,
                    "error_type": "API_CONNECTION_ERROR",
                    "elapsed_time": time.time() - start_time
                }
            
            # Lấy tham số generation
            max_tokens = gen_config.get("max_tokens", 1024)
            temperature = gen_config.get("temperature", 0.7)
            top_p = gen_config.get("top_p", 0.95)
            model_name = gen_config.get("model", "llama3-70b-8192")
            
            # Log tham số
            logger.debug(f"Tham số sinh cho Groq: model={model_name}, max_tokens={max_tokens}, "
                        f"temp={temperature}, top_p={top_p}")
            
            # Sinh văn bản
            start_generate = time.time()
            
            # Set a timeout for the request
            timeout = gen_config.get("timeout", 30)  # 30 seconds default
            
            # Gọi Groq API
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            )
            
            # Trích xuất phản hồi
            generated_text = completion.choices[0].message.content
            
            # Tính thời gian và tốc độ
            end_time = time.time()
            generation_time = end_time - start_time
            decoding_time = end_time - start_generate
            output_length = len(generated_text.split())
            
            tokens_per_second = output_length / decoding_time if decoding_time > 0 else 0
            
            # Trả về văn bản và thống kê
            stats = {
                "token_count": output_length,
                "elapsed_time": generation_time,
                "decoding_time": decoding_time,
                "tokens_per_second": tokens_per_second,
                "has_error": False,
                "model": model_name,
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
            
            return generated_text.strip(), stats
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Lỗi khi sinh văn bản với Groq API: {str(e)}"
            logger.error(error_msg)
            if hasattr(e, 'status_code'):
                logger.error(f"Status code: {e.status_code}")
            logger.debug(traceback.format_exc())
            
            # Customized error message based on error type
            if 'quota' in str(e).lower() or 'rate limit' in str(e).lower():
                detail_msg = "Đã vượt quá giới hạn tỷ lệ của API Groq. Đang thử lại với backoff..."
                error_type = "RATE_LIMIT_ERROR"
            elif 'invalid api key' in str(e).lower() or 'authentication' in str(e).lower():
                detail_msg = "API key không hợp lệ. Vui lòng kiểm tra GROQ_API_KEY trong .env"
                error_type = "INVALID_API_KEY_ERROR"
            elif 'timeout' in str(e).lower():
                detail_msg = "Request bị timeout. Đang thử lại..."
                error_type = "TIMEOUT_ERROR"
            elif 'not found' in str(e).lower() and 'model' in str(e).lower():
                detail_msg = f"Model {model_name} không tồn tại hoặc không khả dụng."
                error_type = "MODEL_NOT_FOUND_ERROR"
            else:
                detail_msg = "Lỗi không rõ. Đang thử lại..."
                error_type = "UNKNOWN_ERROR"
                
            logger.warning(f"Groq API error: {detail_msg}")
            
            # Đối với một số lỗi nghiêm trọng, trả về ngay lập tức thay vì retry
            if error_type in ["INVALID_API_KEY_ERROR", "MODEL_NOT_FOUND_ERROR"]:
                return f"[Error: {detail_msg}]", {
                    "has_error": True,
                    "error_message": detail_msg,
                    "error_type": error_type,
                    "elapsed_time": time.time() - start_time
                }
            
            # Raise để retry được kích hoạt
            raise Exception(f"{error_type}: {detail_msg}. Original error: {str(e)}")
    
    def _apply_rate_limiting(self, api_name):
        """
        Áp dụng rate limiting cho API.
        
        Args:
            api_name (str): Tên API ('gemini', 'groq', etc.)
        """
        if api_name not in _RATE_LIMITERS:
            return
        
        rate_limiter = _RATE_LIMITERS[api_name]
        min_interval = rate_limiter["min_interval"]
        
        with rate_limiter["lock"]:
            # Tính thời gian cần chờ
            current_time = time.time()
            elapsed = current_time - rate_limiter["last_request_time"]
            wait_time = max(0, min_interval - elapsed)
            
            if wait_time > 0:
                logger.debug(f"Rate limiting: chờ {wait_time:.2f}s cho API {api_name}")
                time.sleep(wait_time)
            
            # Cập nhật thời gian gọi
            rate_limiter["last_request_time"] = time.time()
    
    def _load_model(self, model_name):
        """
        Tải model local (Llama hoặc Qwen).
        
        Args:
            model_name (str): Tên model ('llama' hoặc 'qwen')
            
        Returns:
            tuple: (tokenizer, model)
        """
        model_name = model_name.lower()
        cache_key = f"{model_name}_model"
        
        # Nếu cache được bật và model đã được tải vào memory
        if self.use_cache and cache_key in _MODEL_CACHE:
            logger.debug(f"Sử dụng {model_name} từ memory cache")
            _LAST_USED[cache_key] = time.time()
            return _TOKENIZER_CACHE.get(f"{model_name}_tokenizer"), _MODEL_CACHE.get(cache_key)
        
        # Giải phóng bộ nhớ trước khi tải model mới
        self._clear_memory()
        
        # Lấy đường dẫn cho model
        if model_name == "llama":
            model_path = config.LLAMA_MODEL_PATH
            tokenizer_path = config.LLAMA_TOKENIZER_PATH
        elif model_name == "qwen":
            model_path = config.QWEN_MODEL_PATH
            tokenizer_path = config.QWEN_TOKENIZER_PATH
        else:
            logger.error(f"Model không hợp lệ: {model_name}")
            return None, None
        
        # Kiểm tra đường dẫn
        if not model_path or not tokenizer_path:
            logger.error(f"Đường dẫn cho {model_name} không được cấu hình trong .env")
            return None, None
        
        logger.info(f"Tải model {model_name} từ {model_path}")
        
        # Biến để kiểm soát fallback sang CPU nếu GPU không đủ bộ nhớ
        use_gpu = torch.cuda.is_available()
        tried_cpu_fallback = False
        
        try:
            # Quản lý cache model để tránh OOM
            self._manage_model_cache()
            
            # Tải tokenizer
            logger.debug(f"Tải tokenizer từ {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Xử lý pad_token nếu cần
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.unk_token
            
            # THÊM: Tải model với vòng lặp thử các mức quantization khác nhau
            for attempt in range(3):  # 3 nỗ lực: 4-bit trên GPU -> 8-bit trên GPU -> CPU fallback
                try:
                    # Thiết lập các tham số dựa trên nỗ lực hiện tại
                    if attempt == 0:
                        # Nỗ lực đầu tiên: 4-bit trên GPU (nhẹ nhất)
                        use_4bit = True
                        use_8bit = False
                        use_gpu = torch.cuda.is_available()
                        logger.info(f"Nỗ lực tải model {model_name} với 4-bit quantization trên GPU")
                    elif attempt == 1:
                        # Nỗ lực thứ hai: 8-bit trên GPU (sử dụng nhiều bộ nhớ hơn)
                        use_4bit = False
                        use_8bit = True
                        use_gpu = torch.cuda.is_available()
                        logger.info(f"Thử lại: Tải model {model_name} với 8-bit quantization trên GPU")
                    else:
                        # Nỗ lực cuối: CPU fallback (chậm nhưng ít bộ nhớ)
                        use_4bit = False
                        use_8bit = False
                        use_gpu = False
                        logger.warning(f"Fallback: Tải model {model_name} trên CPU (sẽ chậm đáng kể)")
                    
                    # Đảm bảo thư mục offload tồn tại
                    offload_folder = Path(config.MODEL_CACHE_DIR) / "offload_folder"
                    offload_folder.mkdir(exist_ok=True, parents=True)
                    
                    # Cấu hình quantization nếu sử dụng GPU
                    quantization_config = None
                    if use_gpu:
                        if use_4bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                load_in_8bit=False,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_use_cpu_offload=True,
                                offload_folder=str(offload_folder)
                            )
                            logger.info(f"Sử dụng 4-bit quantization cho model {model_name}")
                        elif use_8bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=False,
                                load_in_8bit=True,
                                llm_int8_skip_modules=["lm_head"] if model_name.lower() == "qwen" else None,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=True,
                                offload_folder=str(offload_folder)
                            )
                            logger.info(f"Sử dụng 8-bit quantization cho model {model_name}")
                    
                    # Tạo device map & memory config
                    device_map = "auto" if use_gpu else "cpu"
                    max_memory = self._get_max_memory_config() if use_gpu else None
                    
                    # Cấu hình dựa trên model cụ thể
                    model_kwargs = {
                        "device_map": device_map,
                        "torch_dtype": torch.bfloat16 if use_gpu else torch.float32
                    }
                    
                    # Thêm cấu hình cho quantization nếu có
                    if quantization_config:
                        model_kwargs["quantization_config"] = quantization_config
                    
                    # Thêm cấu hình bộ nhớ nếu sử dụng GPU
                    if use_gpu and max_memory:
                        model_kwargs["max_memory"] = max_memory
                    
                    # Thêm cấu hình cho model cụ thể
                    if model_name == "llama":
                        model_kwargs["attn_implementation"] = "eager"
                    elif model_name == "qwen":
                        model_kwargs["attn_implementation"] = "eager"
                        model_kwargs["use_flash_attention_2"] = False
                        model_kwargs["trust_remote_code"] = True
                        
                        # Tắt cảnh báo về sliding window attention trong eager mode
                        if config.MODEL_CONFIGS["qwen"].get("disable_attention_warnings", False):
                            import transformers
                            transformers.logging.set_verbosity_error()
                            # Vô hiệu hóa cảnh báo cụ thể từ PyTorch
                            import warnings
                            warnings.filterwarnings("ignore", message=".*Sliding window attention is currently only supported.*")
                    
                    # Tải model
                    logger.debug(f"Tải model weights từ {model_path} vào {'GPU' if use_gpu else 'CPU'}...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    
                    # Lưu vào memory cache
                    _TOKENIZER_CACHE[f"{model_name}_tokenizer"] = tokenizer
                    _MODEL_CACHE[cache_key] = model
                    _LAST_USED[cache_key] = time.time()
                    
                    logger.info(f"Đã tải thành công model {model_name} vào {'GPU' if use_gpu else 'CPU'}")
                    
                    return tokenizer, model
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    # Xử lý lỗi OOM cụ thể
                    logger.error(f"CUDA Out of Memory khi tải model {model_name}: {str(oom_error)}")
                    logger.info("Giải phóng bộ nhớ GPU và thử lại với cấu hình nhẹ hơn")
                    self._clear_memory()  # Giải phóng bộ nhớ GPU
                    
                    # Nếu đây là lần thử cuối, ghi log và cho phép chuyển sang exception handler bên ngoài
                    if attempt == 2:
                        logger.error("Đã thử tất cả các cấu hình, không thể tải model")
                        raise oom_error
                    
                except Exception as e:
                    # Lỗi khác
                    logger.error(f"Lỗi khi tải model {model_name} (nỗ lực {attempt+1}/3): {str(e)}")
                    if "CUDA out of memory" in str(e):
                        logger.info("Phát hiện lỗi hết bộ nhớ CUDA, thử lại với cấu hình khác")
                        self._clear_memory()
                    else:
                        # Nếu không phải lỗi OOM, không cần thử lại
                        raise e
        
        except Exception as e:
            logger.error(f"Lỗi khi tải model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Nếu chưa thử fallback sang CPU và đây là lỗi OOM, thử lại trên CPU
            if use_gpu and not tried_cpu_fallback and ("CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError)):
                logger.warning(f"Thử fallback sang CPU cho model {model_name}")
                tried_cpu_fallback = True
                use_gpu = False
                
                try:
                    # Giải phóng bộ nhớ GPU
                    self._clear_memory()
                    
                    # Tải model trên CPU (không quantization)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=True if model_name == "qwen" else False
                    )
                    
                    # Lưu vào memory cache
                    _TOKENIZER_CACHE[f"{model_name}_tokenizer"] = tokenizer
                    _MODEL_CACHE[cache_key] = model
                    _LAST_USED[cache_key] = time.time()
                    
                    logger.info(f"Đã tải thành công model {model_name} vào CPU (fallback)")
                    return tokenizer, model
                    
                except Exception as cpu_error:
                    logger.error(f"Fallback CPU cũng thất bại cho model {model_name}: {str(cpu_error)}")
            
            return None, None
    
    def _load_model_from_disk(self, model_name):
        """
        Tải model từ disk cache.
        
        Args:
            model_name (str): Tên model
            
        Returns:
            tuple: (tokenizer, model) hoặc None nếu không tìm thấy trong cache
        """
        cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
        
        # Kiểm tra xem model có trong index không
        if model_name not in _DISK_CACHE_INDEX:
            return None
        
        cache_info = _DISK_CACHE_INDEX[model_name]
        tokenizer_path = cache_dir / f"{model_name}_tokenizer.pkl"
        model_config_path = cache_dir / f"{model_name}_config.json"
        model_state_path = cache_dir / f"{model_name}_state.pkl"
        
        # Kiểm tra các file tồn tại
        if not tokenizer_path.exists() or not model_config_path.exists() or not model_state_path.exists():
            logger.warning(f"Thiếu file cache cho {model_name}, xóa khỏi index")
            if model_name in _DISK_CACHE_INDEX:
                del _DISK_CACHE_INDEX[model_name]
            self._save_cache_index()
            return None
        
        # Kiểm tra thời gian cache nếu quá hạn TTL
        ttl = config.DISK_CACHE_CONFIG.get("ttl", 24 * 60 * 60)  # Mặc định: 24 giờ
        cache_age = time.time() - cache_info.get("timestamp", 0)
        if cache_age > ttl:
            logger.info(f"Cache cho {model_name} đã hết hạn (TTL: {ttl}s), tải lại")
            return None
        
        try:
            logger.info(f"Tải model {model_name} từ disk cache")
            
            # Quyết định sử dụng 4-bit hay 8-bit dựa trên cấu hình
            use_4bit = True  # Mặc định là 4-bit cho hiệu suất tốt nhất
            use_8bit = False
            
            # Kiểm tra thông tin GPU
            gpu_memory_total = 0
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                    gpu_memory_total += gpu_memory
                logger.info(f"Tổng bộ nhớ GPU: {gpu_memory_total:.2f} GB trên {gpu_count} GPU")
                
                # Đối với các model lớn (> 30B tham số), nếu tổng bộ nhớ > 80GB, sử dụng 8-bit để cải thiện chất lượng
                if model_name.lower() == "llama" and gpu_memory_total > 80:
                    use_8bit = True
                    use_4bit = False
                    logger.info(f"Sử dụng 8-bit quantization cho model {model_name} từ disk cache (GPU memory: {gpu_memory_total:.2f} GB)")
            
            # Đảm bảo thư mục offload tồn tại
            offload_folder = Path(config.MODEL_CACHE_DIR) / "offload_folder"
            offload_folder.mkdir(exist_ok=True, parents=True)
            
            # Cấu hình quantization
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_use_cpu_offload=True,
                    offload_folder=str(offload_folder)
                )
                logger.info(f"Sử dụng 4-bit quantization cho model {model_name} từ disk cache")
            elif use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=False,
                    load_in_8bit=True,
                    llm_int8_skip_modules=["lm_head"] if model_name.lower() == "qwen" else None,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True,
                    offload_folder=str(offload_folder)
                )
                logger.info(f"Sử dụng 8-bit quantization cho model {model_name} từ disk cache")
            
            # Tạo device map & memory config
            device_map = "auto"
            max_memory = self._get_max_memory_config()
            
            # Tải tokenizer từ pickle
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Tải cấu hình model
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            
            # Tạo model từ cấu hình
            model_class = model_config.get("class", "AutoModelForCausalLM")
            model_constructor = getattr(sys.modules["transformers"], model_class)
            
            # Tải model với cấu hình
            model_init_args = {
                "device_map": device_map,
                "max_memory": max_memory,
                "quantization_config": quantization_config,
                "torch_dtype": torch.bfloat16
            }
            
            # Thêm cấu hình cho model cụ thể
            if model_name == "llama":
                model_init_args["attn_implementation"] = "eager"
            elif model_name == "qwen":
                model_init_args["attn_implementation"] = "eager"
                model_init_args["use_flash_attention_2"] = False
                model_init_args["trust_remote_code"] = True
            
            # Tải model weights
            with open(model_state_path, 'rb') as f:
                model_state = pickle.load(f)
            
            # Tạo model
            model = model_constructor.from_pretrained(model_config.get("pretrained_path"), **model_init_args)
            
            # Load state dict
            model.load_state_dict(model_state)
            
            # Cập nhật thời gian truy cập
            _DISK_CACHE_INDEX[model_name]["last_accessed"] = time.time()
            self._save_cache_index()
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model {model_name} từ disk cache: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_model_to_disk(self, model_name, tokenizer, model):
        """
        Lưu model vào disk cache.
        
        Args:
            model_name (str): Tên model
            tokenizer: Tokenizer
            model: Model
            
        Returns:
            bool: True nếu lưu thành công
        """
        try:
            cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
            cache_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Lưu model {model_name} vào disk cache")
            
            # Lưu tokenizer
            tokenizer_path = cache_dir / f"{model_name}_tokenizer.pkl"
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            
            # Lưu cấu hình model
            model_config = {
                "class": model.__class__.__name__,
                "config": {
                    k: v for k, v in model.config.to_dict().items() 
                    if not k.startswith('_') and not callable(v)
                },
                "pretrained_path": model.name_or_path
            }
            
            model_config_path = cache_dir / f"{model_name}_config.json"
            with open(model_config_path, 'w') as f:
                json.dump(model_config, f)
            
            # Lưu state dict model
            model_state_path = cache_dir / f"{model_name}_state.pkl"
            with open(model_state_path, 'wb') as f:
                pickle.dump(model.state_dict(), f)
            
            # Cập nhật index
            _DISK_CACHE_INDEX[model_name] = {
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "tokenizer_path": str(tokenizer_path),
                "model_config_path": str(model_config_path),
                "model_state_path": str(model_state_path)
            }
            
            # Lưu index
            self._save_cache_index()
            
            # Quản lý số lượng model trong cache
            self._manage_disk_cache()
            
            logger.info(f"Đã lưu model {model_name} vào disk cache")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu model {model_name} vào disk cache: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _save_cache_index(self):
        """Lưu index cache vào file."""
        try:
            cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
            index_path = cache_dir / "cache_index.json"
            
            with open(index_path, 'w') as f:
                json.dump(_DISK_CACHE_INDEX, f)
                
        except Exception as e:
            logger.error(f"Lỗi khi lưu index cache: {e}")
    
    def _manage_disk_cache(self):
        """
        Quản lý số lượng model trong disk cache.
        Xóa các model ít được sử dụng nhất nếu vượt quá số lượng cho phép.
        """
        max_models = config.DISK_CACHE_CONFIG.get("max_cached_models", 2)
        
        # Nếu số lượng model trong cache ít hơn max_models, không cần xóa
        if len(_DISK_CACHE_INDEX) <= max_models:
            return
            
        # Sắp xếp model theo thời gian truy cập gần đây nhất
        sorted_models = sorted(
            [(name, info.get("last_accessed", 0)) for name, info in _DISK_CACHE_INDEX.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Giữ lại những model được sử dụng gần đây nhất
        models_to_keep = [model[0] for model in sorted_models[:max_models]]
        
        # Xóa các model ít sử dụng
        for model_name in list(_DISK_CACHE_INDEX.keys()):
            if model_name not in models_to_keep:
                logger.info(f"Xóa model {model_name} khỏi disk cache để dọn dẹp")
                self._remove_model_from_disk_cache(model_name)
    
    def _cleanup_disk_cache(self):
        """Xóa toàn bộ disk cache."""
        logger.info("Xóa toàn bộ disk cache")
        
        for model_name in list(_DISK_CACHE_INDEX.keys()):
            self._remove_model_from_disk_cache(model_name)
    
    def _remove_model_from_disk_cache(self, model_name):
        """
        Xóa model khỏi disk cache.
        
        Args:
            model_name (str): Tên model
        """
        if model_name not in _DISK_CACHE_INDEX:
            return
            
        cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
        
        # Xóa các file
        files_to_remove = [
            cache_dir / f"{model_name}_tokenizer.pkl",
            cache_dir / f"{model_name}_config.json",
            cache_dir / f"{model_name}_state.pkl"
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Lỗi khi xóa file {file_path}: {e}")
        
        # Xóa khỏi index
        del _DISK_CACHE_INDEX[model_name]
        
        # Lưu index
        self._save_cache_index()
    
    def _get_gemini_client(self):
        """
        Lấy hoặc tạo Gemini API client.
        
        Returns:
            Client: Gemini API client
        """
        if "gemini" in _API_CLIENTS:
            return _API_CLIENTS["gemini"]
        
        # Cấu hình Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Lưu client để tái sử dụng
        _API_CLIENTS["gemini"] = genai
        
        return genai
    
    def _get_groq_client(self):
        """
        Lấy hoặc tạo Groq API client.
        
        Returns:
            groq.Client: Groq client
        """
        if "groq" in _API_CLIENTS:
            return _API_CLIENTS["groq"]
        
        # Đảm bảo thư viện groq được cài đặt
        if 'groq' not in sys.modules:
            logger.error("Thư viện groq không được cài đặt")
            return None
        
        # Tạo client
        client = groq.Client(api_key=config.GROQ_API_KEY)
        _API_CLIENTS["groq"] = client
        
        return client
    
    def _clear_memory(self):
        """Giải phóng bộ nhớ GPU và CPU."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _manage_model_cache(self, max_models=2):
        """
        Quản lý cache model để tránh OOM.
        Giữ lại tối đa max_models model trong bộ nhớ.
        
        Args:
            max_models (int): Số lượng model tối đa để giữ trong cache
        """
        global _MODEL_CACHE, _LAST_USED
        
        # Nếu số lượng model trong cache ít hơn max_models, không cần offload
        if len(_MODEL_CACHE) <= max_models:
            return
        
        # Sắp xếp model theo thời gian sử dụng gần đây nhất
        sorted_models = sorted(_LAST_USED.items(), key=lambda x: x[1], reverse=True)
        
        # Giữ lại những model được sử dụng gần đây nhất
        models_to_keep = [model[0] for model in sorted_models[:max_models]]
        
        # Offload các model ít sử dụng
        for model_key in list(_MODEL_CACHE.keys()):
            if model_key not in models_to_keep:
                logger.info(f"Offload model {model_key} để tiết kiệm bộ nhớ")
                del _MODEL_CACHE[model_key]
                
                # Tokenizer chiếm ít bộ nhớ hơn, có thể giữ lại
                # tokenizer_key = model_key.replace("_model", "_tokenizer")
                # if tokenizer_key in _TOKENIZER_CACHE:
                #     del _TOKENIZER_CACHE[tokenizer_key]
                
                # Giải phóng bộ nhớ
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _get_max_memory_config(self):
        """
        Tạo cấu hình bộ nhớ tối ưu cho các GPU và CPU.
        Phân bổ bộ nhớ thông minh dựa trên số lượng GPU có sẵn.
        
        Returns:
            dict: Cấu hình bộ nhớ tối đa cho mỗi thiết bị
        """
        max_memory = {}
        
        try:
            # Xử lý bộ nhớ GPU nếu có
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                total_gpu_memory = 0
                gpu_info = []
                
                # Thu thập thông tin về mỗi GPU
                for i in range(num_gpus):
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    free_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3) - \
                                 torch.cuda.memory_reserved(i) / (1024**3)
                    
                    gpu_info.append({
                        'id': i,
                        'total_memory': total_memory,
                        'free_memory': free_memory,
                        'name': torch.cuda.get_device_properties(i).name
                    })
                    
                    total_gpu_memory += total_memory
                
                logger.info(f"Đã phát hiện {num_gpus} GPU với tổng bộ nhớ {total_gpu_memory:.2f} GB")
                for i, gpu in enumerate(gpu_info):
                    logger.info(f"GPU {i}: {gpu['name']}, {gpu['total_memory']:.2f} GB tổng, {gpu['free_memory']:.2f} GB trống")
                
                # Chiến lược phân bổ bộ nhớ
                if num_gpus >= 3:
                    # Đối với hệ thống nhiều GPU (3+), phân bổ nhiều bộ nhớ hơn cho GPU đầu tiên
                    # vì nó thường phải xử lý các lớp đầu vào và đầu ra cộng với một số lớp ẩn
                    for i in range(num_gpus):
                        reserved_memory = config.SYSTEM_RESERVE_MEMORY_GB
                        
                        if i == 0:
                            # GPU đầu tiên: để lại nhiều bộ nhớ hơn cho hệ thống
                            reserved_memory += 1.0
                        
                        usable_memory = max(1, int(gpu_info[i]['total_memory'] - reserved_memory))
                        max_memory[i] = f"{usable_memory}GiB"
                
                elif num_gpus == 2:
                    # Đối với hệ thống 2 GPU, cân bằng giữa chúng
                    for i in range(num_gpus):
                        reserved_memory = config.SYSTEM_RESERVE_MEMORY_GB
                        usable_memory = max(1, int(gpu_info[i]['total_memory'] - reserved_memory))
                        max_memory[i] = f"{usable_memory}GiB"
                
                elif num_gpus == 1:
                    # Đối với hệ thống 1 GPU, tối ưu hóa bộ nhớ cho GPU đơn
                    reserved_memory = config.SYSTEM_RESERVE_MEMORY_GB
                    usable_memory = max(1, int(gpu_info[0]['total_memory'] - reserved_memory))
                    max_memory[0] = f"{usable_memory}GiB"
                
                # Bộ nhớ CPU cho offload - tăng lên dựa trên số lượng GPU
                # Với nhiều GPU, chúng ta có thể giảm offload CPU
                if num_gpus >= 3:
                    cpu_offload = max(8, int(config.CPU_OFFLOAD_GB * 0.6))  # Giảm offload với 3+ GPU
                elif num_gpus == 2:
                    cpu_offload = max(12, int(config.CPU_OFFLOAD_GB * 0.8))  # Giảm nhẹ với 2 GPU
                else:
                    cpu_offload = config.CPU_OFFLOAD_GB  # Giữ nguyên với 1 GPU
                
                max_memory["cpu"] = f"{cpu_offload}GiB"
                
            else:
                # Nếu không có GPU, sử dụng CPU để xử lý mô hình
                max_memory["cpu"] = f"{config.CPU_OFFLOAD_GB}GiB"
            
            logger.info(f"Cấu hình bộ nhớ cho model: {max_memory}")
                
        except Exception as e:
            logger.error(f"Không thể tạo cấu hình bộ nhớ tối ưu: {str(e)}")
            logger.debug(traceback.format_exc())
            # Trả về None để sử dụng cấu hình mặc định
            return None
        
        return max_memory

    def get_response(self, model_name, prompt, max_tokens=1024):
        """
        Lấy response từ model với giao diện đơn giản hơn.
        
        Args:
            model_name (str): Tên của model
            prompt (str): Prompt đầu vào
            max_tokens (int): Số lượng token tối đa
            
        Returns:
            str: Phản hồi từ model
        """
        # Xử lý trường hợp đặc biệt "groq/model_name"
        actual_model_name = model_name
        model_config = {}
        
        if model_name.startswith("groq/"):
            # Trích xuất tên model thực tế 
            actual_model_name = "groq"
            model_config["model"] = model_name.replace("groq/", "")
            logger.debug(f"Đã chuyển đổi {model_name} -> {actual_model_name} với model config: {model_config}")
        
        config = {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Thêm model config nếu có
        if model_config:
            config.update(model_config)
            
        response, stats = self.generate_text(actual_model_name, prompt, config)
        
        if stats.get("has_error", False):
            logger.error(f"Lỗi khi lấy phản hồi từ {model_name}: {stats.get('error_message', 'Unknown error')}")
            return f"Error: {stats.get('error_message', 'Unknown error')}"
            
        return response

# Singleton để sử dụng từ bên ngoài
_model_interface = None

def get_model_interface(use_cache=True, use_disk_cache=None):
    """
    Lấy instance của ModelInterface (singleton).
    
    Args:
        use_cache: Có sử dụng memory cache không
        use_disk_cache: Có sử dụng disk cache không
        
    Returns:
        ModelInterface: Instance của ModelInterface
    """
    global _model_interface
    
    if _model_interface is None:
        _model_interface = ModelInterface(use_cache=use_cache, use_disk_cache=use_disk_cache)
    
    return _model_interface

def generate_text(model_name, prompt, generation_config=None):
    """
    Sinh văn bản từ model (hàm tiện ích).
    
    Args:
        model_name: Tên model
        prompt: Prompt đầu vào
        generation_config: Cấu hình sinh
        
    Returns:
        tuple: (text, stats)
    """
    interface = get_model_interface()
    return interface.generate_text(model_name, prompt, generation_config)

def clear_model_cache():
    """Xóa cache model."""
    global _MODEL_CACHE, _TOKENIZER_CACHE, _LAST_USED
    
    logger.info("Xóa cache model")
    
    # Xóa tất cả model từ memory
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()
    _LAST_USED.clear()
    
    # Giải phóng bộ nhớ
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

def get_available_models():
    """
    Lấy danh sách model có sẵn.
    
    Returns:
        dict: Danh sách model và trạng thái
    """
    models = {}
    
    # Model local
    if config.LLAMA_MODEL_PATH:
        models["llama"] = {
            "type": "local",
            "cached": "llama_model" in _MODEL_CACHE,
            "disk_cached": "llama" in _DISK_CACHE_INDEX
        }
    
    if config.QWEN_MODEL_PATH:
        models["qwen"] = {
            "type": "local",
            "cached": "qwen_model" in _MODEL_CACHE,
            "disk_cached": "qwen" in _DISK_CACHE_INDEX
        }
    
    # Model API
    if config.GEMINI_API_KEY:
        models["gemini"] = {
            "type": "api",
            "api": "gemini"
        }
    
    if config.GROQ_API_KEY:
        models["groq"] = {
            "type": "api",
            "api": "groq"
        }
    
    return models
