"""
Giao diện thống nhất để tương tác với các model LLM khác nhau bao gồm cả model local và API.

Chức năng:
- Tạo ra giao diện thống nhất cho cả model local (Llama, Qwen) và API (Gemini, Groq)
- Quản lý bộ nhớ và cache model để tránh OOM
- Xử lý rate limiting và error handling với circuit breaker
- Quản lý inference trên nhiều GPU

Cải tiến:
- [2025-04-19] Thêm cơ chế retry thông minh với backoff thích ứng:
  * Jitter ngẫu nhiên để tránh thundering herd
  * Học từ lịch sử lỗi và điều chỉnh chiến lược retry
  * Xác định thời gian retry dựa trên header và nội dung lỗi
  * Circuit breaker cải tiến với phát hiện cách mở/đóng thông minh
  * Rate limiting thích ứng dựa trên thành công/thất bại liên tiếp
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
import random
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parents[1].absolute()))

# Import các module cần thiết
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_fixed, retry_if_result
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
_CIRCUIT_BREAKERS = {}  # Theo dõi lỗi rate limit để tạm dừng API khi cần

def create_smart_retry(api_name):
    """
    Tạo một retry decorator thông minh với các tính năng nâng cao:
    - Jitter ngẫu nhiên để tránh thundering herd
    - Chỉ retry cho các mã lỗi cụ thể
    - Tích hợp với circuit breaker
    - Backoff thích ứng dựa trên kiểu lỗi
    
    Args:
        api_name (str): Tên của API ('gemini', 'groq', etc.)
        
    Returns:
        Function: Retry decorator tùy chỉnh
    """
    # Lấy cấu hình từ config.py
    cfg = config.API_CONFIGS.get(api_name, {})
    max_retries = cfg.get("max_retries", 5)
    base_delay = cfg.get("retry_base_delay", 2)
    max_delay = cfg.get("max_retry_delay", 60)
    jitter_factor = cfg.get("jitter_factor", 0.25)
    error_codes = cfg.get("error_codes_to_retry", [429, 500, 502, 503, 504])
    
    def should_retry_exception(exception):
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            bool: Whether to retry the request
        """
        # Check for HTTP status codes in various exception types
        status_code = None
        
        # Extract status code from different exception types
        if hasattr(exception, 'status_code'):
            status_code = exception.status_code
        elif hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            status_code = exception.response.status_code
        elif hasattr(exception, 'code'):
            status_code = exception.code
        
        # Always retry on connection errors and timeouts
        if isinstance(exception, (ConnectionError, TimeoutError)) or 'timeout' in str(exception).lower():
            return True
            
        # Always retry on rate limit errors (429)
        if status_code == 429:
            return True
        
        # Always retry on bad gateway (502), service unavailable (503), and gateway timeout (504)
        if status_code in (502, 503, 504):
            return True
            
        # Retry on specific error messages
        error_msg = str(exception).lower()
        retry_keywords = [
            'rate limit', 
            'too many requests',
            'timeout', 
            'connection', 
            'socket',
            'reset',
            'broken pipe',
            'server overloaded',
            'bad gateway',
            'service unavailable',
            'gateway timeout',
            'internal server error'
        ]
        
        if any(keyword in error_msg for keyword in retry_keywords):
            return True
            
        # Check if it's a transient API error
        if 'try again' in error_msg or 'please retry' in error_msg:
            return True
        
        # Don't retry on authentication errors or input validation errors
        if status_code in (400, 401, 403, 404, 422):
            return False
            
        # By default, don't retry
        return False
    
    def wait_with_jitter(retry_state):
        """
        Calculate wait time using exponential backoff with jitter.
        
        Args:
            retry_state: The current retry state
            
        Returns:
            float: Number of seconds to wait
        """
        # Get retry attempt number (starting from 0)
        retry_number = retry_state.attempt_number - 1
        
        # Base delay (in seconds)
        base_delay = 1.0
        
        # Max delay (in seconds) - cap at 60 seconds
        max_delay = 60.0
        
        # Calculate exponential backoff
        exponential_delay = min(max_delay, base_delay * (2 ** retry_number))
        
        # Add jitter (random value between 0 and exponential_delay * 0.5)
        jitter = random.uniform(0, exponential_delay * 0.5)
        
        # If the exception contains a Retry-After header, respect it
        exception = retry_state.outcome.exception()
        if exception is not None:
            retry_after = None
            
            # Try to extract Retry-After header from different exception types
            if hasattr(exception, 'headers') and 'Retry-After' in exception.headers:
                retry_after = exception.headers['Retry-After']
            elif hasattr(exception, 'response') and hasattr(exception.response, 'headers') and 'Retry-After' in exception.response.headers:
                retry_after = exception.response.headers['Retry-After']
            
            # If we found a Retry-After header, parse and use it (with some additional jitter)
            if retry_after is not None:
                try:
                    # Retry-After can be a number of seconds or an HTTP date
                    if retry_after.isdigit():
                        # It's a number of seconds
                        retry_after_seconds = float(retry_after)
                        # Add a small random jitter (0-1 seconds)
                        return retry_after_seconds + random.uniform(0, 1)
                    else:
                        # It might be an HTTP date format
                        # Parse the date and calculate seconds from now
                        from email.utils import parsedate_to_datetime
                        retry_date = parsedate_to_datetime(retry_after)
                        now = datetime.datetime.now(datetime.timezone.utc)
                        wait_seconds = max(0, (retry_date - now).total_seconds())
                        return wait_seconds + random.uniform(0, 1)
                except (ValueError, TypeError):
                    # If parsing fails, fall back to exponential backoff
                    pass
        
        # Log retry attempt with wait time
        from llm_evaluation.utils.logging_setup import get_logger
        logger = get_logger("model_interface")
        logger.info(f"Retry attempt {retry_number + 1} for {exception.__class__.__name__}. Waiting {exponential_delay + jitter:.2f} seconds...")
        
        return exponential_delay + jitter
    
    # Tạo decorator tùy chỉnh
    retry_decorator = retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_with_jitter,
        retry=retry_if_exception_type((Exception,)) & retry_if_result(should_retry_exception),
        before_sleep=lambda retry_state: logger.info(f"Đang chờ {retry_state.next_action.sleep} giây trước khi thử lại...")
    )
    
    return retry_decorator

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
        
        # Đọc cấu hình disk cache từ .env thông qua config
        self.use_disk_cache = config.DISK_CACHE_CONFIG["enabled"] if use_disk_cache is None else use_disk_cache
        
        # Log trạng thái disk cache
        if self.use_disk_cache:
            logger.info("Disk cache đang được BẬT. Các model sẽ được lưu và tải từ disk cache khi cần.")
        else:
            logger.info("Disk cache đang TẮT. Model sẽ được tải trực tiếp từ đường dẫn đã cấu hình.")
            
        # Khởi tạo quản lý API keys
        self.current_gemini_key_index = 0
        self.current_groq_key_index = 0
        
        # Thay thế set bằng dict để lưu thông tin hết hạn
        # Key: API key, Value: Dictionary {timestamp: thời gian hết hạn, reason: lý do}
        self.exhausted_gemini_keys = {}
        self.exhausted_groq_keys = {}
        
        # Tần suất kiểm tra key hết hạn (giây)
        self.key_refresh_interval = 3600  # 1 giờ
        self.last_key_refresh_time = time.time()
        
        self._setup_rate_limiters()
        
        # Chỉ khởi tạo disk cache nếu được bật
        if self.use_disk_cache:
            self._init_disk_cache()
        else:
            logger.debug("Bỏ qua việc khởi tạo disk cache vì đã tắt")
    
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
        """Thiết lập rate limiters cho các API với khả năng thích ứng động."""
        global _RATE_LIMITERS, _CIRCUIT_BREAKERS
        
        for api_name, api_config in config.API_CONFIGS.items():
            requests_per_minute = api_config.get("requests_per_minute", 60)
            min_interval = 60.0 / requests_per_minute  # Khoảng thời gian tối thiểu giữa các request (giây)
            adaptive_mode = api_config.get("adaptive_rate_limiting", False)
            
            if api_name not in _RATE_LIMITERS:
                _RATE_LIMITERS[api_name] = {
                    "min_interval": min_interval,
                    "last_request_time": 0,
                    "lock": threading.Lock(),
                    "adaptive_interval": min_interval,  # Khoảng thời gian tự động điều chỉnh
                    "backoff_factor": 1.0,  # Hệ số tăng interval khi gặp rate limit
                    "success_counter": 0,   # Đếm số lần thành công liên tiếp
                    "failure_counter": 0,   # Đếm số lần thất bại liên tiếp
                    "adaptive_mode": adaptive_mode,  # Có sử dụng chế độ thích ứng không
                    "last_rate_limit_time": 0, # Thời điểm gặp rate limit gần nhất
                    "retry_delay_history": [],  # Lịch sử các thời gian retry để học
                    "recovery_factor": 0.95,  # Hệ số giảm interval sau mỗi lần thành công (95%)
                    "base_interval": min_interval  # Lưu trữ interval ban đầu
                }
                logger.info(f"Đã thiết lập rate limiter cho {api_name} với {requests_per_minute} RPM " +
                           f"(interval: {min_interval:.2f}s, adaptive mode: {adaptive_mode})")
            else:
                # Cập nhật cấu hình nếu rate limiter đã tồn tại
                with _RATE_LIMITERS[api_name]["lock"]:
                    _RATE_LIMITERS[api_name]["min_interval"] = min_interval
                    _RATE_LIMITERS[api_name]["base_interval"] = min_interval
                    _RATE_LIMITERS[api_name]["adaptive_mode"] = adaptive_mode
                    # Nếu interval hiện tại nhỏ hơn min_interval mới, điều chỉnh lên
                    if _RATE_LIMITERS[api_name]["adaptive_interval"] < min_interval:
                        _RATE_LIMITERS[api_name]["adaptive_interval"] = min_interval
            
            # Đọc cấu hình circuit breaker từ config
            circuit_breaker_config = api_config.get("circuit_breaker", {})
            failure_threshold = circuit_breaker_config.get("failure_threshold", 5)
            cooldown_period = circuit_breaker_config.get("cooldown_period", 60.0)
            half_open_timeout = circuit_breaker_config.get("half_open_timeout", 30.0)
            consecutive_success_threshold = circuit_breaker_config.get("consecutive_success_threshold", 2)
            
            if api_name not in _CIRCUIT_BREAKERS:
                _CIRCUIT_BREAKERS[api_name] = {
                    "failures": 0,  # Số lần lỗi rate limit liên tiếp
                    "last_failure_time": 0,  # Thời điểm lỗi rate limit gần nhất
                    "is_open": False,  # Circuit breaker có đang mở không (tạm dừng API)
                    "cooldown_period": cooldown_period,  # Thời gian cooldown (giây)
                    "failure_threshold": failure_threshold,  # Ngưỡng lỗi để mở circuit breaker
                    "half_open_time": 0,  # Thời điểm circuit breaker chuyển sang half-open
                    "half_open_timeout": half_open_timeout,  # Thời gian timeout cho trạng thái half-open
                    "consecutive_successes": 0,  # Số lần thành công liên tiếp trong trạng thái half-open
                    "consecutive_success_threshold": consecutive_success_threshold,  # Ngưỡng để đóng circuit breaker
                    "lock": threading.Lock()  # Lock để đồng bộ hóa
                }
            else:
                # Cập nhật cấu hình nếu circuit breaker đã tồn tại
                with _CIRCUIT_BREAKERS[api_name]["lock"]:
                    _CIRCUIT_BREAKERS[api_name]["cooldown_period"] = cooldown_period
                    _CIRCUIT_BREAKERS[api_name]["failure_threshold"] = failure_threshold
                    _CIRCUIT_BREAKERS[api_name]["half_open_timeout"] = half_open_timeout
                    _CIRCUIT_BREAKERS[api_name]["consecutive_success_threshold"] = consecutive_success_threshold
        
        logger.debug("Đã thiết lập rate limiters và circuit breakers cho các API.")
    
    def generate_text(self, model_name, prompt, config=None):
        """
        Sinh văn bản từ model được chỉ định với prompt và cấu hình cho trước.
        
        Args:
            model_name (str): Tên của model (llama, qwen, gemini, groq)
            prompt (str): Prompt input
            config (dict): Cấu hình generation (temperature, max_tokens, etc.) 
                           Có thể chứa prompt_type để lấy max_tokens phù hợp từ cấu hình
            
        Returns:
            tuple: (text, stats)
                - text (str): Văn bản được sinh
                - stats (dict): Thống kê về quá trình sinh văn bản
        """
        logger.debug(f"Sinh văn bản cho {model_name} với prompt dài {len(prompt)} ký tự")
        
        # Chuẩn bị cấu hình mặc định từ config module
        import config as app_config
        model_config = app_config.MODEL_CONFIGS.get(model_name, {}).copy()
        
        # Xử lý prompt_type nếu có để lấy max_tokens phù hợp
        if config and 'prompt_type' in config:
            prompt_type = config.pop('prompt_type')  # Lấy và xóa prompt_type khỏi config
            # Lấy max_tokens phù hợp từ cấu hình
            model_config['max_tokens'] = app_config.get_max_tokens(model_name, prompt_type)
            logger.debug(f"Sử dụng max_tokens={model_config['max_tokens']} cho {model_name}/{prompt_type}")
        
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
    
    @property
    def gemini_retry_decorator(self):
        """Tạo decorator retry cho Gemini tại runtime để đọc cấu hình mới nhất"""
        return create_smart_retry("gemini")
        
    def _generate_with_gemini(self, prompt, gen_config):
        """
        Sinh văn bản sử dụng Gemini API.
        
        Args:
            prompt (str): Prompt đầu vào
            gen_config (dict): Cấu hình generation
            
        Returns:
            tuple: (text, stats)
        """
        # Áp dụng decorator tại runtime để đảm bảo cấu hình mới nhất
        decorated_function = self.gemini_retry_decorator(self._generate_with_gemini_impl)
        return decorated_function(prompt, gen_config)
        
    def _generate_with_gemini_impl(self, prompt, gen_config):
        """
        Triển khai thực tế của việc gọi Gemini API.
        """
        start_time = time.time()
        api_name = "gemini"
        
        try:
            # Áp dụng rate limiting với circuit breaker
            if not self._apply_rate_limiting(api_name):
                # Circuit breaker đang mở, raise exception để trigger retry
                raise Exception("Circuit breaker đang mở, đang chờ cooldown")
            
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
            
            # Đặt lại circuit breaker sau khi request thành công
            self._reset_circuit_breaker(api_name)
            
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
            
            # Trích xuất retry-after header nếu có
            retry_after = None
            if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                try:
                    retry_after = float(e.headers['retry-after'])
                    logger.info(f"Tìm thấy header Retry-After: {retry_after}s")
                except (ValueError, TypeError):
                    pass
            
            # Customized error message based on error type
            if 'quota' in str(e).lower() or 'rate limit' in str(e).lower() or hasattr(e, 'status_code') and getattr(e, 'status_code') == 429:
                # Đánh dấu key hiện tại đã hết quota
                current_key = config.GEMINI_API_KEYS[self.current_gemini_key_index]
                
                # Xác định loại lỗi quota: hết quota theo ngày hay chỉ là rate limit tạm thời
                is_daily_quota = False
                reason = "rate_limit_exceeded"
                
                # Phân tích lỗi để xác định đúng loại lỗi quota
                error_str = str(e).lower()
                if 'daily' in error_str and 'quota' in error_str:
                    is_daily_quota = True
                    reason = "daily_quota_exceeded"
                elif 'quota exceeded' in error_str or 'quota limit' in error_str:
                    is_daily_quota = True
                    reason = "daily_quota_exceeded"
                
                # Lưu thông tin hết hạn với timestamp hiện tại
                self.exhausted_gemini_keys[current_key] = {
                    'timestamp': time.time(),
                    'reason': reason,
                    'error': str(e)
                }
                
                # Chuyển sang key tiếp theo
                old_index = self.current_gemini_key_index
                self.current_gemini_key_index = (self.current_gemini_key_index + 1) % len(config.GEMINI_API_KEYS)
                
                # Kiểm tra xem còn key khả dụng không
                if len(self.exhausted_gemini_keys) >= len(config.GEMINI_API_KEYS):
                    detail_msg = "Tất cả Gemini API keys đều đã vượt quá quota. Vui lòng thử lại sau."
                    logger.error(detail_msg)
                else:
                    # Nếu còn key khả dụng, thử lại với key mới
                    quota_type = "theo ngày" if is_daily_quota else "tạm thời"
                    detail_msg = f"Key #{old_index + 1} đã hết quota {quota_type}. Chuyển sang key #{self.current_gemini_key_index + 1}/{len(config.GEMINI_API_KEYS)}"
                    logger.warning(detail_msg)
                    
                    # Khởi tạo lại client với key mới
                    self._get_gemini_client()
                    
                    # Giảm thời gian chờ vì chúng ta đã đổi key
                    wait_time = 1.0  # 1 second để đảm bảo không quá nhanh
                
                error_type = "QUOTA_LIMIT_ERROR"
                
                # Thực hiện sleep ngay tại đây để đảm bảo chờ đủ thời gian
                time.sleep(min(wait_time, 5))  # Giới hạn tối đa 5s khi đổi key
                
            elif 'invalid api key' in str(e).lower():
                # Đánh dấu key hiện tại không hợp lệ
                current_key = config.GEMINI_API_KEYS[self.current_gemini_key_index]
                
                # Lưu thông tin lỗi với timestamp hiện tại
                self.exhausted_gemini_keys[current_key] = {
                    'timestamp': time.time(),
                    'reason': "invalid_api_key",
                    'error': str(e)
                }
                
                # Chuyển sang key tiếp theo
                self.current_gemini_key_index = (self.current_gemini_key_index + 1) % len(config.GEMINI_API_KEYS)
                
                # Kiểm tra xem còn key khả dụng không
                if len(self.exhausted_gemini_keys) >= len(config.GEMINI_API_KEYS):
                    detail_msg = "Tất cả Gemini API keys đều không hợp lệ. Vui lòng kiểm tra cấu hình."
                else:
                    detail_msg = f"Key không hợp lệ. Chuyển sang key #{self.current_gemini_key_index + 1}/{len(config.GEMINI_API_KEYS)}"
                    
                    # Khởi tạo lại client với key mới
                    self._get_gemini_client()
                
                error_type = "INVALID_API_KEY_ERROR"
            elif 'timeout' in str(e).lower():
                detail_msg = "Request bị timeout. Đang thử lại..."
                error_type = "TIMEOUT_ERROR"
            elif 'block' in str(e).lower() or 'content filter' in str(e).lower() or 'safety' in str(e).lower():
                detail_msg = "Nội dung bị chặn bởi bộ lọc nội dung của Gemini. Thử điều chỉnh prompt."
                error_type = "CONTENT_FILTER_ERROR"
            elif 'circuit breaker' in str(e).lower():
                detail_msg = "Circuit breaker đang mở, đang chờ cooldown"
                error_type = "CIRCUIT_BREAKER_OPEN"
            else:
                detail_msg = "Lỗi không rõ. Đang thử lại..."
                error_type = "UNKNOWN_ERROR"
            
            logger.warning(f"Gemini API error: {detail_msg}")
            
            # Đối với một số lỗi nghiêm trọng, trả về ngay lập tức thay vì retry
            if error_type in ["INVALID_API_KEY_ERROR", "CONTENT_FILTER_ERROR"]:
                return f"[Error: {detail_msg}]", {
                    "has_error": True,
                    "error_message": detail_msg,
                    "error_type": error_type,
                    "elapsed_time": time.time() - start_time
                }
            
            # Raise để retry được kích hoạt
            raise Exception(f"{error_type}: {detail_msg}. Original error: {str(e)}")
    
    @property
    def groq_retry_decorator(self):
        """Tạo decorator retry cho Groq tại runtime để đọc cấu hình mới nhất"""
        return create_smart_retry("groq")
        
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
        max_retries = 5  # Số lần thử lại tối đa
        
        # Áp dụng decorator tại runtime để đảm bảo cấu hình mới nhất
        decorated_function = self.groq_retry_decorator(self._generate_with_groq_impl)
        
        for attempt in range(max_retries):
            try:
                return decorated_function(prompt, gen_config)
            except Exception as e:
                error_msg = str(e)
                
                # Trích xuất loại lỗi
                error_type = None
                if ":" in error_msg:
                    error_type = error_msg.split(":")[0].strip()
                
                # Kiểm tra nếu đã đến lần retry cuối cùng
                if attempt == max_retries - 1:
                    logger.error(f"Đã thử lại tối đa {max_retries} lần nhưng vẫn thất bại. Lỗi cuối cùng: {error_msg}")
                    
                    # Trả về thông báo lỗi thay vì throw exception
                    detail_msg = error_msg.split(".")[0] if "." in error_msg else error_msg
                    return f"[Error: {detail_msg}]", {
                        "has_error": True,
                        "error_message": detail_msg,
                        "error_type": error_type or "UNKNOWN_ERROR",
                        "elapsed_time": time.time() - start_time
                    }
                
                # Kiểm tra các lỗi đặc biệt không thể recover
                if "INVALID_API_KEY_ERROR" in error_msg and "Tất cả Groq API keys đều không hợp lệ" in error_msg:
                    logger.error("Tất cả API keys đều không hợp lệ, không thể tiếp tục retry")
                    return f"[Error: Tất cả API keys không hợp lệ]", {
                        "has_error": True,
                        "error_message": "Tất cả API keys không hợp lệ",
                        "error_type": "INVALID_API_KEY_ERROR",
                        "elapsed_time": time.time() - start_time
                    }
                
                # Các lỗi đã được xử lý ở _generate_with_groq_impl, 
                # decorator sẽ tự động retry
                logger.warning(f"Đang thử lại lần {attempt + 1}/{max_retries} sau lỗi: {error_msg}")
        
        # Không bao giờ nên đến đây do đã có xử lý ở trên,
        # nhưng thêm để đảm bảo code an toàn
        return f"[Error: Quá nhiều lần thử không thành công]", {
            "has_error": True,
            "error_message": "Quá nhiều lần thử không thành công",
            "error_type": "MAX_RETRIES_EXCEEDED",
            "elapsed_time": time.time() - start_time
        }
    
    def _generate_with_groq_impl(self, prompt, gen_config):
        """
        Triển khai thực tế của việc gọi Groq API.
        """
        start_time = time.time()
        api_name = "groq"
        
        try:
            # Áp dụng rate limiting với circuit breaker
            if not self._apply_rate_limiting(api_name):
                # Circuit breaker đang mở, raise exception để trigger retry
                raise Exception("Circuit breaker đang mở, đang chờ cooldown")
            
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
            timeout = gen_config.get("timeout", 30)
            
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
            
            # Đặt lại circuit breaker sau khi request thành công
            self._reset_circuit_breaker(api_name)
            
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
                "model": model_name
            }
            
            return generated_text.strip(), stats
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Lỗi khi sinh văn bản với Groq API: {str(e)}"
            logger.error(error_msg)
            
            # Log status code nếu có
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
                logger.error(f"Status code: {status_code}")
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
                logger.error(f"Status code: {status_code}")
            
            logger.debug(traceback.format_exc())
            
            # Trích xuất retry-after header nếu có
            retry_after = None
            if hasattr(e, 'headers') and e.headers and 'retry-after' in e.headers:
                try:
                    retry_after = float(e.headers['retry-after'])
                    logger.info(f"Tìm thấy header Retry-After: {retry_after}s")
                except (ValueError, TypeError):
                    pass
            elif hasattr(e, 'response') and hasattr(e.response, 'headers') and 'retry-after' in e.response.headers:
                try:
                    retry_after = float(e.response.headers['retry-after'])
                    logger.info(f"Tìm thấy header Retry-After: {retry_after}s")
                except (ValueError, TypeError):
                    pass
            
            # Xử lý lỗi 503 Service Unavailable
            if status_code == 503 or 'service unavailable' in str(e).lower():
                logger.warning(f"Groq API error: Lỗi không rõ. Đang thử lại...")
                
                # Tùy chỉnh thời gian chờ
                wait_time = self._handle_rate_limit_error(api_name, e, retry_after)
                
                # Thực hiện retry ngay tại đây thay vì lan truyền lỗi
                time.sleep(min(wait_time, 10))  # Giới hạn tối đa 10 giây
                
                # Throw lỗi để retry decorator bắt và xử lý
                raise Exception(f"RETRIABLE_ERROR: Lỗi 503 Service Unavailable. Đang thử lại.")
            
            # Xử lý lỗi 500 Internal Server Error 
            if status_code == 500 or 'internal server error' in str(e).lower():
                logger.warning(f"Groq API error: Lỗi không rõ. Đang thử lại...")
                
                # Tùy chỉnh thời gian chờ
                wait_time = self._handle_rate_limit_error(api_name, e, retry_after)
                
                # Thực hiện retry ngay tại đây thay vì lan truyền lỗi
                time.sleep(min(wait_time, 10))  # Giới hạn tối đa 10 giây
                
                # Throw lỗi để retry decorator bắt và xử lý
                raise Exception(f"RETRIABLE_ERROR: Lỗi 500 Internal Server Error. Đang thử lại.")
            
            # Customized error message based on error type
            if 'quota' in str(e).lower() or 'rate limit' in str(e).lower() or (status_code == 429):
                # Đánh dấu key hiện tại đã hết quota
                current_key = config.GROQ_API_KEYS[self.current_groq_key_index]
                
                # Xác định loại lỗi quota: hết quota theo ngày hay chỉ là rate limit tạm thời
                is_daily_quota = False
                reason = "rate_limit_exceeded"
                wait_time = 0
                
                # Phân tích lỗi để xác định đúng loại lỗi quota
                error_str = str(e).lower()
                if 'daily' in error_str and 'quota' in error_str:
                    is_daily_quota = True
                    reason = "daily_quota_exceeded"
                elif 'quota exceeded' in error_str or 'quota limit' in error_str:
                    is_daily_quota = True
                    reason = "daily_quota_exceeded"
                
                # Lưu thông tin hết hạn với timestamp hiện tại
                self.exhausted_groq_keys[current_key] = {
                    'timestamp': time.time(),
                    'reason': reason,
                    'error': str(e)
                }
                
                # Chuyển sang key tiếp theo
                old_index = self.current_groq_key_index
                self.current_groq_key_index = (self.current_groq_key_index + 1) % len(config.GROQ_API_KEYS)
                
                # Kiểm tra xem còn key khả dụng không
                if len(self.exhausted_groq_keys) >= len(config.GROQ_API_KEYS):
                    detail_msg = "Tất cả Groq API keys đều đã vượt quá quota. Vui lòng thử lại sau."
                    logger.error(detail_msg)
                    
                    # Vẫn tạm thời reset rate limiters để ngăn circuit breaker nếu đó là lỗi tạm thời
                    if not is_daily_quota:
                        wait_time = self._handle_rate_limit_error(api_name, e, retry_after)
                else:
                    # Nếu còn key khả dụng, thử lại với key mới
                    quota_type = "theo ngày" if is_daily_quota else "tạm thời"
                    detail_msg = f"Key #{old_index + 1} đã hết quota {quota_type}. Chuyển sang key #{self.current_groq_key_index + 1}/{len(config.GROQ_API_KEYS)}"
                    logger.warning(detail_msg)
                    
                    # Khởi tạo lại client với key mới
                    self._get_groq_client()
                    
                    # Giảm thời gian chờ vì chúng ta đã đổi key
                    wait_time = 1.0  # 1 second để đảm bảo không quá nhanh
                
                error_type = "QUOTA_LIMIT_ERROR"
                
                # Thực hiện sleep ngay tại đây để đảm bảo chờ đủ thời gian
                time.sleep(min(wait_time, 5))  # Giới hạn tối đa 5s khi đổi key
                
            elif 'invalid api key' in str(e).lower() or 'authentication' in str(e).lower():
                # Đánh dấu key hiện tại không hợp lệ
                current_key = config.GROQ_API_KEYS[self.current_groq_key_index]
                
                # Lưu thông tin lỗi với timestamp hiện tại
                self.exhausted_groq_keys[current_key] = {
                    'timestamp': time.time(),
                    'reason': "invalid_api_key",
                    'error': str(e)
                }
                
                # Chuyển sang key tiếp theo
                self.current_groq_key_index = (self.current_groq_key_index + 1) % len(config.GROQ_API_KEYS)
                
                # Kiểm tra xem còn key khả dụng không
                if len(self.exhausted_groq_keys) >= len(config.GROQ_API_KEYS):
                    detail_msg = "Tất cả Groq API keys đều không hợp lệ. Vui lòng kiểm tra cấu hình."
                    logger.error(detail_msg)
                    error_type = "INVALID_API_KEY_ERROR"
                else:
                    # Nếu còn key khả dụng, thử lại với key mới
                    detail_msg = f"Key #{self.current_groq_key_index} không hợp lệ. Chuyển sang key #{self.current_groq_key_index + 1}/{len(config.GROQ_API_KEYS)}"
                    logger.warning(detail_msg)
                    
                    # Khởi tạo lại client với key mới
                    self._get_groq_client()
                    
                    # Thử lại lập tức với key mới
                    error_type = "INVALID_API_KEY_ERROR"
            else:
                # Lỗi khác, xem như lỗi chung và retry
                detail_msg = f"Lỗi không rõ. Đang thử lại..."
                logger.warning(f"Groq API error: {detail_msg}")
                
                # Tạm thời đặt lại rate limiters và cập nhật circuit breaker
                wait_time = self._handle_rate_limit_error(api_name, e, retry_after)
                
                # Thực hiện sleep ngay tại đây
                time.sleep(min(wait_time, 10))  # Giới hạn tối đa 10 giây
                
                error_type = "UNKNOWN_ERROR"
            
            # Tạo thông điệp lỗi chi tiết
            detail_msg = detail_msg if 'detail_msg' in locals() else "Lỗi không rõ. Đang thử lại..."
            
            # Throw lỗi để retry decorator bắt và xử lý
            raise Exception(f"{error_type}: {detail_msg}. Original error: {str(e)}")
    
    def _apply_rate_limiting(self, api_name):
        """
        Áp dụng rate limiting cho API với cơ chế circuit breaker và khả năng thích ứng.
        
        Args:
            api_name (str): Tên API ('gemini', 'groq', etc.)
        
        Returns:
            bool: True nếu có thể gửi request, False nếu cần chờ thêm
        """
        if api_name not in _RATE_LIMITERS or api_name not in _CIRCUIT_BREAKERS:
            return True
        
        rate_limiter = _RATE_LIMITERS[api_name]
        circuit_breaker = _CIRCUIT_BREAKERS[api_name]
        
        # Kiểm tra circuit breaker trước
        with circuit_breaker["lock"]:
            current_time = time.time()
            
            # Nếu circuit breaker đang mở (đã vượt quá ngưỡng lỗi)
            if circuit_breaker["is_open"]:
                # Kiểm tra xem đã qua thời gian cooldown chưa
                elapsed_since_failure = current_time - circuit_breaker["last_failure_time"]
                
                if elapsed_since_failure > circuit_breaker["cooldown_period"]:
                    # Chuyển sang trạng thái half-open để thử
                    circuit_breaker["is_open"] = False
                    circuit_breaker["half_open_time"] = current_time
                    circuit_breaker["consecutive_successes"] = 0
                    logger.info(f"Circuit breaker cho {api_name} chuyển sang trạng thái half-open sau {elapsed_since_failure:.1f}s cooldown")
                    
                    # Khi chuyển sang half-open, giảm interval một chút để thử lại
                    if rate_limiter["adaptive_mode"]:
                        with rate_limiter["lock"]:
                            # Giảm interval đi một nửa so với lần gặp lỗi, nhưng không nhỏ hơn base_interval
                            current_interval = rate_limiter["adaptive_interval"]
                            base_interval = rate_limiter["base_interval"]
                            new_interval = max(base_interval, current_interval * 0.5)
                            rate_limiter["adaptive_interval"] = new_interval
                            logger.info(f"Giảm interval cho {api_name} từ {current_interval:.2f}s xuống {new_interval:.2f}s")
                else:
                    # Circuit breaker vẫn đang mở
                    logger.debug(f"Circuit breaker cho {api_name} vẫn đang mở. Còn {circuit_breaker['cooldown_period'] - elapsed_since_failure:.1f}s nữa để cooldown")
                    return False
            
            # Kiểm tra trạng thái half-open
            if circuit_breaker["half_open_time"] > 0:
                elapsed_since_half_open = current_time - circuit_breaker["half_open_time"]
                
                # Nếu ở trạng thái half-open quá lâu và chưa đạt đủ successes, mở lại circuit breaker
                if elapsed_since_half_open > circuit_breaker["half_open_timeout"] and circuit_breaker["consecutive_successes"] < circuit_breaker["consecutive_success_threshold"]:
                    circuit_breaker["is_open"] = True
                    circuit_breaker["last_failure_time"] = current_time
                    logger.warning(f"Quá thời gian half-open cho {api_name}, mở lại circuit breaker")
                    return False
        
        # Áp dụng rate limiting sau khi kiểm tra circuit breaker
        with rate_limiter["lock"]:
            current_time = time.time()
            elapsed = current_time - rate_limiter["last_request_time"]
            interval = rate_limiter["adaptive_interval"] if rate_limiter["adaptive_mode"] else rate_limiter["min_interval"]
            
            if elapsed < interval:
                # Cần đợi thêm
                wait_time = interval - elapsed
                if wait_time > 0.1:  # Nếu thời gian chờ > 100ms
                    logger.debug(f"Rate limiting cho {api_name}: đợi thêm {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            # Cập nhật thời gian request cuối cùng
            rate_limiter["last_request_time"] = time.time()
            
            # Trong chế độ thích ứng, nếu đã có nhiều request thành công liên tiếp, giảm dần interval
            if rate_limiter["adaptive_mode"] and rate_limiter["success_counter"] >= 5:
                # Giảm interval dần dần (nhưng không dưới mức cơ bản)
                current_interval = rate_limiter["adaptive_interval"]
                base_interval = rate_limiter["base_interval"]
                recovery_factor = rate_limiter["recovery_factor"]
                
                # Giảm 5% interval sau mỗi 5 lần thành công liên tiếp
                new_interval = max(base_interval, current_interval * recovery_factor)
                
                if new_interval < current_interval:
                    rate_limiter["adaptive_interval"] = new_interval
                    logger.debug(f"Giảm interval cho {api_name} từ {current_interval:.2f}s xuống {new_interval:.2f}s sau {rate_limiter['success_counter']} lần thành công")
                    rate_limiter["success_counter"] = 0  # Reset counter
            
            return True
    
    def _handle_rate_limit_error(self, api_name, error, retry_after=None):
        """
        Xử lý lỗi rate limit và cập nhật cấu hình rate limiting với chiến lược thông minh.
        Sử dụng học máy từ lịch sử lỗi và các header từ API để tối ưu thời gian retry.
        
        Args:
            api_name (str): Tên API ('gemini', 'groq', etc.)
            error: Lỗi được trả về
            retry_after (float, optional): Thời gian được đề xuất chờ từ API (seconds)
        
        Returns:
            float: Thời gian đề xuất chờ trước khi thử lại (seconds)
        """
        if api_name not in _RATE_LIMITERS or api_name not in _CIRCUIT_BREAKERS:
            return 10.0  # Giá trị mặc định nếu không tìm thấy cấu hình
        
        rate_limiter = _RATE_LIMITERS[api_name]
        circuit_breaker = _CIRCUIT_BREAKERS[api_name]
        current_time = time.time()
        
        # Lưu thời điểm gặp rate limit để phân tích mẫu
        rate_limiter["last_rate_limit_time"] = current_time
        
        # Tăng bộ đếm thất bại và reset bộ đếm thành công
        rate_limiter["failure_counter"] += 1
        rate_limiter["success_counter"] = 0
        
        # Cập nhật circuit breaker
        with circuit_breaker["lock"]:
            # Tăng số lỗi rate limit liên tiếp
            circuit_breaker["failures"] += 1
            circuit_breaker["last_failure_time"] = current_time
            
            # Log thông tin về lỗi
            logger.warning(f"Rate limit error #{circuit_breaker['failures']} cho {api_name}")
            
            # Kiểm tra nếu cần mở circuit breaker
            if circuit_breaker["failures"] >= circuit_breaker["failure_threshold"]:
                if not circuit_breaker["is_open"]:
                    circuit_breaker["is_open"] = True
                    logger.warning(f"Circuit breaker cho {api_name} đã mở sau {circuit_breaker['failures']} lỗi liên tiếp")
                    logger.warning(f"Tạm dừng gọi {api_name} trong {circuit_breaker['cooldown_period']}s")
            
            # Trích xuất thời gian retry từ lỗi
            extracted_retry_after = None
            error_str = str(error).lower()
            
            # Trích xuất từ nhiều định dạng thông báo lỗi khác nhau
            retry_patterns = [
                r"try again in ([0-9.]+)s",  # Groq: try again in 20s
                r"retry after ([0-9.]+)",    # Gemini: retry after 10 
                r"wait ([0-9.]+) seconds",   # Generic: wait 30 seconds
                r"available in ([0-9.]+)",   # Available in 15 seconds
                r"retry-after: ([0-9.]+)"    # Header format in string
            ]
            
            for pattern in retry_patterns:
                try:
                    import re
                    matches = re.findall(pattern, error_str)
                    if matches:
                        extracted_retry_after = float(matches[0])
                        logger.info(f"Trích xuất thời gian retry từ lỗi: {extracted_retry_after}s (pattern: {pattern})")
                        break
                except Exception as e:
                    logger.debug(f"Không thể trích xuất thời gian retry với pattern '{pattern}': {e}")
            
            # Xác định thời gian retry thực tế
            if retry_after is not None:
                actual_retry_after = retry_after
                logger.info(f"Sử dụng retry-after từ header: {actual_retry_after}s")
            elif extracted_retry_after is not None:
                actual_retry_after = extracted_retry_after
                logger.info(f"Sử dụng retry-after từ nội dung lỗi: {actual_retry_after}s")
            else:
                # Sử dụng thuật toán backoff thông minh dựa trên lịch sử
                base_wait = rate_limiter["adaptive_interval"] * 2
                jitter = random.uniform(0.8, 1.2)  # Thêm 20% jitter
                
                # Nếu đây là lỗi nghiêm trọng (nhiều lần thất bại liên tiếp), tăng thời gian chờ nhanh hơn
                severity_multiplier = min(5, circuit_breaker["failures"] ** 1.5)
                
                # Tính toán thời gian retry với mô hình backoff động
                actual_retry_after = min(60, base_wait * severity_multiplier * jitter)
                logger.info(f"Tính toán thời gian retry: {actual_retry_after:.2f}s (base: {base_wait:.2f}s, severity: {severity_multiplier:.1f}, jitter: {jitter:.2f})")
            
            # Thêm vào lịch sử retry để học
            rate_limiter["retry_delay_history"].append({
                "time": current_time,
                "delay": actual_retry_after,
                "failures": circuit_breaker["failures"]
            })
            
            # Giữ lịch sử gọn (tối đa 10 mục)
            if len(rate_limiter["retry_delay_history"]) > 10:
                rate_limiter["retry_delay_history"] = rate_limiter["retry_delay_history"][-10:]
            
            # Điều chỉnh cấu hình rate limiting dựa trên adaptive mode
            with rate_limiter["lock"]:
                if rate_limiter["adaptive_mode"]:
                    # Phân tích mẫu lỗi để điều chỉnh thông minh hơn
                    current_interval = rate_limiter["adaptive_interval"]
                    base_interval = rate_limiter["base_interval"]
                    failure_count = circuit_breaker["failures"]
                    
                    if failure_count == 1:
                        # Lỗi đầu tiên, tăng nhẹ
                        new_interval = current_interval * 1.5
                    elif failure_count == 2:
                        # Lỗi thứ hai, tăng vừa
                        new_interval = current_interval * 1.75
                    elif failure_count >= 3:
                        # Nhiều lỗi liên tiếp, tăng mạnh
                        new_interval = current_interval * 2.0
                    
                    # Phân tích mô hình thời gian để điều chỉnh thông minh
                    history = rate_limiter["retry_delay_history"]
                    if len(history) >= 3:
                        # Tính toán thời gian trung bình giữa các lỗi gần đây
                        recent_errors = sorted(history, key=lambda x: x["time"], reverse=True)[:3]
                        avg_time_between_errors = 0
                        
                        if len(recent_errors) >= 2:
                            times = [error["time"] for error in recent_errors]
                            intervals = [times[i] - times[i+1] for i in range(len(times)-1)]
                            if intervals:
                                avg_time_between_errors = sum(intervals) / len(intervals)
                        
                        # Nếu lỗi xảy ra rất nhanh sau nhau, cần tăng interval mạnh hơn
                        if 0 < avg_time_between_errors < 5:  # Lỗi trong vòng 5 giây
                            new_interval *= 2.5  # Tăng nhanh hơn
                            logger.warning(f"Lỗi xảy ra nhanh (avg {avg_time_between_errors:.1f}s), tăng mạnh interval")
                    
                    # Áp dụng giới hạn để tránh interval quá lớn
                    max_interval = 60.0  # Tối đa 60s giữa các request
                    rate_limiter["adaptive_interval"] = min(new_interval, max_interval)
                    
                    logger.info(f"Đã điều chỉnh interval cho {api_name} từ {current_interval:.2f}s lên {rate_limiter['adaptive_interval']:.2f}s (lỗi #{failure_count})")
                else:
                    # Chế độ không thích ứng, chỉ tăng dần theo cấp số nhân
                    rate_limiter["adaptive_interval"] *= 1.5
            
            # Nếu lỗi quá nhiều, hãy thêm vào actual_retry_after một khoảng thời gian ngẫu nhiên
            if circuit_breaker["failures"] > 5:
                actual_retry_after += random.uniform(0, 5)  # Thêm tối đa 5s để tránh thundering herd
            
            return actual_retry_after
    
    def _reset_circuit_breaker(self, api_name):
        """
        Đặt lại circuit breaker sau khi request thành công.
        Cập nhật các counter thành công liên tiếp và điều chỉnh adaptive interval.
        
        Args:
            api_name (str): Tên API ('gemini', 'groq', etc.)
        """
        if api_name not in _CIRCUIT_BREAKERS:
            return
        
        circuit_breaker = _CIRCUIT_BREAKERS[api_name]
        
        with circuit_breaker["lock"]:
            # Nếu đã có lỗi trước đó, đặt lại số lỗi
            if circuit_breaker["failures"] > 0:
                prev_failures = circuit_breaker["failures"]
                circuit_breaker["failures"] = 0
                circuit_breaker["is_open"] = False
                logger.info(f"Đã đặt lại circuit breaker cho {api_name} sau {prev_failures} lỗi")
            
            # Tăng số lần thành công liên tiếp trong trạng thái half-open
            if circuit_breaker["half_open_time"] > 0:
                circuit_breaker["consecutive_successes"] += 1
                logger.debug(f"Thành công liên tiếp lần {circuit_breaker['consecutive_successes']} trong half-open cho {api_name}")
                
                # Nếu đạt đủ số lần thành công liên tiếp, đóng hoàn toàn circuit breaker
                if circuit_breaker["consecutive_successes"] >= circuit_breaker["consecutive_success_threshold"]:
                    circuit_breaker["half_open_time"] = 0
                    circuit_breaker["is_open"] = False
                    circuit_breaker["failures"] = 0
                    logger.info(f"Đóng hoàn toàn circuit breaker cho {api_name} sau {circuit_breaker['consecutive_successes']} lần thành công liên tiếp")
        
        # Cập nhật rate limiter nếu có
        if api_name in _RATE_LIMITERS:
            rate_limiter = _RATE_LIMITERS[api_name]
            
            with rate_limiter["lock"]:
                # Đánh dấu thành công và đặt lại bộ đếm thất bại
                rate_limiter["success_counter"] += 1
                rate_limiter["failure_counter"] = 0
                
                # Trong chế độ thích ứng và circuit breaker đã đóng hoàn toàn
                if rate_limiter["adaptive_mode"] and circuit_breaker["half_open_time"] == 0:
                    # Điều chỉnh interval theo tình trạng mạng
                    current_time = time.time()
                    last_rate_limit_time = rate_limiter["last_rate_limit_time"]
                    
                    # Nếu đã lâu không gặp rate limit (> 10 phút), có thể giảm interval
                    if last_rate_limit_time > 0 and (current_time - last_rate_limit_time) > 600:
                        current_interval = rate_limiter["adaptive_interval"]
                        base_interval = rate_limiter["base_interval"]
                        
                        # Giảm từ từ, tiến dần về interval cơ bản
                        if current_interval > base_interval:
                            new_interval = max(base_interval, current_interval * 0.95)  # Giảm 5%
                            
                            if current_interval - new_interval > 0.01:  # Nếu sự thay đổi đáng kể
                                rate_limiter["adaptive_interval"] = new_interval
                                logger.debug(f"Giảm interval cho {api_name} về {new_interval:.2f}s (base: {base_interval:.2f}s)")
                    
                    # Lưu lại số lượng thành công liên tiếp để xem xét giảm interval sau này
                    if rate_limiter["success_counter"] >= 20:
                        logger.debug(f"Đạt {rate_limiter['success_counter']} request thành công liên tiếp cho {api_name}")
                        
                        # Có thể giảm interval hơn nữa nếu ổn định trong thời gian dài
                        current_interval = rate_limiter["adaptive_interval"]
                        base_interval = rate_limiter["base_interval"]
                        
                        if current_interval > base_interval * 1.2:  # Nếu interval vẫn cao hơn 20% so với cơ bản
                            new_interval = max(base_interval, current_interval * 0.9)  # Giảm mạnh hơn (10%)
                            rate_limiter["adaptive_interval"] = new_interval
                            logger.info(f"Giảm mạnh interval cho {api_name} từ {current_interval:.2f}s xuống {new_interval:.2f}s sau nhiều lần thành công")
                        
                        # Reset lại counter sau khi đã xử lý
                        rate_limiter["success_counter"] = 0
    
    def _load_model(self, model_name):
        """
        Tải model và tokenizer cho model được chỉ định.
        
        Args:
            model_name (str): Tên của model để tải (llama, qwen)
            
        Returns:
            tuple: (tokenizer, model) hoặc (None, None) nếu lỗi
        """
        global _MODEL_CACHE, _TOKENIZER_CACHE, _LAST_USED
        
        # Nếu cache được bật và model đã được tải vào memory
        if self.use_cache and model_name in _MODEL_CACHE and model_name in _TOKENIZER_CACHE:
            logger.debug(f"Sử dụng {model_name} từ memory cache")
            _LAST_USED[model_name] = time.time()
            return _TOKENIZER_CACHE[model_name], _MODEL_CACHE[model_name]
            
        # Dọn dẹp cache để làm không gian cho model mới
        self._clear_memory()
        self._manage_model_cache()
        
        # Kiểm tra xem đã có đường dẫn trực tiếp đến model trong .env chưa
        if model_name.lower() == "llama":
            direct_model_path = config.LLAMA_MODEL_PATH
            direct_tokenizer_path = config.LLAMA_TOKENIZER_PATH
        elif model_name.lower() == "qwen":
            direct_model_path = config.QWEN_MODEL_PATH  
            direct_tokenizer_path = config.QWEN_TOKENIZER_PATH
        else:
            direct_model_path = ""
            direct_tokenizer_path = ""
            
        has_direct_paths = direct_model_path and direct_tokenizer_path
        
        # Tải model từ disk cache nếu được bật, KHÔNG có đường dẫn trực tiếp, và model có trong cache
        if self.use_disk_cache and not has_direct_paths and model_name in _DISK_CACHE_INDEX:
            logger.info(f"Tìm thấy {model_name} trong disk cache, cố gắng tải...")
            
            try:
                result = self._load_model_from_disk(model_name)
                if result is None:
                    logger.warning(f"Không thể tải {model_name} từ disk cache: Kết quả trả về None")
                    # Tiếp tục với việc tải model từ đường dẫn trực tiếp
                else:
                    tokenizer, model = result
                    if tokenizer is not None and model is not None:
                        # Lưu vào memory cache
                        if self.use_cache:
                            _TOKENIZER_CACHE[model_name] = tokenizer
                            _MODEL_CACHE[model_name] = model
                            _LAST_USED[model_name] = time.time()
                        
                        logger.info(f"Đã tải thành công {model_name} từ disk cache")
                        return tokenizer, model
                    
            except Exception as e:
                logger.error(f"Lỗi khi tải {model_name} từ disk cache: {str(e)}")
                logger.error(traceback.format_exc())
                # Tiếp tục với việc tải model từ đường dẫn trực tiếp
        elif has_direct_paths:
            logger.info(f"Sử dụng đường dẫn trực tiếp cho {model_name} thay vì disk cache")
        
        # Thiết lập CUDA_VISIBLE_DEVICES để sử dụng cả 3 GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 3:
                gpu_ids = ",".join(str(i) for i in range(3))  # Sử dụng 3 GPU đầu tiên
                logger.info(f"Thiết lập CUDA_VISIBLE_DEVICES={gpu_ids} để sử dụng tất cả GPU cho model {model_name}")
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
        # Biến để kiểm soát fallback sang CPU nếu GPU không đủ bộ nhớ
        use_gpu = torch.cuda.is_available()
        tried_8bit = False
        tried_cpu_fallback = False
        
        # Tải model từ HuggingFace
        # Quản lý cache model để tránh OOM
        if model_name.lower() in ["llama", "qwen"]:
            if model_name.lower() == "llama":
                model_path = config.LLAMA_MODEL_PATH
                tokenizer_path = config.LLAMA_TOKENIZER_PATH
            else:  # qwen
                model_path = config.QWEN_MODEL_PATH
                tokenizer_path = config.QWEN_TOKENIZER_PATH
            
            # Kiểm tra đường dẫn model và tokenizer
            if not model_path or not tokenizer_path:
                error_msg = f"Đường dẫn cho model {model_name} không được thiết lập trong file .env"
                logger.error(error_msg)
                return None, None
                
            for attempt in range(3):  # 3 nỗ lực: 4-bit trên GPU -> 8-bit trên GPU -> CPU fallback
                try:
                    if attempt == 0:
                        # Nỗ lực đầu tiên: 4-bit trên GPU (nhẹ nhất)
                        tried_8bit = False
                        tried_cpu_fallback = False
                        use_gpu = torch.cuda.is_available()
                        logger.info(f"Nỗ lực tải model {model_name} với 4-bit quantization trên GPU")
                    elif attempt == 1:
                        # Nỗ lực thứ hai: 8-bit trên GPU (sử dụng nhiều bộ nhớ hơn)
                        tried_8bit = True
                        tried_cpu_fallback = False
                        use_gpu = torch.cuda.is_available()
                        logger.info(f"Thử lại: Tải model {model_name} với 8-bit quantization trên GPU")
                    else:
                        # Nỗ lực cuối cùng: CPU fallback (chậm nhưng đáng tin cậy nhất)
                        tried_8bit = False
                        tried_cpu_fallback = True
                        use_gpu = False
                        logger.info(f"Nỗ lực cuối cùng: Tải model {model_name} trên CPU")
                    
                    tokenizer = None
                    model = None
                    
                    # Tải tokenizer trước
                    logger.debug(f"Tải tokenizer từ {tokenizer_path}...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        use_fast=True,
                    )
                    
                    # Cấu hình quantization nếu sử dụng GPU
                    quantization_config = None
                    if use_gpu:
                        nf4_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )
                        
                        int8_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_skip_modules=None,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        
                        quantization_config = nf4_config if not tried_8bit else int8_config
                    
                    # Tạo device map & memory config
                    device_map = "auto" if use_gpu else "cpu"
                    max_memory = self._get_max_memory_config() if use_gpu else None
                    
                    # Xây dựng model kwargs
                    model_kwargs = {
                        "pretrained_model_name_or_path": model_path,
                        "device_map": device_map,
                        "torch_dtype": torch.bfloat16 if use_gpu else torch.float32,
                        "low_cpu_mem_usage": True,
                        "trust_remote_code": True,
                    }
                    
                    # Thêm cấu hình bộ nhớ nếu sử dụng GPU
                    if use_gpu and max_memory:
                        model_kwargs["max_memory"] = max_memory
                    
                    # Thêm cấu hình quantization nếu sử dụng GPU
                    if use_gpu and quantization_config:
                        model_kwargs["quantization_config"] = quantization_config
                    
                    # Thêm bất kỳ cấu hình model cụ thể nào
                    if model_name.lower() == "qwen" and "disable_attention_warnings" in config.MODEL_CONFIGS.get("qwen", {}):
                        # Cách an toàn để tắt cảnh báo attention
                        try:
                            # Tắt cảnh báo từ logging thay vì truy cập _DEFAULT_LOGGERS trực tiếp
                            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
                            logging.getLogger("transformers").setLevel(logging.ERROR)
                            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                            logger.info(f"Đã tắt cảnh báo attention cho model {model_name}")
                        except Exception as warning_error:
                            logger.warning(f"Không thể tắt cảnh báo attention: {warning_error}")
                    
                    # Tải model weights
                    logger.debug(f"Tải model weights từ {model_path} vào {'GPU' if use_gpu else 'CPU'}...")
                    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                    
                    # Sử dụng torch.compile() để tăng tốc inference nếu phiên bản Torch hỗ trợ
                    if torch.__version__ >= "2.0" and use_gpu:
                        try:
                            original_model = model
                            model = torch.compile(model)
                            logger.info(f"Đã kích hoạt torch.compile() để tăng tốc inference cho model {model_name}")
                        except Exception as e:
                            logger.warning(f"Không thể kích hoạt torch.compile(): {e}")
                            # Khôi phục model gốc nếu compile thất bại
                            model = original_model
                    
                    # Lưu vào memory cache
                    if self.use_cache:
                        _TOKENIZER_CACHE[model_name] = tokenizer
                        _MODEL_CACHE[model_name] = model
                        _LAST_USED[model_name] = time.time()
                    
                    logger.info(f"Đã tải thành công model {model_name} vào {'GPU' if use_gpu else 'CPU'}")
                    
                    # Lưu vào disk cache nếu được bật
                    if self.use_disk_cache and model_name in config.DISK_CACHE_CONFIG.get("models_to_cache", []):
                        self._save_model_to_disk(model_name, tokenizer, model)
                    
                    return tokenizer, model
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    # Xử lý lỗi OOM cụ thể
                    logger.error(f"CUDA Out of Memory khi tải model {model_name}: {str(oom_error)}")
                    logger.info("Giải phóng bộ nhớ GPU và thử lại với cấu hình nhẹ hơn")
                    self._clear_memory()  # Giải phóng bộ nhớ GPU
                    
                    # Nếu đã thử tất cả các phương pháp, ném ra ngoại lệ
                    if tried_8bit and tried_cpu_fallback:
                        raise oom_error
                        
                except Exception as e:
                    # Xử lý các lỗi khác
                    logger.error(f"Lỗi khi tải model {model_name}: {str(e)}")
                    
                    if "CUDA out of memory" in str(e):
                        logger.error("Phát hiện lỗi OOM trong ngoại lệ chung")
                        self._clear_memory()
                    else:
                        # Nếu không phải lỗi OOM, không cần thử lại
                        logger.error("Không phải lỗi OOM, không thử lại")
                        logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
                        return None, None
                    
                    # Nếu chưa thử fallback sang CPU và đây là lỗi OOM, thử lại trên CPU
                    if use_gpu and not tried_cpu_fallback and ("CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError)):
                        logger.info("Thử lại với CPU fallback")
                        tried_cpu_fallback = True
                        use_gpu = False
                        
                        # Giải phóng bộ nhớ GPU
                        self._clear_memory()
                    elif tried_8bit and not tried_cpu_fallback:
                        logger.info("Thử lại với CPU fallback")
                        tried_cpu_fallback = True
                        use_gpu = False
                    else:
                        logger.error(f"Đã thử tất cả các phương pháp, không thể tải model {model_name}")
                        return None, None
        
            # Nếu chạy đến đây mà vẫn chưa return, có nghĩa là tất cả các nỗ lực đều thất bại
            logger.error(f"Không thể tải model {model_name} sau tất cả các nỗ lực")
            return None, None
    
    def _load_model_from_disk(self, model_name):
        """
        Tải model và tokenizer từ disk cache.
        
        Args:
            model_name (str): Tên của model cần tải
            
        Returns:
            tuple: (tokenizer, model) hoặc None nếu không tìm thấy hoặc hết hạn
        """
        if not self.use_disk_cache or model_name not in _DISK_CACHE_INDEX:
            logger.warning(f"Model {model_name} không có trong disk cache index hoặc disk cache bị tắt")
            return None

        try:
            cache_info = _DISK_CACHE_INDEX[model_name]
            cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
            model_dir = cache_dir / f"{model_name}_model"
            tokenizer_dir = cache_dir / f"{model_name}_tokenizer"
            
            # Kiểm tra xem thư mục cache có tồn tại không
            if not model_dir.exists() or not tokenizer_dir.exists():
                logger.warning(f"Thư mục cache cho {model_name} không tồn tại, mặc dù có trong index")
                return None
            
            # Kiểm tra TTL (time-to-live)
            ttl = config.DISK_CACHE_CONFIG.get("ttl", 24 * 60 * 60)  # Mặc định 24 giờ
            current_time = time.time()
            if current_time - cache_info.get("timestamp", 0) > ttl:
                logger.info(f"Cache cho {model_name} đã hết hạn (TTL: {ttl}s), tải lại")
                return None
            
            # Tải model và tokenizer
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            try:
                # Tải tokenizer trước vì nhẹ hơn
                logger.info(f"Tải tokenizer từ disk cache: {tokenizer_dir}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir),
                    trust_remote_code=True,
                    use_fast=True
                )
                
                # Kiểm tra xem tokenizer có được tải thành công không
                if tokenizer is None:
                    logger.error(f"Không thể tải tokenizer cho {model_name} từ disk cache")
                    return None
                
                # Cấu hình device map
                use_gpu = torch.cuda.is_available()
                device_map = "auto" if use_gpu else "cpu"
                
                # Tải model
                logger.info(f"Tải model từ disk cache: {model_dir}")
                
                # Kiểm tra xem model có bị nén không
                is_compressed = cache_info.get("compressed", False)
                
                # Cấu hình bộ nhớ
                max_memory = self._get_max_memory_config() if use_gpu else None
                
                # Xây dựng model kwargs
                model_kwargs = {
                    "pretrained_model_name_or_path": str(model_dir),
                    "device_map": device_map,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16 if use_gpu else torch.float32,
                }
                
                # Thêm cấu hình bộ nhớ nếu sử dụng GPU
                if use_gpu and max_memory:
                    model_kwargs["max_memory"] = max_memory
                
                # Tải model
                model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                
                # Kiểm tra xem model có được tải thành công không
                if model is None:
                    logger.error(f"Không thể tải model {model_name} từ disk cache")
                    return None
                
                logger.info(f"Đã tải thành công {model_name} từ disk cache")
                
                # Cập nhật timestamp trong index
                _DISK_CACHE_INDEX[model_name]["last_used"] = current_time
                self._save_cache_index()
                
                return tokenizer, model
                
            except Exception as e:
                logger.error(f"Lỗi khi tải {model_name} từ disk cache: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Lỗi không mong đợi khi tải {model_name} từ disk cache: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_model_to_disk(self, model_name, tokenizer, model):
        """
        Lưu model và tokenizer vào disk cache.
        
        Args:
            model_name (str): Tên của model
            tokenizer: Tokenizer object
            model: Model object
            
        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        if not self.use_disk_cache:
            logger.info("Disk cache bị tắt, không lưu model")
            return False
            
        # Kiểm tra xem đã có đường dẫn trực tiếp đến model chưa
        if model_name.lower() == "llama":
            direct_path = config.LLAMA_MODEL_PATH
        elif model_name.lower() == "qwen":
            direct_path = config.QWEN_MODEL_PATH
        else:
            direct_path = ""
            
        if direct_path:
            logger.info(f"Đã có đường dẫn trực tiếp đến {model_name} trong .env, không lưu vào disk cache")
            return False
        
        try:
            cache_dir = Path(config.DISK_CACHE_CONFIG["cache_dir"])
            model_dir = cache_dir / f"{model_name}_model"
            tokenizer_dir = cache_dir / f"{model_name}_tokenizer"
            
            # Xóa thư mục cũ nếu tồn tại
            if model_dir.exists():
                logger.info(f"Xóa thư mục model cũ: {model_dir}")
                shutil.rmtree(model_dir, ignore_errors=True)
                
            if tokenizer_dir.exists():
                logger.info(f"Xóa thư mục tokenizer cũ: {tokenizer_dir}")
                shutil.rmtree(tokenizer_dir, ignore_errors=True)
            
            # Tạo thư mục mới
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu model và tokenizer
            logger.info(f"Lưu model {model_name} vào disk cache")
            
            # Lưu tokenizer
            tokenizer.save_pretrained(str(tokenizer_dir))
            
            # Lưu model
            model.save_pretrained(str(model_dir))
            
            # Cập nhật index
            _DISK_CACHE_INDEX[model_name] = {
                "timestamp": time.time(),
                "last_used": time.time(),
                "model_dir": str(model_dir),
                "tokenizer_dir": str(tokenizer_dir),
                "compressed": False
            }
            
            # Lưu index
            self._save_cache_index()
            
            logger.info(f"Đã lưu {model_name} vào disk cache thành công")
            
            # Quản lý disk cache nếu cần
            self._manage_disk_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu {model_name} vào disk cache: {str(e)}")
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
        Lấy hoặc tạo Gemini API client với key hiện tại.
        
        Returns:
            Client: Gemini API client
        """
        # Trước tiên kiểm tra và làm mới danh sách keys đã hết hạn
        self._refresh_exhausted_keys()
        
        # Xóa client cũ nếu có để cấu hình lại với key mới
        if "gemini" in _API_CLIENTS:
            del _API_CLIENTS["gemini"]
        
        # Lấy key hiện tại
        keys = config.GEMINI_API_KEYS
        if not keys:
            logger.error("Không có Gemini API key nào được cấu hình")
            return None
        
        # Nếu tất cả các key đều đã hết quota, reset danh sách key đã thử
        if len(self.exhausted_gemini_keys) >= len(keys):
            logger.warning("Tất cả Gemini API keys đều đã hết quota. Reset danh sách và thử lại.")
            self.exhausted_gemini_keys.clear()
        
        # Đảm bảo key_index nằm trong phạm vi hợp lệ
        self.current_gemini_key_index = self.current_gemini_key_index % len(keys)
        
        # Lấy key hiện tại
        current_key = keys[self.current_gemini_key_index]
        
        # Nếu key hiện tại đã hết quota, tìm key tiếp theo
        while self._is_key_exhausted(current_key, self.exhausted_gemini_keys) and len(self.exhausted_gemini_keys) < len(keys):
            self.current_gemini_key_index = (self.current_gemini_key_index + 1) % len(keys)
            current_key = keys[self.current_gemini_key_index]
        
        logger.debug(f"Sử dụng Gemini API key #{self.current_gemini_key_index + 1}/{len(keys)}")
        
        # Cấu hình Gemini API với key hiện tại
        genai.configure(api_key=current_key)
        
        # Lưu client để tái sử dụng
        _API_CLIENTS["gemini"] = genai
        
        return genai
    
    def _get_groq_client(self):
        """
        Lấy hoặc tạo Groq API client với key hiện tại.
        
        Returns:
            groq.Client: Groq client
        """
        # Trước tiên kiểm tra và làm mới danh sách keys đã hết hạn
        self._refresh_exhausted_keys()
        
        # Xóa client cũ nếu có để cấu hình lại với key mới
        if "groq" in _API_CLIENTS:
            del _API_CLIENTS["groq"]
        
        # Đảm bảo thư viện groq được cài đặt
        if 'groq' not in sys.modules:
            logger.error("Thư viện groq không được cài đặt")
            return None
        
        # Lấy key hiện tại
        keys = config.GROQ_API_KEYS
        if not keys:
            logger.error("Không có Groq API key nào được cấu hình")
            return None
        
        # Nếu tất cả các key đều đã hết quota, reset danh sách key đã thử
        if len(self.exhausted_groq_keys) >= len(keys):
            logger.warning("Tất cả Groq API keys đều đã hết quota. Reset danh sách và thử lại.")
            self.exhausted_groq_keys.clear()
        
        # Đảm bảo key_index nằm trong phạm vi hợp lệ
        self.current_groq_key_index = self.current_groq_key_index % len(keys)
        
        # Lấy key hiện tại
        current_key = keys[self.current_groq_key_index]
        
        # Nếu key hiện tại đã hết quota, tìm key tiếp theo
        while self._is_key_exhausted(current_key, self.exhausted_groq_keys) and len(self.exhausted_groq_keys) < len(keys):
            self.current_groq_key_index = (self.current_groq_key_index + 1) % len(keys)
            current_key = keys[self.current_groq_key_index]
        
        logger.debug(f"Sử dụng Groq API key #{self.current_groq_key_index + 1}/{len(keys)}")
        
        # Tạo client mới với key hiện tại
        client = groq.Client(api_key=current_key)
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
                if num_gpus == 3:
                    # Tối ưu cụ thể cho 3 GPU RTX 6000 Ada 48GB
                    # Phân bổ bộ nhớ cân đối hơn để tận dụng toàn bộ 3 GPU cho một mô hình
                    for i in range(num_gpus):
                        reserved_memory = config.SYSTEM_RESERVE_MEMORY_GB
                        
                        if i == 0:
                            # GPU đầu tiên: để lại nhiều bộ nhớ hơn cho hệ thống
                            reserved_memory += 1.0
                            usable_memory = max(1, int(gpu_info[i]['total_memory'] - reserved_memory))
                            # Phân bổ 32% cho GPU đầu tiên - chứa lớp embedding và đầu ra
                            max_memory[i] = f"{usable_memory}GiB"
                        else:
                            # GPU còn lại chia đều phần còn lại
                            usable_memory = max(1, int(gpu_info[i]['total_memory'] - reserved_memory))
                            max_memory[i] = f"{usable_memory}GiB"
                            
                    logger.info(f"Cấu hình phân bổ bộ nhớ cho 3 GPU: {max_memory}")
                
                elif num_gpus >= 4:
                    # Đối với hệ thống nhiều GPU (4+), phân bổ nhiều bộ nhớ hơn cho GPU đầu tiên
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
                if num_gpus == 3:
                    # Cho trường hợp 3 GPU, giảm offload CPU vì đã có đủ VRAM
                    cpu_offload = max(8, int(config.CPU_OFFLOAD_GB * 0.5))
                elif num_gpus >= 4:
                    cpu_offload = max(8, int(config.CPU_OFFLOAD_GB * 0.6))  # Giảm offload với 4+ GPU
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

    def get_response(self, model_name, prompt, max_tokens=None):
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
        
        # Import lại config để đảm bảo dùng phiên bản mới nhất
        import config as app_config
        
        # Nếu không có max_tokens được chỉ định, sử dụng giá trị mặc định từ config
        if max_tokens is None:
            # Mặc định là lấy config model cơ bản
            max_tokens = app_config.MODEL_CONFIGS[actual_model_name].get("max_tokens", 1024)
        
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
    def _is_key_exhausted(self, key, exhausted_keys):
        """
        Kiểm tra xem key có đang bị hết hạn không, có tính đến việc sang ngày mới.
        
        Args:
            key (str): API key cần kiểm tra
            exhausted_keys (dict): Dictionary lưu thông tin keys đã hết hạn
            
        Returns:
            bool: True nếu key vẫn đang hết hạn, False nếu có thể sử dụng
        """
        if key not in exhausted_keys:
            return False
            
        # Lấy thông tin hết hạn
        exhaustion_info = exhausted_keys[key]
        exhaustion_time = exhaustion_info.get('timestamp', 0)
        reason = exhaustion_info.get('reason', 'unknown')
        
        # Nếu lý do hết hạn không phải quota theo ngày, luôn coi là exhausted
        if reason != 'daily_quota_exceeded':
            return True
            
        # Kiểm tra xem key có reset theo ngày không
        current_time = time.time()
        
        # Tính toán thời điểm reset quota (giả sử reset lúc 00:00 UTC)
        exhaustion_date = datetime.datetime.fromtimestamp(exhaustion_time).date()
        current_date = datetime.datetime.fromtimestamp(current_time).date()
        
        # Nếu đã sang ngày mới, key có thể sử dụng lại
        if current_date > exhaustion_date:
            logger.info(f"Key {key[:10]}... đã sang ngày mới, có thể sử dụng lại")
            return False
            
        return True
        
    def _refresh_exhausted_keys(self):
        """
        Kiểm tra và làm mới danh sách keys đã hết hạn, loại bỏ các keys đã có thể sử dụng lại.
        Gọi định kỳ hoặc khi tất cả keys đều đã hết hạn.
        """
        current_time = time.time()
        
        # Kiểm tra xem có cần refresh không, chỉ refresh định kỳ theo interval
        if current_time - self.last_key_refresh_time < self.key_refresh_interval:
            return
            
        self.last_key_refresh_time = current_time
        logger.info("Đang làm mới danh sách API keys hết hạn...")
        
        # Kiểm tra keys Gemini
        refreshed_keys = []
        for key in list(self.exhausted_gemini_keys.keys()):
            if not self._is_key_exhausted(key, self.exhausted_gemini_keys):
                del self.exhausted_gemini_keys[key]
                refreshed_keys.append(key[:10] + "...")
                
        if refreshed_keys:
            logger.info(f"Đã làm mới {len(refreshed_keys)} Gemini API keys: {', '.join(refreshed_keys)}")
        
        # Kiểm tra keys Groq
        refreshed_keys = []
        for key in list(self.exhausted_groq_keys.keys()):
            if not self._is_key_exhausted(key, self.exhausted_groq_keys):
                del self.exhausted_groq_keys[key]
                refreshed_keys.append(key[:10] + "...")
                
        if refreshed_keys:
            logger.info(f"Đã làm mới {len(refreshed_keys)} Groq API keys: {', '.join(refreshed_keys)}")

    def batch_generate_text(self, model_name, prompts, config=None):
        """
        Sinh văn bản cho nhiều prompt cùng lúc, tận dụng khả năng xử lý batch của model.
        
        Args:
            model_name (str): Tên của model (llama, qwen, gemini, groq)
            prompts (list): Danh sách các prompt đầu vào
            config (dict): Cấu hình generation (temperature, max_tokens, etc.)
            
        Returns:
            list: Danh sách các văn bản được sinh
        """
        logger.info(f"Đang xử lý batch với {len(prompts)} prompt trên model {model_name}")
        
        # Chuẩn bị cấu hình mặc định từ config module
        import config as app_config
        model_config = app_config.MODEL_CONFIGS.get(model_name, {}).copy()
        
        # Ghi đè cấu hình nếu được cung cấp
        if config:
            model_config.update(config)
            
        # Xác định phương thức sinh phù hợp theo loại model
        responses = []
        
        if model_name.lower() in ["llama", "qwen"]:
            # Để tận dụng GPU cache, chúng ta xử lý cùng lúc nhiều prompt
            try:
                # Tải model nếu chưa có trong cache
                tokenizer, model = self._load_model(model_name)
                
                if tokenizer is None or model is None:
                    error_msg = f"Không thể tải model {model_name}"
                    logger.error(error_msg)
                    return [f"[Error: {error_msg}]"] * len(prompts)
                
                # Xử lý từng prompt nhưng tận dụng model đã load vào GPU
                for prompt in prompts:
                    response, _ = self._generate_with_local_model(model_name, prompt, model_config)
                    responses.append(response)
                
                return responses
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý batch với model {model_name}: {str(e)}")
                logger.debug(traceback.format_exc())
                return [f"[Error: {str(e)}]"] * len(prompts)
            
        elif model_name.lower() in ["gemini", "groq"]:
            # API models không xử lý được batch thực sự, nên xử lý tuần tự
            for prompt in prompts:
                if model_name.lower() == "gemini":
                    response, _ = self._generate_with_gemini(prompt, model_config)
                else:
                    response, _ = self._generate_with_groq(prompt, model_config)
                responses.append(response)
                
            return responses
            
        else:
            error_msg = f"Model không được hỗ trợ: {model_name}"
            logger.error(error_msg)
            return [f"[Error: {error_msg}]"] * len(prompts)

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
    import config as app_config
    models = {}
    
    # Model local
    if app_config.LLAMA_MODEL_PATH:
        models["llama"] = {
            "type": "local",
            "cached": "llama_model" in _MODEL_CACHE,
            "disk_cached": "llama" in _DISK_CACHE_INDEX
        }
    
    if app_config.QWEN_MODEL_PATH:
        models["qwen"] = {
            "type": "local",
            "cached": "qwen_model" in _MODEL_CACHE,
            "disk_cached": "qwen" in _DISK_CACHE_INDEX
        }
    
    # Model API
    if app_config.GEMINI_API_KEYS:
        models["gemini"] = {
            "type": "api",
            "api": "gemini",
            "keys_count": len(app_config.GEMINI_API_KEYS)
        }
    
    if app_config.GROQ_API_KEYS:
        models["groq"] = {
            "type": "api",
            "api": "groq",
            "keys_count": len(app_config.GROQ_API_KEYS)
        }
    
    return models

