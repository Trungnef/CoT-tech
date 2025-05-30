"""
Quản lý cấu hình và cài đặt cho quá trình đánh giá LLM.
Tải cấu hình từ .env và cung cấp giá trị mặc định.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path
import torch

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình logging cơ bản cho module config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("config")

# Paths cơ bản
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
QUESTIONS_FILE = DATA_DIR / "questions" / "problems.json"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = RESULTS_DIR / "plots"
RAW_RESULTS_DIR = RESULTS_DIR / "raw_results"

# Cache paths
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ROOT_DIR / "cache"))
ENABLE_DISK_CACHE = os.getenv("ENABLE_DISK_CACHE", "true").lower() == "true"
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "2"))

# Đảm bảo các thư mục tồn tại
for dir_path in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, REPORTS_DIR, PLOTS_DIR, RAW_RESULTS_DIR, MODEL_CACHE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# API Keys (danh sách)
GEMINI_API_KEYS = [key.strip() for key in os.getenv("GEMINI_API_KEYS", GEMINI_API_KEY).split(",") if key.strip()]
OPENAI_API_KEYS = [key.strip() for key in os.getenv("OPENAI_API_KEYS", OPENAI_API_KEY).split(",") if key.strip()]
GROQ_API_KEYS = [key.strip() for key in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if key.strip()]

# Đảm bảo có ít nhất một key trong mỗi danh sách nếu biến đơn lẻ được đặt
if GEMINI_API_KEY and not GEMINI_API_KEYS:
    GEMINI_API_KEYS = [GEMINI_API_KEY]
if OPENAI_API_KEY and not OPENAI_API_KEYS:
    OPENAI_API_KEYS = [OPENAI_API_KEY]
if GROQ_API_KEY and not GROQ_API_KEYS:
    GROQ_API_KEYS = [GROQ_API_KEY]

# Model Paths
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "")
LLAMA_TOKENIZER_PATH = os.getenv("LLAMA_TOKENIZER_PATH", "")
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "")
QWEN_TOKENIZER_PATH = os.getenv("QWEN_TOKENIZER_PATH", "")

# GPU Configuration
MAX_GPU_MEMORY_GB = float(os.getenv("MAX_GPU_MEMORY_GB", 140))
SYSTEM_RESERVE_MEMORY_GB = float(os.getenv("SYSTEM_RESERVE_MEMORY_GB", 2.5))
CPU_OFFLOAD_GB = float(os.getenv("CPU_OFFLOAD_GB", 24))

# Danh sách models và prompts mặc định
DEFAULT_MODELS = ["llama", "qwen", "gemini"]
DEFAULT_PROMPTS = [
    "zero_shot", 
    "few_shot_3", "few_shot_5", "few_shot_7", 
    "cot", 
    "cot_self_consistency_3", "cot_self_consistency_5", "cot_self_consistency_7", 
    "react"
]

# Cấu hình đánh giá mặc định
DEFAULT_BATCH_SIZE = 20  # Kích thước batch tối ưu cho 3 GPU: xử lý 15 câu hỏi trong 1 lần chạy
DEFAULT_MAX_QUESTIONS = None  # None = all questions
DEFAULT_CHECKPOINT_FREQUENCY = 5  # Save checkpoint after every X questions
MAX_CHECKPOINTS = 5  # Maximum number of checkpoints to keep

# Cấu hình max_tokens dựa trên loại prompt
# Cung cấp max_tokens cao hơn cho các prompts đòi hỏi suy luận nhiều
PROMPT_TOKEN_CONFIGS = {
    # Giá trị mặc định cho mỗi model
    "default": {
        "llama": 512,
        "qwen": 384,
        "gemini": 1024,
        "groq": 1024,
        "openai": 1024
    },
    # Prompt đơn giản
    "zero_shot": {
        "llama": 512,
        "qwen": 384,
        "gemini": 1024,
        "groq": 1024,
        "openai": 1024
    },
    # Few-shot prompts
    "few_shot_3": {
        "llama": 768,
        "qwen": 512,
        "gemini": 1536,
        "groq": 1536,
        "openai": 1536
    },
    "few_shot_5": {
        "llama": 1024,
        "qwen": 768,
        "gemini": 1792,
        "groq": 1792,
        "openai": 1792
    },
    "few_shot_7": {
        "llama": 1280,
        "qwen": 1024,
        "gemini": 2048,
        "groq": 2048,
        "openai": 2048
    },
    # Chain-of-Thought prompts
    "cot": {
        "llama": 1024,
        "qwen": 768,
        "gemini": 2048,
        "groq": 2048,
        "openai": 2048
    },
    # CoT with self-consistency
    "cot_self_consistency_3": {
        "llama": 1280,
        "qwen": 1024,
        "gemini": 2560,
        "groq": 2560,
        "openai": 2560
    },
    "cot_self_consistency_5": {
        "llama": 1536,
        "qwen": 1280,
        "gemini": 3072,
        "groq": 3072,
        "openai": 3072
    },
    "cot_self_consistency_7": {
        "llama": 1792,
        "qwen": 1536,
        "gemini": 3584,
        "groq": 3584,
        "openai": 3584
    },
    # ReAct prompts (cần nhiều tokens nhất do suy luận phức tạp)
    "react": {
        "llama": 2048,
        "qwen": 1792,
        "gemini": 4096,
        "groq": 4096,
        "openai": 4096
    }
}

# Cấu hình model mặc định
MODEL_CONFIGS = {
    "llama": {
        "max_tokens": 512,  # Giá trị mặc định, sẽ được ghi đè bởi cấu hình prompt cụ thể
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    },
    "qwen": {
        "max_tokens": 384,  # Giá trị mặc định, sẽ được ghi đè bởi cấu hình prompt cụ thể
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "disable_attention_warnings": True,
    },
    "gemini": {
        "max_tokens": 1024,  # Giá trị mặc định, sẽ được ghi đè bởi cấu hình prompt cụ thể
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    },
    "groq": {
        "max_tokens": 1024,  # Giá trị mặc định, sẽ được ghi đè bởi cấu hình prompt cụ thể
        "temperature": 0.7,
        "top_p": 0.95,
        "model": "llama3-70b-8192"  # Default model for Groq
    }
}

# Cấu hình model embedding cho semantic similarity
EMBEDDING_MODELS = {
    "english": {
        "default": "all-MiniLM-L6-v2",  # Model mặc định cho tiếng Anh
        "options": [
            "all-MiniLM-L6-v2",  # Nhỏ và nhanh
            "all-mpnet-base-v2",  # Chất lượng cao hơn
            "all-distilroberta-v1"  # Cân bằng giữa kích thước và chất lượng
        ]
    },
    "vietnamese": {
        "default": "bkai-foundation-models/vietnamese-bi-encoder",  # Model mặc định cho tiếng Việt
        "options": [
            "bkai-foundation-models/vietnamese-bi-encoder",
            "imthanhlv/vietnamese-sentence-embedding"
        ]
    },
    "multilingual": {
        "default": "paraphrase-multilingual-MiniLM-L12-v2",  # Model đa ngôn ngữ mặc định
        "options": [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v1"
        ]
    }
}

# Thông tin về API management
API_CONFIGS = {
    "gemini": {
        "requests_per_minute": 30,  # RPM mặc định
        "max_retries": 7,  # Tăng từ 5 lên 7
        "retry_base_delay": 1.5,  # Giảm từ 2s xuống 1.5s
        "max_retry_delay": 45,  # Giảm từ 60s xuống 45s
        "jitter_factor": 0.25,  # Thêm jitter để tránh thundering herd
        "timeout": 30,  # seconds
        "error_codes_to_retry": [429, 500, 502, 503, 504],  # Chỉ định cụ thể các mã lỗi cần retry
        "adaptive_rate_limiting": True,  # Kích hoạt giới hạn tốc độ thích ứng
        "circuit_breaker": {
            "failure_threshold": 5,  # Số lần lỗi rate limit liên tiếp trước khi mở circuit breaker
            "cooldown_period": 60,  # Thời gian cooldown (giây)
            "half_open_timeout": 30,  # Thời gian thử lại sau khi cooldown (giây)
            "consecutive_success_threshold": 2  # Số lần thành công liên tiếp để đóng circuit breaker
        },
        "models": {
            "reasoning_evaluation": "gemini-1.5-pro",  # Model để đánh giá khả năng suy luận
            "general": "gemini-1.5-flash"  # Model mặc định cho general usage
        }
    },
    "groq": {
        "requests_per_minute": 20,  # Giảm RPM để tránh rate limit
        "max_retries": 10,  # Tăng từ 8 lên 10
        "retry_base_delay": 2,  # Giảm từ 4s xuống 2s
        "max_retry_delay": 90,  # Giảm từ 120s xuống 90s
        "jitter_factor": 0.3,  # Thêm jitter lớn hơn cho Groq vì API thường quá tải
        "timeout": 45,  # seconds
        "error_codes_to_retry": [429, 500, 502, 503, 504],  # Chỉ định cụ thể các mã lỗi cần retry
        "adaptive_rate_limiting": True,  # Kích hoạt giới hạn tốc độ thích ứng
        "circuit_breaker": {
            "failure_threshold": 3,  # Số lần lỗi rate limit liên tiếp trước khi mở circuit breaker
            "cooldown_period": 90,  # Thời gian cooldown dài hơn cho Groq (giây)
            "half_open_timeout": 45,  # Thời gian thử lại sau khi cooldown (giây)
            "consecutive_success_threshold": 2  # Số lần thành công liên tiếp để đóng circuit breaker
        },
        "models": {
            "reasoning_evaluation": "llama3-70b-8192",  # Model để đánh giá khả năng suy luận
            "general": "llama3-8b-8192"  # Model mặc định cho general usage
        }
    }
}

# Cấu hình đánh giá reasoning
REASONING_EVALUATION_CONFIG = {
    "enabled": True,
    "sample_size": 50,  # Số lượng mẫu để đánh giá suy luận
    "metrics": ["coherence", "relevance", "logical_structure", "factual_accuracy", "overall"],
    "use_groq": True,  # Sử dụng Groq API để đánh giá
    "model": "groq/llama3-70b-8192",  # Model để đánh giá suy luận
    "criteria_weights": {  # Trọng số cho các tiêu chí đánh giá
        "logical_flow": 0.25,
        "mathematical_correctness": 0.25,
        "clarity": 0.20,
        "completeness": 0.15,
        "relevance": 0.15
    }
}

# Cấu hình disk cache cho model
DISK_CACHE_CONFIG = {
    "enabled": ENABLE_DISK_CACHE,
    "cache_dir": MODEL_CACHE_DIR,
    "max_cached_models": MAX_CACHED_MODELS,
    "ttl": 24 * 60 * 60,  # Time-to-live: 24 giờ
    "models_to_cache": ["llama", "qwen"],  # Chỉ cache model local
    "compression": True,  # Nén cache để tiết kiệm dung lượng ổ đĩa
    "cleanup_on_startup": False  # Có xóa cache cũ khi khởi động hay không
}

def get_max_tokens(model_name, prompt_type):
    """
    Lấy giá trị max_tokens phù hợp dựa trên loại model và prompt.
    
    Args:
        model_name (str): Tên model (llama, qwen, gemini, groq)
        prompt_type (str): Loại prompt (zero_shot, few_shot_3, cot, react, etc.)
    
    Returns:
        int: Giá trị max_tokens phù hợp
    """
    # Nếu không tìm thấy cấu hình cho prompt_type, sử dụng default
    if prompt_type not in PROMPT_TOKEN_CONFIGS:
        prompt_type = "default"
    
    # Nếu không tìm thấy cấu hình cho model_name, sử dụng default
    if model_name not in PROMPT_TOKEN_CONFIGS[prompt_type]:
        # Trả về giá trị mặc định từ MODEL_CONFIGS nếu có
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name]["max_tokens"]
        # Hoặc giá trị mặc định an toàn
        return 1024
    
    return PROMPT_TOKEN_CONFIGS[prompt_type][model_name]

def validate_config():
    """Kiểm tra tính hợp lệ của cấu hình và hiển thị cảnh báo cho các giá trị bị thiếu."""
    warnings = []
    errors = []
    
    # Kiểm tra API keys
    if not GEMINI_API_KEYS:
        warnings.append("GEMINI_API_KEYS không được thiết lập trong .env")
    
    if not GROQ_API_KEYS and REASONING_EVALUATION_CONFIG["enabled"] and REASONING_EVALUATION_CONFIG["use_groq"]:
        warnings.append("GROQ_API_KEYS không được thiết lập trong .env, nhưng cấu hình để sử dụng Groq cho đánh giá suy luận")
    
    # Kiểm tra model paths cho các model được chọn
    for model_name in DEFAULT_MODELS:
        if model_name.lower() == "llama":
            if not LLAMA_MODEL_PATH or not LLAMA_TOKENIZER_PATH:
                warnings.append(f"Model path cho Llama không được thiết lập trong .env, nhưng '{model_name}' có trong DEFAULT_MODELS")
        elif model_name.lower() == "qwen":
            if not QWEN_MODEL_PATH or not QWEN_TOKENIZER_PATH:
                warnings.append(f"Model path cho Qwen không được thiết lập trong .env, nhưng '{model_name}' có trong DEFAULT_MODELS")
        elif model_name.lower() == "gemini":
            if not GEMINI_API_KEYS:
                errors.append(f"GEMINI_API_KEYS không được thiết lập, nhưng '{model_name}' có trong DEFAULT_MODELS")
        elif model_name.lower() == "groq":
            if not GROQ_API_KEYS:
                errors.append(f"GROQ_API_KEYS không được thiết lập, nhưng '{model_name}' có trong DEFAULT_MODELS")
    
    # Kiểm tra file questions
    if not Path(QUESTIONS_FILE).exists():
        errors.append(f"File câu hỏi không tồn tại: {QUESTIONS_FILE}")
    else:
        # Kiểm tra định dạng file questions
        try:
            import json
            with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            if not isinstance(questions_data, dict) or 'questions' not in questions_data:
                errors.append(f"File câu hỏi {QUESTIONS_FILE} không có định dạng hợp lệ. Cần có field 'questions'")
            elif not isinstance(questions_data['questions'], list) or len(questions_data['questions']) == 0:
                errors.append(f"Không tìm thấy câu hỏi nào trong file {QUESTIONS_FILE}")
        except json.JSONDecodeError:
            errors.append(f"File câu hỏi {QUESTIONS_FILE} không phải là file JSON hợp lệ")
        except Exception as e:
            errors.append(f"Lỗi khi đọc file câu hỏi {QUESTIONS_FILE}: {str(e)}")
    
    # Kiểm tra thư mục cache
    if ENABLE_DISK_CACHE:
        if not MODEL_CACHE_DIR.exists():
            try:
                MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Đã tạo thư mục cache: {MODEL_CACHE_DIR}")
            except Exception as e:
                errors.append(f"Không thể tạo thư mục cache {MODEL_CACHE_DIR}: {str(e)}")
    
    # Kiểm tra cấu hình model
    for model_name, model_config in MODEL_CONFIGS.items():
        if model_name in DEFAULT_MODELS:
            required_params = ["max_tokens", "temperature"]
            for param in required_params:
                if param not in model_config:
                    errors.append(f"Thiếu tham số bắt buộc '{param}' trong cấu hình cho model {model_name}")
    
    # Kiểm tra các thư mục kết quả
    for dir_path in [RESULTS_DIR, CHECKPOINTS_DIR, REPORTS_DIR, PLOTS_DIR, RAW_RESULTS_DIR]:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Đã tạo thư mục: {dir_path}")
            except Exception as e:
                errors.append(f"Không thể tạo thư mục {dir_path}: {str(e)}")
    
    # Kiểm tra GPU nếu sử dụng model local
    if "llama" in DEFAULT_MODELS or "qwen" in DEFAULT_MODELS:
        if not torch.cuda.is_available():
            warnings.append("CUDA không khả dụng, nhưng có model local trong DEFAULT_MODELS. Model sẽ chạy trên CPU, có thể rất chậm.")
        else:
            try:
                # Kiểm tra GPU memory
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    if props.total_memory / 1024**3 < 8:  # Ít hơn 8GB
                        warnings.append(f"GPU {i} ({props.name}) chỉ có {props.total_memory / 1024**3:.1f} GB VRAM, có thể không đủ cho một số model lớn.")
            except Exception as e:
                warnings.append(f"Không thể kiểm tra thông tin GPU: {str(e)}")
    
    # Hiển thị lỗi và cảnh báo
    if errors:
        logger.error("=== LỖI NGHIÊM TRỌNG TRONG CẤU HÌNH ===")
        for error in errors:
            logger.error(f"- {error}")
        logger.error("Các lỗi trên cần được xử lý trước khi tiếp tục!")
        return False
    
    if warnings:
        logger.warning("=== CẢNH BÁO TRONG CẤU HÌNH ===")
        for warning in warnings:
            logger.warning(f"- {warning}")
    
    logger.info("Cấu hình cơ bản hợp lệ.")
    return len(errors) == 0

def display_config_summary():
    """Hiển thị tóm tắt cấu hình hiện tại."""
    logger.info("=== Cấu hình hệ thống ===")
    logger.info(f"Models: {DEFAULT_MODELS}")
    logger.info(f"Prompts: {DEFAULT_PROMPTS}")
    logger.info(f"Questions file: {QUESTIONS_FILE}")
    logger.info(f"Batch size: {DEFAULT_BATCH_SIZE}")
    logger.info(f"Checkpoint frequency: {DEFAULT_CHECKPOINT_FREQUENCY}")
    
    # Hiển thị thông tin max_tokens cho các prompt
    logger.info("=== Cấu hình max_tokens theo prompt ===")
    for prompt_type in DEFAULT_PROMPTS:
        logger.info(f"Prompt '{prompt_type}':")
        for model_name in DEFAULT_MODELS:
            max_tokens = get_max_tokens(model_name, prompt_type)
            logger.info(f"  - {model_name}: {max_tokens} tokens")
    
    # Hiển thị thông tin API
    logger.info("=== Thông tin API ===")
    logger.info(f"Gemini API: {len(GEMINI_API_KEYS)} keys")
    logger.info(f"Groq API: {len(GROQ_API_KEYS)} keys")
    logger.info(f"OpenAI API: {len(OPENAI_API_KEYS)} keys")
    
    # Hiển thị thông tin model local
    logger.info("=== Thông tin model local ===")
    logger.info(f"Llama: {'Đã cấu hình' if LLAMA_MODEL_PATH else 'Chưa cấu hình'}")
    logger.info(f"Qwen: {'Đã cấu hình' if QWEN_MODEL_PATH else 'Chưa cấu hình'}")
    
    # Hiển thị cấu hình đánh giá suy luận
    logger.info("=== Đánh giá suy luận ===")
    logger.info(f"Enabled: {REASONING_EVALUATION_CONFIG['enabled']}")
    if REASONING_EVALUATION_CONFIG['enabled']:
        logger.info(f"Metrics: {REASONING_EVALUATION_CONFIG['metrics']}")
        logger.info(f"Sử dụng Groq: {REASONING_EVALUATION_CONFIG['use_groq']}")
        
    # Hiển thị cấu hình cache
    logger.info("=== Cấu hình cache ===")
    logger.info(f"Disk cache: {'Bật' if ENABLE_DISK_CACHE else 'Tắt'}")
    logger.info(f"Thư mục cache: {MODEL_CACHE_DIR}")
    logger.info(f"Số lượng model lưu cache tối đa: {MAX_CACHED_MODELS}")
    
    # Hiển thị thông tin GPU
    if torch.cuda.is_available():
        logger.info("=== Thông tin GPU ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
