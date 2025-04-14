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
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_QUESTIONS = None  # None = all questions
DEFAULT_CHECKPOINT_FREQUENCY = 5  # Save checkpoint after every X questions
MAX_CHECKPOINTS = 5  # Maximum number of checkpoints to keep
# Cấu hình model mặc định
MODEL_CONFIGS = {
    "llama": {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    },
    "qwen": {
        "max_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "disable_attention_warnings": True,
    },
    "gemini": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    },
    "groq": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "model": "llama3-70b-8192"  # Default model for Groq
    }
}

# Thông tin về API management
API_CONFIGS = {
    "gemini": {
        "requests_per_minute": 30,
        "max_retries": 5,
        "retry_base_delay": 2,  # seconds
        "max_retry_delay": 60,  # seconds
        "timeout": 30,  # seconds
        "models": {
            "reasoning_evaluation": "gemini-1.5-pro",  # Model để đánh giá khả năng suy luận
            "general": "gemini-1.5-flash"  # Model mặc định cho general usage
        }
    },
    "groq": {
        "requests_per_minute": 30,
        "max_retries": 5,
        "retry_base_delay": 2,  # seconds
        "max_retry_delay": 60,  # seconds
        "timeout": 30,  # seconds
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

def validate_config():
    """Kiểm tra tính hợp lệ của cấu hình và hiển thị cảnh báo cho các giá trị bị thiếu."""
    warnings = []
    
    # Kiểm tra API keys
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY không được thiết lập trong .env")
    
    if not GROQ_API_KEY and REASONING_EVALUATION_CONFIG["enabled"] and REASONING_EVALUATION_CONFIG["use_groq"]:
        warnings.append("GROQ_API_KEY không được thiết lập trong .env, nhưng cấu hình để sử dụng Groq cho đánh giá suy luận")
    
    # Kiểm tra model paths
    if "llama" in DEFAULT_MODELS:
        if not LLAMA_MODEL_PATH or not LLAMA_TOKENIZER_PATH:
            warnings.append("Model path cho Llama không được thiết lập trong .env, nhưng 'llama' có trong DEFAULT_MODELS")
    
    if "qwen" in DEFAULT_MODELS:
        if not QWEN_MODEL_PATH or not QWEN_TOKENIZER_PATH:
            warnings.append("Model path cho Qwen không được thiết lập trong .env, nhưng 'qwen' có trong DEFAULT_MODELS")
    
    # Kiểm tra file questions
    if not Path(QUESTIONS_FILE).exists():
        warnings.append(f"File câu hỏi không tồn tại: {QUESTIONS_FILE}")
    
    # Kiểm tra thư mục cache
    if ENABLE_DISK_CACHE and not MODEL_CACHE_DIR.exists():
        warnings.append(f"Thư mục cache không tồn tại: {MODEL_CACHE_DIR}")
    
    # Hiển thị cảnh báo
    if warnings:
        logger.warning("Vấn đề với cấu hình:")
        for warning in warnings:
            logger.warning(f"- {warning}")
        return False
    
    logger.info("Cấu hình hợp lệ.")
    return True

def display_config_summary():
    """Hiển thị tóm tắt cấu hình hiện tại."""
    logger.info("=== Cấu hình hệ thống ===")
    logger.info(f"Models: {DEFAULT_MODELS}")
    logger.info(f"Prompts: {DEFAULT_PROMPTS}")
    logger.info(f"Questions file: {QUESTIONS_FILE}")
    logger.info(f"Batch size: {DEFAULT_BATCH_SIZE}")
    logger.info(f"Checkpoint frequency: {DEFAULT_CHECKPOINT_FREQUENCY}")
    
    # Hiển thị thông tin API
    logger.info("=== Thông tin API ===")
    logger.info(f"Gemini API: {'Đã cấu hình' if GEMINI_API_KEY else 'Chưa cấu hình'}")
    logger.info(f"Groq API: {'Đã cấu hình' if GROQ_API_KEY else 'Chưa cấu hình'}")
    logger.info(f"OpenAI API: {'Đã cấu hình' if OPENAI_API_KEY else 'Chưa cấu hình'}")
    
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
