"""
Centralized logging setup for the LLM evaluation framework.
This module provides a logger that can be imported and used across the application.
"""

import os
import sys
import logging
import logging.handlers
import datetime
import threading
from pathlib import Path
from typing import Dict, Optional, Union, List, Any

# Định nghĩa màu sắc cho các log level khác nhau (cho console)
LOGGING_COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[41m',  # Red background
    'RESET': '\033[0m'       # Reset color
}

# Định nghĩa các custom log levels
METRICS = 15  # Between DEBUG and INFO
logging.addLevelName(METRICS, "METRICS")

# Global logger instance
logger = None
_is_initializing = False  # Flag to track initialization state
lock = threading.RLock()  # Add a lock for thread safety

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to console logs.
    """
    
    def format(self, record):
        if hasattr(record, 'color'):
            # Already processed this record
            return super().format(record)
            
        levelname = record.levelname
        message = super().format(record)
        
        if levelname in LOGGING_COLORS:
            colored_levelname = f"{LOGGING_COLORS[levelname]}{levelname}{LOGGING_COLORS['RESET']}"
            # Replace the levelname with colored version
            message = message.replace(levelname, colored_levelname, 1)
            record.color = True  # Mark as processed
        
        return message

class ThreadSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Thread-safe implementation of RotatingFileHandler"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()
    
    def emit(self, record):
        with self.lock:
            super().emit(record)
    
    def doRollover(self):
        with self.lock:
            super().doRollover()
    
    def shouldRollover(self, record):
        with self.lock:
            return super().shouldRollover(record)

def metrics(self, message, *args, **kws):
    """
    Log metrics information (between DEBUG and INFO levels).
    """
    if self.isEnabledFor(METRICS):
        self._log(METRICS, message, args, **kws)

# Add the metrics method to the Logger class
logging.Logger.metrics = metrics

def setup_logging(
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    module_levels: Optional[Dict[str, int]] = None,
    log_format: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    max_file_size: int = 10485760,  # 10MB
    backup_count: int = 10,
    colored_console: bool = True
) -> logging.Logger:
    """
    Thiết lập logging tập trung với cấu hình có thể điều chỉnh.
    
    Args:
        log_dir: Thư mục lưu file log
        log_file: Tên file log (nếu None, tự động tạo tên với timestamp)
        console_level: Level log cho console
        file_level: Level log cho file
        module_levels: Dict chứa level riêng cho từng module
        log_format: Format chuỗi log
        date_format: Format ngày giờ
        max_file_size: Kích thước tối đa của file log (byte)
        backup_count: Số file log backup giữ lại
        colored_console: Có hiển thị màu trong console không
    
    Returns:
        logging.Logger: Logger tập trung
    """
    global logger, _is_initializing
    
    with lock:  # Use lock to prevent concurrent setup
        # Check if already initialized or currently initializing
        if logger is not None:
            return logger
            
        # Set initializing flag to prevent reentrant calls
        if _is_initializing:
            # Return a temporary logger to avoid blocking
            temp_logger = logging.getLogger("llm_evaluation.temp")
            if not temp_logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter(log_format, date_format))
                temp_logger.addHandler(handler)
            return temp_logger
            
        _is_initializing = True
        
        try:
            # Tạo thư mục logs nếu chưa tồn tại
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(exist_ok=True, parents=True)
            
            # Tạo tên file log dựa trên timestamp nếu không được chỉ định
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"llm_evaluation_{timestamp}.log"
            
            log_file_path = log_dir_path / log_file
            
            # Tạo root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)  # Bắt tất cả log, filter sẽ được áp dụng ở handlers
            
            # Xóa handlers cũ (nếu có)
            if root_logger.handlers:
                root_logger.handlers.clear()
            
            # Tạo formatters
            console_formatter = ColoredFormatter(log_format, date_format) if colored_console else logging.Formatter(log_format, date_format)
            file_formatter = logging.Formatter(log_format, date_format)
            
            # Tạo console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
            # Tạo file handler sử dụng ThreadSafeRotatingFileHandler
            file_handler = ThreadSafeRotatingFileHandler(
                log_file_path, maxBytes=max_file_size, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # Thiết lập level cho từng module cụ thể
            if module_levels:
                for module_name, level in module_levels.items():
                    module_logger = logging.getLogger(module_name)
                    module_logger.setLevel(level)
            
            # Tạo logger riêng cho ứng dụng
            logger = logging.getLogger("llm_evaluation")
            
            # Log thông báo khởi tạo
            logger.info(f"Logging initialized. Log file: {log_file_path}")
            logger.debug(f"Console log level: {logging.getLevelName(console_level)}")
            logger.debug(f"File log level: {logging.getLevelName(file_level)}")
            
            return logger
        finally:
            _is_initializing = False

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Lấy logger đã được cấu hình.
    
    Args:
        name: Tên của logger. Nếu None, trả về root logger
             Nếu là string, trả về logger với namespace 'llm_evaluation.{name}'
    
    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    global logger
    
    # Nếu chưa thiết lập logging, thiết lập với cấu hình mặc định
    if logger is None:
        with lock:
            if logger is None:  # Double-check pattern
                setup_logging()
    
    if name is None:
        return logger
    
    # Tạo namespace cho logger
    return logging.getLogger(f"llm_evaluation.{name}")

def log_dict(logger: logging.Logger, data: Dict[str, Any], title: str = "Data", level: int = logging.DEBUG) -> None:
    """
    Log một dictionary với format dễ đọc.
    
    Args:
        logger: Logger để ghi log
        data: Dictionary cần log
        title: Tiêu đề cho log
        level: Level log
    """
    if not logger.isEnabledFor(level):
        return
    
    logger.log(level, f"{title}:")
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")

def log_list(logger: logging.Logger, items: List[Any], title: str = "Items", level: int = logging.DEBUG) -> None:
    """
    Log một list với format dễ đọc.
    
    Args:
        logger: Logger để ghi log
        items: List cần log
        title: Tiêu đề cho log
        level: Level log
    """
    if not logger.isEnabledFor(level):
        return
    
    logger.log(level, f"{title} ({len(items)} items):")
    for i, item in enumerate(items):
        logger.log(level, f"  {i}: {item}")

def log_section(logger: logging.Logger, title: str, level: int = logging.INFO) -> None:
    """
    Log một tiêu đề phần với định dạng nổi bật.
    
    Args:
        logger: Logger để ghi log
        title: Tiêu đề phần
        level: Level log
    """
    if not logger.isEnabledFor(level):
        return
    
    separator = "=" * (len(title) + 4)
    logger.log(level, separator)
    logger.log(level, f"= {title} =")
    logger.log(level, separator)

def log_subsection(logger: logging.Logger, title: str, level: int = logging.INFO) -> None:
    """
    Log một tiêu đề phần phụ với định dạng nổi bật.
    
    Args:
        logger: Logger để ghi log
        title: Tiêu đề phần phụ
        level: Level log
    """
    if not logger.isEnabledFor(level):
        return
    
    logger.log(level, f"--- {title} ---")

def log_evaluation_start(logger: logging.Logger, model: str, prompt_type: str, total_questions: int) -> None:
    """
    Log thông tin bắt đầu đánh giá.
    
    Args:
        logger: Logger để ghi log
        model: Tên model
        prompt_type: Loại prompt
        total_questions: Tổng số câu hỏi
    """
    log_section(logger, f"Bắt đầu đánh giá {model} với prompt {prompt_type}")
    logger.info(f"Tổng số câu hỏi: {total_questions}")

def log_evaluation_progress(logger: logging.Logger, model: str, prompt_type: str, 
                         current: int, total: int, elapsed_time: float) -> None:
    """
    Log thông tin tiến độ đánh giá.
    
    Args:
        logger: Logger để ghi log
        model: Tên model
        prompt_type: Loại prompt
        current: Số thứ tự câu hỏi hiện tại
        total: Tổng số câu hỏi
        elapsed_time: Thời gian đã trôi qua (giây)
    """
    progress = (current / total) * 100
    logger.info(f"Tiến độ {model}/{prompt_type}: {current}/{total} ({progress:.1f}%) - Đã chạy {elapsed_time:.1f}s")

def log_evaluation_complete(logger: logging.Logger, model: str, prompt_type: str, 
                         total_questions: int, total_time: float, 
                         accuracy: Optional[float] = None) -> None:
    """
    Log thông tin hoàn thành đánh giá.
    
    Args:
        logger: Logger để ghi log
        model: Tên model
        prompt_type: Loại prompt
        total_questions: Tổng số câu hỏi
        total_time: Tổng thời gian (giây)
        accuracy: Độ chính xác (nếu có)
    """
    log_section(logger, f"Hoàn thành đánh giá {model} với prompt {prompt_type}")
    logger.info(f"Tổng số câu hỏi: {total_questions}")
    logger.info(f"Tổng thời gian: {total_time:.2f}s")
    logger.info(f"Thời gian trung bình: {total_time/total_questions:.2f}s/câu hỏi")
    
    if accuracy is not None:
        logger.info(f"Độ chính xác: {accuracy:.4f}")

def log_api_error(logger: logging.Logger, model: str, error: Exception, 
               question_id: Optional[str] = None, retry_count: Optional[int] = None) -> None:
    """
    Log lỗi API.
    
    Args:
        logger: Logger để ghi log
        model: Tên model
        error: Exception
        question_id: ID câu hỏi (nếu có)
        retry_count: Số lần thử lại (nếu có)
    """
    question_info = f" khi xử lý câu hỏi {question_id}" if question_id else ""
    retry_info = f" (Lần thử {retry_count})" if retry_count is not None else ""
    
    logger.error(f"Lỗi API {model}{question_info}{retry_info}: {str(error)}")
    
    # Log thêm traceback ở debug level
    import traceback
    logger.debug(f"Traceback: {traceback.format_exc()}")

def log_checkpoint(logger: logging.Logger, checkpoint_path: str, model: str, 
                prompt_type: str, question_index: int, total: int) -> None:
    """
    Log thông tin checkpoint.
    
    Args:
        logger: Logger để ghi log
        checkpoint_path: Đường dẫn file checkpoint
        model: Tên model
        prompt_type: Loại prompt
        question_index: Index câu hỏi 
        total: Tổng số câu hỏi
    """
    progress = (question_index / total) * 100
    logger.info(f"Đã lưu checkpoint tại {checkpoint_path}")
    logger.info(f"Trạng thái: {model}/{prompt_type} - {question_index}/{total} ({progress:.1f}%)")

def log_checkpoint_resume(logger: logging.Logger, checkpoint_path: str, model: str, 
                       prompt_type: str, question_index: int, total: int) -> None:
    """
    Log thông tin khôi phục từ checkpoint.
    
    Args:
        logger: Logger để ghi log
        checkpoint_path: Đường dẫn file checkpoint
        model: Tên model
        prompt_type: Loại prompt
        question_index: Index câu hỏi sẽ tiếp tục
        total: Tổng số câu hỏi
    """
    progress = (question_index / total) * 100
    log_section(logger, f"Khôi phục từ checkpoint {checkpoint_path}")
    logger.info(f"Tiếp tục từ: {model}/{prompt_type} - {question_index}/{total} ({progress:.1f}%)")

# Thiết lập logging khi module được import
if __name__ != "__main__":
    # Chỉ thiết lập mặc định khi import, không chạy trực tiếp
    pass
