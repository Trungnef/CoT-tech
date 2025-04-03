"""
Tiện ích quản lý logging cho framework đánh giá LLM.
Cung cấp cấu hình logging nhất quán, định dạng, màu sắc, và xử lý đầu ra.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import traceback

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# Định nghĩa các level logging tùy chỉnh
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

# Định nghĩa format mặc định
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ColoredFormatter(logging.Formatter):
    """
    Formatter cho phép hiển thị log với màu sắc.
    Kết hợp với colorama.
    """
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and HAS_COLORAMA
        
        # Định nghĩa màu cho từng level
        if self.use_colors:
            self.COLORS = {
                'DEBUG': Fore.CYAN,
                'VERBOSE': Fore.BLUE,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Style.BRIGHT,
                'RESET': Style.RESET_ALL
            }
        else:
            self.COLORS = {}
    
    def format(self, record):
        # Format chuỗi log
        log_message = super().format(record)
        
        # Thêm màu sắc nếu được hỗ trợ
        if self.use_colors and record.levelname in self.COLORS:
            log_message = self.COLORS[record.levelname] + log_message + self.COLORS['RESET']
            
        return log_message

def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  log_to_console: bool = True,
                  log_format: str = DEFAULT_FORMAT,
                  date_format: str = DEFAULT_DATE_FORMAT,
                  use_colors: bool = True,
                  log_dir: Optional[str] = None) -> logging.Logger:
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng.
    
    Args:
        log_level: Level logging (DEBUG, VERBOSE, INFO, WARNING, ERROR, CRITICAL)
        log_file: Đường dẫn đến file log, nếu None thì không log vào file
        log_to_console: Có log ra console không
        log_format: Format của log message
        date_format: Format của trường datetime
        use_colors: Có sử dụng màu sắc cho log không
        log_dir: Thư mục chứa log file, sẽ tự động tạo nếu không tồn tại
        
    Returns:
        Logger đã được cấu hình
    """
    # Chuyển đổi tên level thành số
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        # Xử lý trường hợp VERBOSE
        if log_level.upper() == "VERBOSE":
            numeric_level = VERBOSE
        else:
            numeric_level = logging.INFO
            print(f"Invalid log level: {log_level}, using INFO")
    
    # Tạo root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Xóa các handler hiện có để tránh trùng lặp
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Cấu hình formatter với màu sắc
    formatter = ColoredFormatter(fmt=log_format, datefmt=date_format, use_colors=use_colors)
    
    # Thêm StreamHandler nếu cần
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Thêm FileHandler nếu cần
    if log_file:
        # Tạo thư mục chứa log nếu cần
        if log_dir:
            full_path = os.path.join(log_dir, log_file)
            os.makedirs(log_dir, exist_ok=True)
        else:
            full_path = log_file
        
        try:
            file_handler = logging.FileHandler(full_path, encoding='utf-8')
            file_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Không thể tạo log file: {e}")
            traceback.print_exc()
    
    # Tạo logger mới cho ứng dụng
    logger = logging.getLogger("llm_evaluation")
    logger.debug(f"Đã thiết lập logging với level {log_level}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger với tên đã cho.
    
    Args:
        name: Tên của logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_function_call(logger: logging.Logger = None, level: int = logging.DEBUG):
    """
    Decorator để log thông tin về function call.
    
    Args:
        logger: Logger instance, nếu None sẽ tạo logger mới dựa trên tên module
        level: Level logging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            logger.log(level, f"Gọi {func.__name__}(args={args}, kwargs={kwargs})")
            result = func(*args, **kwargs)
            logger.log(level, f"{func.__name__} trả về {result}")
            return result
        
        return wrapper
    
    return decorator

def verbose(self, message, *args, **kwargs):
    """
    Method để log ở level VERBOSE.
    
    Args:
        message: Message cần log
        args, kwargs: Tham số bổ sung 
    """
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

# Thêm method verbose vào class Logger
logging.Logger.verbose = verbose 