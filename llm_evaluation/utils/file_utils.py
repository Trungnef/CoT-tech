"""
Tiện ích xử lý file và I/O cho framework đánh giá LLM.
Cung cấp các hàm đọc/ghi file, xử lý đường dẫn và quản lý dữ liệu.
"""

import os
import json
import yaml
import csv
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple, TextIO
import logging
import datetime
import traceback

from .logging_utils import get_logger

logger = get_logger(__name__)

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Đảm bảo thư mục tồn tại, nếu không thì tạo mới.
    
    Args:
        dir_path: Đường dẫn tới thư mục
        
    Returns:
        Path object cho thư mục
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_files(directory: Union[str, Path], 
               pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """
    Liệt kê các file trong thư mục theo pattern.
    
    Args:
        directory: Thư mục gốc
        pattern: Pattern glob để lọc file (*.json, *.txt, etc.)
        recursive: Có tìm kiếm đệ quy trong thư mục con không
        
    Returns:
        Danh sách các đường dẫn file
    """
    path = Path(directory)
    
    if not path.exists():
        logger.warning(f"Thư mục không tồn tại: {path}")
        return []
    
    if recursive:
        return list(path.glob(f"**/{pattern}"))
    else:
        return list(path.glob(pattern))

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Lấy timestamp hiện tại theo định dạng.
    
    Args:
        format_str: Định dạng timestamp
        
    Returns:
        Timestamp dạng string
    """
    return datetime.datetime.now().strftime(format_str)

def save_json(data: Any, file_path: Union[str, Path], 
              ensure_dir_exists: bool = True,
              encoding: str = 'utf-8',
              indent: int = 2,
              **kwargs) -> bool:
    """
    Lưu dữ liệu dạng JSON.
    
    Args:
        data: Dữ liệu cần lưu
        file_path: Đường dẫn file
        ensure_dir_exists: Tự động tạo thư mục nếu chưa tồn tại
        encoding: Encoding cho file
        indent: Số khoảng trắng để căn lề JSON
        kwargs: Tham số bổ sung cho json.dump()
        
    Returns:
        True nếu lưu thành công, False nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if ensure_dir_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, **kwargs)
        
        logger.debug(f"Đã lưu dữ liệu JSON vào {path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu JSON: {e}")
        logger.debug(traceback.format_exc())
        return False

def load_json(file_path: Union[str, Path], 
              default: Any = None,
              encoding: str = 'utf-8',
              **kwargs) -> Any:
    """
    Đọc dữ liệu từ file JSON.
    
    Args:
        file_path: Đường dẫn file
        default: Giá trị mặc định nếu đọc file thất bại
        encoding: Encoding của file
        kwargs: Tham số bổ sung cho json.load()
        
    Returns:
        Dữ liệu từ file hoặc default nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File không tồn tại: {path}")
            return default
        
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f, **kwargs)
        
        return data
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc JSON từ {file_path}: {e}")
        logger.debug(traceback.format_exc())
        return default

def save_yaml(data: Any, file_path: Union[str, Path],
              ensure_dir_exists: bool = True,
              encoding: str = 'utf-8',
              **kwargs) -> bool:
    """
    Lưu dữ liệu dạng YAML.
    
    Args:
        data: Dữ liệu cần lưu
        file_path: Đường dẫn file
        ensure_dir_exists: Tự động tạo thư mục nếu chưa tồn tại
        encoding: Encoding cho file
        kwargs: Tham số bổ sung cho yaml.dump()
        
    Returns:
        True nếu lưu thành công, False nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if ensure_dir_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            yaml.dump(data, f, **kwargs)
        
        logger.debug(f"Đã lưu dữ liệu YAML vào {path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu YAML: {e}")
        logger.debug(traceback.format_exc())
        return False

def load_yaml(file_path: Union[str, Path],
              default: Any = None,
              encoding: str = 'utf-8',
              **kwargs) -> Any:
    """
    Đọc dữ liệu từ file YAML.
    
    Args:
        file_path: Đường dẫn file
        default: Giá trị mặc định nếu đọc file thất bại
        encoding: Encoding của file
        kwargs: Tham số bổ sung cho yaml.load()
        
    Returns:
        Dữ liệu từ file hoặc default nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File không tồn tại: {path}")
            return default
        
        with open(path, 'r', encoding=encoding) as f:
            data = yaml.safe_load(f, **kwargs)
        
        return data
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc YAML từ {file_path}: {e}")
        logger.debug(traceback.format_exc())
        return default

def save_csv(data: List[Dict], file_path: Union[str, Path],
             ensure_dir_exists: bool = True,
             encoding: str = 'utf-8',
             **kwargs) -> bool:
    """
    Lưu dữ liệu dạng CSV.
    
    Args:
        data: Dữ liệu cần lưu (danh sách các dictionary)
        file_path: Đường dẫn file
        ensure_dir_exists: Tự động tạo thư mục nếu chưa tồn tại
        encoding: Encoding cho file
        kwargs: Tham số bổ sung cho csv.DictWriter
        
    Returns:
        True nếu lưu thành công, False nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if ensure_dir_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        if not data:
            logger.warning(f"Dữ liệu rỗng, tạo CSV trống tại {path}")
            with open(path, 'w', encoding=encoding, newline='') as f:
                pass
            return True
        
        fieldnames = list(data[0].keys())
        
        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(data)
        
        logger.debug(f"Đã lưu dữ liệu CSV vào {path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu CSV: {e}")
        logger.debug(traceback.format_exc())
        return False

def load_csv(file_path: Union[str, Path],
             default: Any = None,
             encoding: str = 'utf-8',
             **kwargs) -> List[Dict]:
    """
    Đọc dữ liệu từ file CSV.
    
    Args:
        file_path: Đường dẫn file
        default: Giá trị mặc định nếu đọc file thất bại
        encoding: Encoding của file
        kwargs: Tham số bổ sung cho csv.DictReader
        
    Returns:
        Danh sách các dictionary từ file hoặc default nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File không tồn tại: {path}")
            return default
        
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f, **kwargs)
            data = list(reader)
        
        return data
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc CSV từ {file_path}: {e}")
        logger.debug(traceback.format_exc())
        return default

def save_pickle(data: Any, file_path: Union[str, Path],
                ensure_dir_exists: bool = True,
                **kwargs) -> bool:
    """
    Lưu dữ liệu dạng pickle.
    
    Args:
        data: Dữ liệu cần lưu
        file_path: Đường dẫn file
        ensure_dir_exists: Tự động tạo thư mục nếu chưa tồn tại
        kwargs: Tham số bổ sung cho pickle.dump()
        
    Returns:
        True nếu lưu thành công, False nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if ensure_dir_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, **kwargs)
        
        logger.debug(f"Đã lưu dữ liệu pickle vào {path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu pickle: {e}")
        logger.debug(traceback.format_exc())
        return False

def load_pickle(file_path: Union[str, Path],
                default: Any = None,
                **kwargs) -> Any:
    """
    Đọc dữ liệu từ file pickle.
    
    Args:
        file_path: Đường dẫn file
        default: Giá trị mặc định nếu đọc file thất bại
        kwargs: Tham số bổ sung cho pickle.load()
        
    Returns:
        Dữ liệu từ file hoặc default nếu có lỗi
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File không tồn tại: {path}")
            return default
        
        with open(path, 'rb') as f:
            data = pickle.load(f, **kwargs)
        
        return data
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc pickle từ {file_path}: {e}")
        logger.debug(traceback.format_exc())
        return default

def safe_file_write(file_path: Union[str, Path], 
                    write_function: callable,
                    ensure_dir_exists: bool = True,
                    use_temp: bool = True) -> bool:
    """
    Ghi file với tính năng an toàn sử dụng file tạm thời.
    
    Args:
        file_path: Đường dẫn file
        write_function: Hàm ghi dữ liệu, nhận tham số là file handle
        ensure_dir_exists: Tự động tạo thư mục nếu chưa tồn tại
        use_temp: Sử dụng file tạm thời để đảm bảo an toàn
        
    Returns:
        True nếu ghi thành công, False nếu có lỗi
    """
    path = Path(file_path)
    
    if ensure_dir_exists:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    if not use_temp:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                write_function(f)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi ghi file {path}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    # Sử dụng file tạm thời
    try:
        # Tạo file tạm thời cùng thư mục với file đích
        temp_dir = path.parent
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir, encoding='utf-8') as temp_file:
            temp_path = temp_file.name
            write_function(temp_file)
        
        # Di chuyển file tạm thời thành file đích
        shutil.move(temp_path, path)
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi ghi file an toàn {path}: {e}")
        logger.debug(traceback.format_exc())
        
        # Dọn dẹp nếu có lỗi
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return False

def get_file_size(file_path: Union[str, Path], 
                  human_readable: bool = False) -> Union[int, str]:
    """
    Lấy kích thước file.
    
    Args:
        file_path: Đường dẫn file
        human_readable: Trả về dạng đọc được (KB, MB, GB) thay vì byte
        
    Returns:
        Kích thước file
    """
    path = Path(file_path)
    
    if not path.exists():
        return 0
    
    size_bytes = path.stat().st_size
    
    if not human_readable:
        return size_bytes
    
    # Chuyển đổi sang dạng đọc được
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = size_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}" 