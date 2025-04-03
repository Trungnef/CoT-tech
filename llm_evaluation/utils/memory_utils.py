"""
Tiện ích quản lý bộ nhớ cho framework đánh giá LLM.
Cung cấp các hàm giám sát bộ nhớ, thu gom rác và giải phóng tài nguyên.
"""

import os
import gc
import psutil
import threading
import time
import weakref
import sys
from typing import Dict, List, Any, Optional, Callable, Set, Union

from .logging_utils import get_logger

logger = get_logger(__name__)

# Ngưỡng cảnh báo và ngưỡng xử lý
MEMORY_WARNING_THRESHOLD = 75  # Phần trăm
MEMORY_CRITICAL_THRESHOLD = 85  # Phần trăm
MEMORY_MONITORING_INTERVAL = 5  # Giây

# Đối tượng quản lý bộ nhớ toàn cục
_memory_manager = None
_lock = threading.RLock()

class MemoryManager:
    """
    Quản lý bộ nhớ và tài nguyên.
    """
    
    def __init__(self, warning_threshold: float = MEMORY_WARNING_THRESHOLD,
                 critical_threshold: float = MEMORY_CRITICAL_THRESHOLD,
                 monitoring_interval: float = MEMORY_MONITORING_INTERVAL):
        """
        Khởi tạo quản lý bộ nhớ.
        
        Args:
            warning_threshold: Ngưỡng cảnh báo (phần trăm)
            critical_threshold: Ngưỡng xử lý (phần trăm)
            monitoring_interval: Chu kỳ giám sát (giây)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_interval = monitoring_interval
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._cleanup_callbacks = []
        self._resource_registry = weakref.WeakValueDictionary()
        self._tracked_objects = {}
        self._last_warning_time = 0
        self._clean_in_progress = False
        
        # Thông tin quá trình
        self.process = psutil.Process(os.getpid())
    
    def start_monitoring(self):
        """
        Bắt đầu giám sát bộ nhớ trong một thread riêng.
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.debug("Đã có thread giám sát bộ nhớ đang chạy")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True,
            name="MemoryMonitoringThread"
        )
        self._monitoring_thread.start()
        logger.info("Đã bắt đầu giám sát bộ nhớ")
    
    def stop_monitoring(self):
        """
        Dừng giám sát bộ nhớ.
        """
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=2.0)
        self._monitoring_thread = None
        logger.info("Đã dừng giám sát bộ nhớ")
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """
        Đăng ký hàm callback để dọn dẹp khi bộ nhớ gần hết.
        
        Args:
            callback: Hàm callback không có tham số
        """
        if callback not in self._cleanup_callbacks:
            self._cleanup_callbacks.append(callback)
            logger.debug(f"Đã đăng ký cleanup callback: {callback.__name__}")
    
    def register_resource(self, resource_id: str, resource: Any):
        """
        Đăng ký tài nguyên cần giải phóng khi cần.
        
        Args:
            resource_id: ID của tài nguyên
            resource: Đối tượng tài nguyên
        """
        self._resource_registry[resource_id] = resource
        logger.debug(f"Đã đăng ký tài nguyên: {resource_id}")
    
    def release_resource(self, resource_id: str) -> bool:
        """
        Giải phóng tài nguyên theo ID.
        
        Args:
            resource_id: ID của tài nguyên
            
        Returns:
            True nếu giải phóng thành công, False nếu không tìm thấy
        """
        if resource_id in self._resource_registry:
            del self._resource_registry[resource_id]
            logger.debug(f"Đã giải phóng tài nguyên: {resource_id}")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Lấy thông tin sử dụng bộ nhớ hiện tại.
        
        Returns:
            Dict với các thông tin về bộ nhớ
        """
        # Bộ nhớ của quá trình
        process_info = self.process.memory_info()
        rss = process_info.rss / (1024 * 1024)  # MB
        vms = process_info.vms / (1024 * 1024)  # MB
        
        # Bộ nhớ hệ thống
        system_memory = psutil.virtual_memory()
        system_used_percent = system_memory.percent
        system_available = system_memory.available / (1024 * 1024)  # MB
        system_total = system_memory.total / (1024 * 1024)  # MB
        
        return {
            "process_rss_mb": rss,
            "process_vms_mb": vms,
            "system_used_percent": system_used_percent,
            "system_available_mb": system_available,
            "system_total_mb": system_total
        }
    
    def track_object(self, obj_id: str, obj: Any, size_hint: Optional[int] = None):
        """
        Theo dõi một đối tượng để quản lý bộ nhớ.
        
        Args:
            obj_id: ID của đối tượng
            obj: Đối tượng cần theo dõi
            size_hint: Ước lượng kích thước (bytes), nếu biết
        """
        self._tracked_objects[obj_id] = {
            "obj": obj,
            "size_hint": size_hint,
            "creation_time": time.time()
        }
        logger.debug(f"Đã bắt đầu theo dõi đối tượng: {obj_id}")
    
    def untrack_object(self, obj_id: str) -> bool:
        """
        Dừng theo dõi một đối tượng.
        
        Args:
            obj_id: ID của đối tượng
            
        Returns:
            True nếu thành công, False nếu không tìm thấy
        """
        if obj_id in self._tracked_objects:
            del self._tracked_objects[obj_id]
            logger.debug(f"Đã dừng theo dõi đối tượng: {obj_id}")
            return True
        return False
    
    def get_tracked_objects(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy danh sách các đối tượng đang theo dõi.
        
        Returns:
            Dict với key là object_id và value là thông tin
        """
        return {
            obj_id: {
                "type": type(info["obj"]).__name__,
                "size_hint": info.get("size_hint"),
                "creation_time": info.get("creation_time"),
                "age_seconds": time.time() - info.get("creation_time", time.time())
            }
            for obj_id, info in self._tracked_objects.items()
        }
    
    def cleanup_memory(self, forced: bool = False):
        """
        Dọn dẹp bộ nhớ.
        
        Args:
            forced: Nếu True, bắt buộc dọn dẹp ngay cả khi chưa đến ngưỡng
        """
        with _lock:
            # Kiểm tra xem có đang dọn dẹp không
            if self._clean_in_progress:
                logger.debug("Đã có quá trình dọn dẹp bộ nhớ đang chạy")
                return
            
            try:
                self._clean_in_progress = True
                memory_info = self.get_memory_usage()
                
                if forced or memory_info["system_used_percent"] >= self.critical_threshold:
                    logger.info(
                        f"Bắt đầu dọn dẹp bộ nhớ (forced={forced}, "
                        f"used={memory_info['system_used_percent']:.1f}%)"
                    )
                    
                    # Thu gom rác
                    collected = gc.collect(2)
                    logger.debug(f"Đã thu gom {collected} đối tượng")
                    
                    # Gọi các hàm callback dọn dẹp
                    for callback in self._cleanup_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Lỗi khi gọi callback {callback.__name__}: {str(e)}")
                    
                    # Giải phóng một số tài nguyên lớn nếu cần
                    if forced or memory_info["system_used_percent"] >= self.critical_threshold + 5:
                        for obj_id in list(self._tracked_objects.keys()):
                            self.untrack_object(obj_id)
                    
                    # In thông tin bộ nhớ sau khi dọn dẹp
                    new_memory_info = self.get_memory_usage()
                    savings = memory_info["system_used_percent"] - new_memory_info["system_used_percent"]
                    
                    logger.info(
                        f"Đã dọn dẹp bộ nhớ: {savings:.1f}% tiết kiệm, "
                        f"hiện tại: {new_memory_info['system_used_percent']:.1f}%"
                    )
            
            finally:
                self._clean_in_progress = False
    
    def _monitor_memory(self):
        """
        Hàm giám sát bộ nhớ chạy trong thread riêng.
        """
        logger.debug("Bắt đầu vòng lặp giám sát bộ nhớ")
        
        while not self._stop_monitoring.is_set():
            try:
                memory_info = self.get_memory_usage()
                usage_percent = memory_info["system_used_percent"]
                
                # Kiểm tra ngưỡng cảnh báo
                if usage_percent >= self.warning_threshold:
                    # Giới hạn tần suất cảnh báo
                    current_time = time.time()
                    if current_time - self._last_warning_time >= 60:  # 1 phút
                        self._last_warning_time = current_time
                        logger.warning(
                            f"Cảnh báo: Sử dụng bộ nhớ cao ({usage_percent:.1f}%), "
                            f"Process RSS: {memory_info['process_rss_mb']:.1f} MB"
                        )
                
                # Kiểm tra ngưỡng xử lý
                if usage_percent >= self.critical_threshold:
                    logger.warning(
                        f"Vượt ngưỡng xử lý: {usage_percent:.1f}% > {self.critical_threshold}%, "
                        f"tiến hành dọn dẹp bộ nhớ"
                    )
                    self.cleanup_memory()
                
                # Chờ đến chu kỳ tiếp theo
                self._stop_monitoring.wait(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Lỗi trong quá trình giám sát bộ nhớ: {str(e)}")
                # Chờ một chút trước khi thử lại
                self._stop_monitoring.wait(self.monitoring_interval * 2)
        
        logger.debug("Kết thúc vòng lặp giám sát bộ nhớ")


def get_memory_manager() -> MemoryManager:
    """
    Lấy đối tượng quản lý bộ nhớ toàn cục.
    
    Returns:
        Đối tượng MemoryManager
    """
    global _memory_manager
    
    with _lock:
        if _memory_manager is None:
            _memory_manager = MemoryManager()
    
    return _memory_manager


def start_memory_monitoring():
    """
    Bắt đầu giám sát bộ nhớ.
    """
    manager = get_memory_manager()
    manager.start_monitoring()


def stop_memory_monitoring():
    """
    Dừng giám sát bộ nhớ.
    """
    global _memory_manager
    
    if _memory_manager is not None:
        _memory_manager.stop_monitoring()


def cleanup_memory(forced: bool = False):
    """
    Dọn dẹp bộ nhớ.
    
    Args:
        forced: Nếu True, bắt buộc dọn dẹp
    """
    manager = get_memory_manager()
    manager.cleanup_memory(forced=forced)


def get_memory_usage() -> Dict[str, float]:
    """
    Lấy thông tin sử dụng bộ nhớ hiện tại.
    
    Returns:
        Dict với các thông tin về bộ nhớ
    """
    manager = get_memory_manager()
    return manager.get_memory_usage()


def track_object(obj_id: str, obj: Any, size_hint: Optional[int] = None):
    """
    Theo dõi một đối tượng để quản lý bộ nhớ.
    
    Args:
        obj_id: ID của đối tượng
        obj: Đối tượng cần theo dõi
        size_hint: Ước lượng kích thước (bytes), nếu biết
    """
    manager = get_memory_manager()
    manager.track_object(obj_id, obj, size_hint)


def untrack_object(obj_id: str) -> bool:
    """
    Dừng theo dõi một đối tượng.
    
    Args:
        obj_id: ID của đối tượng
        
    Returns:
        True nếu thành công, False nếu không tìm thấy
    """
    manager = get_memory_manager()
    return manager.untrack_object(obj_id)


def register_cleanup_callback(callback: Callable[[], None]):
    """
    Đăng ký hàm callback để dọn dẹp khi bộ nhớ gần hết.
    
    Args:
        callback: Hàm callback không có tham số
    """
    manager = get_memory_manager()
    manager.register_cleanup_callback(callback)


def estimate_object_size(obj: Any) -> int:
    """
    Ước tính kích thước của đối tượng trong bộ nhớ.
    
    Args:
        obj: Đối tượng cần ước tính
        
    Returns:
        Kích thước ước tính (bytes)
    """
    import sys
    import numpy as np
    import pandas as pd
    
    try:
        # Kiểm tra các loại đối tượng đặc biệt
        if obj is None:
            return 0
        
        # NumPy array
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        
        # Pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        
        # Pandas Series
        if isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True)
        
        # List, dict, set
        if isinstance(obj, (list, tuple)):
            return sys.getsizeof(obj) + sum(estimate_object_size(x) for x in obj)
        
        if isinstance(obj, dict):
            return sys.getsizeof(obj) + sum(
                estimate_object_size(k) + estimate_object_size(v) for k, v in obj.items()
            )
        
        if isinstance(obj, set):
            return sys.getsizeof(obj) + sum(estimate_object_size(x) for x in obj)
        
        # String
        if isinstance(obj, str):
            return sys.getsizeof(obj)
        
        # Các loại dữ liệu cơ bản
        if isinstance(obj, (int, float, bool)):
            return sys.getsizeof(obj)
        
        # Đối tượng khác
        return sys.getsizeof(obj)
    
    except Exception as e:
        logger.warning(f"Lỗi khi ước tính kích thước đối tượng: {str(e)}")
        return sys.getsizeof(obj)


class MemoryUsageDecorator:
    """
    Decorator theo dõi và ghi lại mức sử dụng bộ nhớ của một hàm.
    """
    
    def __init__(self, logger=None):
        """
        Khởi tạo decorator.
        
        Args:
            logger: Đối tượng logger để ghi log
        """
        self.logger = logger or get_logger(__name__)
    
    def __call__(self, func):
        """
        Gọi decorator.
        
        Args:
            func: Hàm cần theo dõi
            
        Returns:
            Hàm wrapper
        """
        def wrapper(*args, **kwargs):
            # Lấy thông tin bộ nhớ trước khi chạy
            gc.collect()
            memory_before = get_memory_usage()
            
            # Ghi log
            self.logger.debug(
                f"Bắt đầu {func.__name__}: "
                f"Bộ nhớ hiện tại: {memory_before['process_rss_mb']:.1f} MB, "
                f"Hệ thống: {memory_before['system_used_percent']:.1f}%"
            )
            
            # Chạy hàm
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Thu gom rác và lấy thông tin bộ nhớ sau khi chạy
            gc.collect()
            memory_after = get_memory_usage()
            
            # Tính sự thay đổi
            memory_diff = memory_after['process_rss_mb'] - memory_before['process_rss_mb']
            
            # Ghi log
            self.logger.debug(
                f"Kết thúc {func.__name__} ({elapsed:.2f}s): "
                f"Bộ nhớ hiện tại: {memory_after['process_rss_mb']:.1f} MB, "
                f"Thay đổi: {memory_diff:+.1f} MB"
            )
            
            return result
        
        return wrapper


# Khởi tạo giám sát bộ nhớ khi module được import
start_memory_monitoring()

# Đăng ký hàm xử lý khi thoát
import atexit
atexit.register(stop_memory_monitoring) 