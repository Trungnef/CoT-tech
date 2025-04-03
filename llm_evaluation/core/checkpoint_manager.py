"""
CheckpointManager quản lý việc lưu và tải checkpoint trong quá trình đánh giá LLM.
Cho phép tạm dừng và tiếp tục quá trình đánh giá, giúp hệ thống có khả năng phục hồi sau lỗi.
"""

import os
import json
import logging
import glob
from pathlib import Path
import datetime
import shutil
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import traceback

logger = logging.getLogger("checkpoint_manager")

class CheckpointManager:
    """
    Quản lý lưu và tải checkpoint trong quá trình đánh giá LLM.
    """
    
    def __init__(self, checkpoint_dir: str = None, 
                timestamp: str = None, 
                max_checkpoints: int = 5,
                compress: bool = False):
        """
        Khởi tạo CheckpointManager.
        
        Args:
            checkpoint_dir (str): Thư mục lưu checkpoint, mặc định là "./checkpoints"
            timestamp (str): Timestamp sử dụng trong tên file, mặc định là thời gian hiện tại
            max_checkpoints (int): Số lượng checkpoint tối đa lưu giữ
            compress (bool): Có nén checkpoint không (chưa triển khai)
        """
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_checkpoints = max_checkpoints
        self.compress = compress
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.debug(f"Khởi tạo CheckpointManager với thư mục: {self.checkpoint_dir}")
        
    def save_checkpoint(self, state: Dict[str, Any]) -> str:
        """
        Lưu trạng thái hiện tại vào checkpoint.
        
        Args:
            state (Dict): Trạng thái cần lưu, chứa results, completed_combinations, 
                         current_model, current_prompt, current_question_idx, v.v.
                        
        Returns:
            str: Đường dẫn đến file checkpoint đã lưu
        """
        try:
            # Thêm metadata
            checkpoint_data = state.copy()
            checkpoint_data["checkpoint_timestamp"] = datetime.datetime.now().isoformat()
            checkpoint_data["checkpoint_version"] = "1.0"
            
            # Chuẩn bị đường dẫn file
            checkpoint_filename = f"checkpoint_{self.timestamp}_{len(checkpoint_data.get('results', []))}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Lưu trạng thái
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Đã lưu checkpoint tại: {checkpoint_path} "
                      f"({len(checkpoint_data.get('results', []))} kết quả)")
            
            # Quản lý số lượng checkpoint
            self._manage_checkpoints()
            
            return checkpoint_path
        
        except Exception as e:
            logger.error(f"Lỗi khi lưu checkpoint: {e}")
            logger.error(traceback.format_exc())
            
            # Thử lưu đến vị trí dự phòng nếu có lỗi
            try:
                emergency_path = os.path.join(self.checkpoint_dir, f"emergency_{self.timestamp}.json")
                with open(emergency_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False)
                logger.info(f"Đã lưu checkpoint khẩn cấp tại: {emergency_path}")
                return emergency_path
            except:
                logger.critical("Không thể lưu checkpoint khẩn cấp!")
                return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Tải checkpoint từ đường dẫn cụ thể.
        
        Args:
            checkpoint_path (str): Đường dẫn đến file checkpoint
            
        Returns:
            Dict: Dữ liệu checkpoint hoặc None nếu không thành công
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Không tìm thấy checkpoint tại: {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Chuyển đổi danh sách completed_combinations thành set tuple nếu cần
            if "completed_combinations" in checkpoint_data:
                if isinstance(checkpoint_data["completed_combinations"], list):
                    checkpoint_data["completed_combinations"] = [
                        tuple(combo) if isinstance(combo, list) else combo 
                        for combo in checkpoint_data["completed_combinations"]
                    ]
            
            logger.info(f"Đã tải checkpoint từ: {checkpoint_path} "
                      f"({len(checkpoint_data.get('results', []))} kết quả)")
                      
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Lỗi khi tải checkpoint {checkpoint_path}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def load_latest_checkpoint(self) -> Dict[str, Any]:
        """
        Tải checkpoint gần nhất theo timestamp.
        
        Returns:
            Dict: Dữ liệu checkpoint hoặc None nếu không có checkpoint
        """
        checkpoint_files = self._get_checkpoint_files()
        
        if not checkpoint_files:
            logger.warning(f"Không tìm thấy checkpoint trong {self.checkpoint_dir}")
            return None
        
        latest_checkpoint = checkpoint_files[-1]  # Đã được sắp xếp tăng dần
        return self.load_checkpoint(latest_checkpoint)
    
    def _get_checkpoint_files(self) -> List[str]:
        """
        Lấy danh sách tất cả các file checkpoint, sắp xếp theo thời gian.
        
        Returns:
            List[str]: Danh sách đường dẫn đến các file checkpoint
        """
        # Lấy tất cả file .json trong thư mục checkpoint
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.json")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # Sắp xếp theo thời gian tạo file
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        
        return checkpoint_files
    
    def _manage_checkpoints(self):
        """
        Quản lý số lượng checkpoint, xóa cũ nhất nếu vượt quá giới hạn.
        """
        checkpoint_files = self._get_checkpoint_files()
        
        if len(checkpoint_files) > self.max_checkpoints:
            # Xóa các checkpoint cũ nhất để giữ số lượng dưới max_checkpoints
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    logger.debug(f"Đã xóa checkpoint cũ: {file_path}")
                except Exception as e:
                    logger.warning(f"Không thể xóa checkpoint cũ {file_path}: {e}")
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Xóa một checkpoint cụ thể.
        
        Args:
            checkpoint_path (str): Đường dẫn đến file checkpoint cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Không tìm thấy checkpoint để xóa: {checkpoint_path}")
            return False
        
        try:
            os.remove(checkpoint_path)
            logger.info(f"Đã xóa checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa checkpoint {checkpoint_path}: {e}")
            return False
    
    def clear_checkpoints(self) -> int:
        """
        Xóa tất cả các checkpoint.
        
        Returns:
            int: Số lượng checkpoint đã xóa
        """
        checkpoint_files = self._get_checkpoint_files()
        count = 0
        
        for file_path in checkpoint_files:
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                logger.warning(f"Không thể xóa checkpoint {file_path}: {e}")
        
        logger.info(f"Đã xóa {count}/{len(checkpoint_files)} checkpoint")
        return count
    
    def get_checkpoint_info(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Lấy thông tin tóm tắt về checkpoint.
        
        Args:
            checkpoint_path (str): Đường dẫn đến file checkpoint, nếu None sẽ dùng checkpoint mới nhất
            
        Returns:
            Dict: Thông tin tóm tắt về checkpoint
        """
        if checkpoint_path is None:
            checkpoint_files = self._get_checkpoint_files()
            if not checkpoint_files:
                logger.warning("Không có checkpoint để lấy thông tin")
                return {}
            checkpoint_path = checkpoint_files[-1]
        
        try:
            # Tải tối thiểu thông tin từ checkpoint
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Trích xuất thông tin quan trọng
            info = {
                "path": checkpoint_path,
                "timestamp": data.get("checkpoint_timestamp", "unknown"),
                "version": data.get("checkpoint_version", "unknown"),
                "num_results": len(data.get("results", [])),
                "num_completed": len(data.get("completed_combinations", [])),
                "current_model": data.get("current_model", ""),
                "current_prompt": data.get("current_prompt", ""),
                "file_size": os.path.getsize(checkpoint_path)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin checkpoint {checkpoint_path}: {e}")
            return {}
