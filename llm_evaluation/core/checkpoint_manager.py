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
        # Nếu checkpoint_dir là tương đối, lấy đường dẫn tuyệt đối
        if checkpoint_dir:
            self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        else:
            self.checkpoint_dir = os.path.abspath("./checkpoints")
            
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
            
            # Lưu thêm đường dẫn trong dữ liệu để tham chiếu sau này
            result_count = len(checkpoint_data.get('results', []))
            checkpoint_filename = f"checkpoint_{self.timestamp}_{result_count}.json"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            checkpoint_data["checkpoint_path"] = checkpoint_path
            
            # Lưu trạng thái
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Đã lưu checkpoint tại: {checkpoint_path} "
                      f"({result_count} kết quả)")
            
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
            
            # Lưu đường dẫn checkpoint vào dữ liệu nếu chưa có
            if "checkpoint_path" not in checkpoint_data:
                checkpoint_data["checkpoint_path"] = checkpoint_path
                
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
        
        # Checkpoint gần nhất
        latest_checkpoint = checkpoint_files[-1]  # Đã được sắp xếp tăng dần
        
        # Log chi tiết hơn về checkpoint sắp được tải
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(latest_checkpoint))) if "run_" in latest_checkpoint else "unknown"
        logger.info(f"Tải checkpoint từ thư mục chạy: {run_dir}")
        logger.info(f"Đường dẫn checkpoint: {latest_checkpoint}")
        
        # Kiểm tra thử thư mục cha (nếu đang ở trong thư mục con run_TIMESTAMP)
        parent_dir = os.path.dirname(os.path.dirname(self.checkpoint_dir))
        if os.path.basename(os.path.dirname(self.checkpoint_dir)).startswith("run_"):
            parent_checkpoint_dir = os.path.join(parent_dir, "checkpoints")
            if os.path.exists(parent_checkpoint_dir):
                logger.debug(f"Kiểm tra thêm checkpoints trong thư mục cha: {parent_checkpoint_dir}")
                
                # Lấy tất cả checkpoint trong thư mục cha
                parent_pattern = os.path.join(parent_checkpoint_dir, "checkpoint_*.json")
                parent_checkpoints = glob.glob(parent_pattern)
                parent_checkpoints.sort(key=lambda x: os.path.getmtime(x))
                
                # Nếu có checkpoint trong thư mục cha, so sánh với checkpoint hiện tại
                if parent_checkpoints:
                    latest_parent = parent_checkpoints[-1]
                    if os.path.exists(latest_parent):
                        # Kiểm tra thời gian sửa đổi để xác định cái nào mới hơn
                        if os.path.getmtime(latest_parent) > os.path.getmtime(latest_checkpoint):
                            latest_checkpoint = latest_parent
                            logger.info(f"Tìm thấy checkpoint mới hơn trong thư mục cha: {latest_parent}")
        
        return self.load_checkpoint(latest_checkpoint)
    
    def _get_checkpoint_files(self) -> List[str]:
        """
        Lấy danh sách tất cả các file checkpoint, sắp xếp theo thời gian.
        
        Returns:
            List[str]: Danh sách đường dẫn đến các file checkpoint
        """
        # Lấy tất cả file checkpoint trong thư mục hiện tại
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.json")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # Thêm cả các file checkpoint khẩn cấp
        emergency_pattern = os.path.join(self.checkpoint_dir, "emergency_*.json")
        emergency_files = glob.glob(emergency_pattern)
        checkpoint_files.extend(emergency_files)
        
        # Nếu đang ở thư mục run cụ thể mà --resume được sử dụng, ưu tiên checkpoint ở thư mục hiện tại
        run_dir_match = None
        if "run_" in self.checkpoint_dir:
            run_dir_match = os.path.basename(os.path.dirname(self.checkpoint_dir))
            if checkpoint_files:
                logger.info(f"Tìm thấy checkpoint trong thư mục chạy {run_dir_match}")
                # Sắp xếp checkpoint trong thư mục hiện tại theo thời gian (tăng dần)
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
                # Lọc ra các file thực sự tồn tại
                checkpoint_files = [f for f in checkpoint_files if os.path.exists(f)]
                logger.debug(f"Tìm thấy {len(checkpoint_files)} checkpoint files trong {run_dir_match}.")
                return checkpoint_files
        
        # Chỉ khi không tìm thấy checkpoint trong thư mục hiện tại, kiểm tra các thư mục khác
        # Kiểm tra thư mục results nếu đang ở trong thư mục con
        # Đây là để tìm các checkpoint từ các lần chạy trước
        root_dir = os.path.dirname(os.path.dirname(self.checkpoint_dir))
        if "checkpoints" in self.checkpoint_dir and os.path.exists(root_dir):
            # Kiểm tra thư mục results/checkpoints nếu nó tồn tại
            results_checkpoints = os.path.join(root_dir, "checkpoints")
            if os.path.exists(results_checkpoints) and results_checkpoints != self.checkpoint_dir:
                logger.debug(f"Quét thêm thư mục checkpoint chung: {results_checkpoints}")
                results_pattern = os.path.join(results_checkpoints, "checkpoint_*.json")
                results_files = glob.glob(results_pattern)
                checkpoint_files.extend(results_files)
                
            # Kiểm tra các thư mục run khác
            run_dirs = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("run_")]
            
            for run_dir in run_dirs:
                run_checkpoint_dir = os.path.join(root_dir, run_dir, "checkpoints")
                if os.path.exists(run_checkpoint_dir) and run_checkpoint_dir != self.checkpoint_dir:
                    logger.debug(f"Quét thêm thư mục checkpoint từ lần chạy khác: {run_checkpoint_dir}")
                    run_pattern = os.path.join(run_checkpoint_dir, "checkpoint_*.json")
                    run_files = glob.glob(run_pattern)
                    checkpoint_files.extend(run_files)
        
        # Lọc ra các file thực sự tồn tại
        checkpoint_files = [f for f in checkpoint_files if os.path.exists(f)]
        
        # Sắp xếp theo thời gian sửa đổi (tăng dần: cũ -> mới)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
        
        logger.debug(f"Tìm thấy {len(checkpoint_files)} checkpoint files.")
        return checkpoint_files
    
    def _manage_checkpoints(self):
        """
        Quản lý số lượng checkpoint, xóa cũ nhất nếu vượt quá giới hạn.
        """
        checkpoint_files = self._get_checkpoint_files()
        
        # Lọc các file chỉ có trong thư mục checkpoint hiện tại
        current_dir_files = [f for f in checkpoint_files if os.path.dirname(f) == self.checkpoint_dir]
        
        if len(current_dir_files) > self.max_checkpoints:
            # Xóa các checkpoint cũ nhất để giữ số lượng dưới max_checkpoints
            files_to_remove = current_dir_files[:-self.max_checkpoints]
            
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
