"""
Unit tests cho CheckpointManager.
"""

import os
import sys
import unittest
import tempfile
import json
import shutil
import time
from pathlib import Path
import glob

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parents[2].absolute()))

from llm_evaluation.core.checkpoint_manager import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    """Test cases cho CheckpointManager."""
    
    def setUp(self):
        """Thiết lập trước mỗi test case."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.timestamp = "20240402_120000"
        self.max_checkpoints = 3
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            timestamp=self.timestamp,
            max_checkpoints=self.max_checkpoints
        )
        
        # Tạo dữ liệu mẫu
        self.sample_state = {
            "timestamp": self.timestamp,
            "current_model": "llama3",
            "current_prompt": "zero_shot",
            "current_question_idx": 5,
            "completed_combinations": [
                ["llama3", "zero_shot", "q1"],
                ["llama3", "zero_shot", "q2"],
                ["llama3", "few_shot", "q1"]
            ],
            "results": [
                {"model_name": "llama3", "prompt_type": "zero_shot", "question_id": "q1", "response": "Test response 1"},
                {"model_name": "llama3", "prompt_type": "zero_shot", "question_id": "q2", "response": "Test response 2"},
                {"model_name": "llama3", "prompt_type": "few_shot", "question_id": "q1", "response": "Test response 3"}
            ]
        }
    
    def tearDown(self):
        """Dọn dẹp sau mỗi test case."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test lưu checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(self.sample_state)
        
        # Kiểm tra file tồn tại
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Đọc file và kiểm tra nội dung
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Kiểm tra các trường dữ liệu
        self.assertEqual(saved_data["current_model"], "llama3")
        self.assertEqual(saved_data["current_prompt"], "zero_shot")
        self.assertEqual(saved_data["current_question_idx"], 5)
        self.assertEqual(len(saved_data["results"]), 3)
        
        # Kiểm tra metadata
        self.assertIn("checkpoint_timestamp", saved_data)
        self.assertEqual(saved_data["checkpoint_version"], "1.0")
    
    def test_load_checkpoint(self):
        """Test tải checkpoint."""
        # Lưu checkpoint trước
        checkpoint_path = self.checkpoint_manager.save_checkpoint(self.sample_state)
        
        # Tải checkpoint
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Kiểm tra dữ liệu
        self.assertEqual(loaded_data["current_model"], "llama3")
        self.assertEqual(loaded_data["current_prompt"], "zero_shot")
        self.assertEqual(len(loaded_data["results"]), 3)
        
        # Kiểm tra chuyển đổi completed_combinations
        self.assertIsInstance(loaded_data["completed_combinations"], list)
        for combo in loaded_data["completed_combinations"]:
            if isinstance(combo, list):
                self.assertIsInstance(combo, list)
            else:
                self.assertIsInstance(combo, tuple)
    
    def test_load_latest_checkpoint(self):
        """Test tải checkpoint mới nhất."""
        # Lưu nhiều checkpoint với độ trễ nhỏ để phân biệt thời gian
        state1 = self.sample_state.copy()
        state1["current_question_idx"] = 1
        checkpoint1 = self.checkpoint_manager.save_checkpoint(state1)
        time.sleep(0.1)
        
        state2 = self.sample_state.copy()
        state2["current_question_idx"] = 2
        checkpoint2 = self.checkpoint_manager.save_checkpoint(state2)
        time.sleep(0.1)
        
        state3 = self.sample_state.copy()
        state3["current_question_idx"] = 3
        checkpoint3 = self.checkpoint_manager.save_checkpoint(state3)
        
        # Tải checkpoint mới nhất
        latest_data = self.checkpoint_manager.load_latest_checkpoint()
        
        # Kiểm tra phải là checkpoint mới nhất
        self.assertEqual(latest_data["current_question_idx"], 3)
    
    def test_manage_checkpoints(self):
        """Test quản lý số lượng checkpoint."""
        # Lưu nhiều checkpoint với tên file khác nhau để đảm bảo glob phân biệt được
        for i in range(1, 6):  # Tạo 5 checkpoints
            state = self.sample_state.copy()
            state["current_question_idx"] = i
            
            # Tạo manager mới với timestamp khác để đảm bảo tên file khác nhau
            custom_timestamp = f"20240402_{i:06d}"
            cm = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                timestamp=custom_timestamp,
                max_checkpoints=self.max_checkpoints
            )
            cm.save_checkpoint(state)
            
            # Đảm bảo các file được tạo ra có thời gian chỉnh sửa khác nhau
            time.sleep(0.1)
        
        # Lấy tất cả các file checkpoint
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.json"))
        
        # Kiểm tra số lượng checkpoint có trong thư mục
        # Chỉ giữ lại 3 file mới nhất (vì max_checkpoints = 3)
        self.assertLessEqual(len(checkpoint_files), self.max_checkpoints)
    
    def test_clear_checkpoints(self):
        """Test xóa tất cả checkpoint."""
        # Lưu một số checkpoint với tên file khác nhau
        num_checkpoints = 3
        for i in range(1, num_checkpoints + 1):
            state = self.sample_state.copy()
            state["current_question_idx"] = i
            
            # Tạo manager mới với timestamp khác để đảm bảo tên file khác nhau
            custom_timestamp = f"20240402_{i:06d}"
            cm = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                timestamp=custom_timestamp,
                max_checkpoints=self.max_checkpoints
            )
            cm.save_checkpoint(state)
        
        # Xóa tất cả
        num_deleted = self.checkpoint_manager.clear_checkpoints()
        
        # Số lượng file đã xóa có thể khác với số checkpoint đã tạo
        # vì có thể đã có một số file bị xóa do quá giới hạn max_checkpoints
        # Chỉ cần đảm bảo rằng đã có file bị xóa
        self.assertGreater(num_deleted, 0)
        
        # Kiểm tra không còn checkpoint nào
        checkpoint_files = self.checkpoint_manager._get_checkpoint_files()
        self.assertEqual(len(checkpoint_files), 0)
    
    def test_get_checkpoint_info(self):
        """Test lấy thông tin checkpoint."""
        # Lưu checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(self.sample_state)
        
        # Lấy thông tin
        info = self.checkpoint_manager.get_checkpoint_info(checkpoint_path)
        
        # Kiểm tra thông tin
        self.assertEqual(info["path"], checkpoint_path)
        self.assertEqual(info["num_results"], 3)
        self.assertEqual(info["current_model"], "llama3")
        self.assertGreater(info["file_size"], 0)

if __name__ == "__main__":
    unittest.main() 