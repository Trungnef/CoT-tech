"""
Unit tests cho các module trong thư mục utils.
"""

import os
import sys
import unittest
import tempfile
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_utils import ensure_dir, load_json, save_json, get_timestamp
from utils.logging_utils import get_logger, setup_logging
from utils.text_utils import clean_text, remove_diacritics, normalize_vietnamese_text
from utils.metrics_utils import calculate_binary_metrics, calculate_exact_match_accuracy
from utils.config_utils import Config, load_config, validate_config, create_default_config

class TestFileUtils(unittest.TestCase):
    """Tests cho file_utils.py."""
    
    def setUp(self):
        # Tạo thư mục tạm thời để test
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_file.json")
    
    def test_ensure_dir(self):
        """Test hàm ensure_dir tạo thư mục đúng cách."""
        test_path = os.path.join(self.test_dir, "nested", "folder")
        ensure_dir(test_path)
        self.assertTrue(os.path.exists(test_path))
    
    def test_save_load_json(self):
        """Test hàm save_json và load_json hoạt động đúng."""
        test_data = {"key": "value", "nested": {"a": 1, "b": 2}}
        save_json(test_data, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        
        loaded_data = load_json(self.test_file)
        self.assertEqual(loaded_data, test_data)
    
    def test_get_timestamp(self):
        """Test hàm get_timestamp trả về định dạng hợp lệ."""
        timestamp = get_timestamp()
        # Kiểm tra định dạng YYYYMMDD_HHMMSS
        self.assertTrue(len(timestamp) == 15)
        self.assertTrue("_" in timestamp)
        self.assertTrue(timestamp.replace("_", "").isdigit())

class TestLoggingUtils(unittest.TestCase):
    """Tests cho logging_utils.py."""
    
    def setUp(self):
        # Tạo file log tạm thời để test
        self.test_log_file = tempfile.mktemp(suffix='.log')
    
    def tearDown(self):
        # Xóa file log nếu tồn tại
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
    
    def test_setup_logging(self):
        """Test hàm setup_logging cấu hình đúng."""
        logger = setup_logging(
            log_level="INFO",
            log_file=self.test_log_file,
            console=True
        )
        self.assertIsNotNone(logger)
        
        # Ghi log để kiểm tra
        logger.info("Test log message")
        
        # Kiểm tra file log
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test log message", content)
    
    def test_get_logger(self):
        """Test hàm get_logger trả về logger với tên đúng."""
        setup_logging(log_level="INFO", log_file=None, console=False)
        logger = get_logger("test_module")
        self.assertEqual(logger.name, "test_module")

class TestTextUtils(unittest.TestCase):
    """Tests cho text_utils.py."""
    
    def test_clean_text(self):
        """Test hàm clean_text làm sạch văn bản đúng cách."""
        text = "  Đây Là Một Câu Văn Bản. 123  "
        
        # Test mặc định: chuyển thành chữ thường
        result = clean_text(text)
        self.assertEqual(result, "  đây là một câu văn bản. 123  ")
        
        # Test xóa dấu câu
        result = clean_text(text, remove_punctuation=True)
        self.assertEqual(result, "  đây là một câu văn bản 123  ")
        
        # Test xóa số
        result = clean_text(text, remove_numbers=True)
        self.assertEqual(result, "  đây là một câu văn bản.   ")
        
        # Test xóa khoảng trắng thừa
        result = clean_text(text, remove_whitespace=True)
        self.assertEqual(result, "đây là một câu văn bản. 123")
        
        # Test kết hợp các tùy chọn
        result = clean_text(text, remove_punctuation=True, remove_numbers=True, remove_whitespace=True)
        self.assertEqual(result, "đây là một câu văn bản")
    
    def test_remove_diacritics(self):
        """Test hàm remove_diacritics loại bỏ dấu tiếng Việt đúng cách."""
        text = "Xin chào thế giới! Đây là tiếng Việt."
        result = remove_diacritics(text)
        self.assertEqual(result, "Xin chao the gioi! Day la tieng Viet.")
    
    def test_normalize_vietnamese_text(self):
        """Test hàm normalize_vietnamese_text chuẩn hóa đúng cách."""
        text = "Đây  là  câu . Có dấu  câu  không chuẩn !"
        result = normalize_vietnamese_text(text)
        self.assertEqual(result, "Đây là câu. Có dấu câu không chuẩn!")

class TestMetricsUtils(unittest.TestCase):
    """Tests cho metrics_utils.py."""
    
    def test_calculate_binary_metrics(self):
        """Test hàm calculate_binary_metrics tính toán đúng."""
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        
        metrics = calculate_binary_metrics(y_true, y_pred)
        
        # Kiểm tra các metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Kiểm tra giá trị
        self.assertEqual(metrics["accuracy"], 0.6)  # 3/5
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
    
    def test_calculate_exact_match_accuracy(self):
        """Test hàm calculate_exact_match_accuracy tính toán đúng."""
        predictions = ["apple", "banana", "orange"]
        references = ["apple", "banana", "cherry"]
        
        # Test mặc định: phân biệt hoa thường = False
        accuracy = calculate_exact_match_accuracy(predictions, references)
        self.assertEqual(accuracy, 2/3)
        
        # Test phân biệt hoa thường = True
        predictions = ["Apple", "banana", "orange"]
        references = ["apple", "banana", "cherry"]
        accuracy = calculate_exact_match_accuracy(predictions, references, case_sensitive=True)
        self.assertEqual(accuracy, 1/3)
        
        # Test normalize = False
        matches = calculate_exact_match_accuracy(predictions, references, normalize=False)
        self.assertEqual(matches, 1)

class TestConfigUtils(unittest.TestCase):
    """Tests cho config_utils.py."""
    
    def setUp(self):
        self.test_config_file = tempfile.mktemp(suffix='.yaml')
    
    def tearDown(self):
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_create_default_config(self):
        """Test hàm create_default_config tạo cấu hình mặc định đúng."""
        config = create_default_config()
        
        # Kiểm tra các thành phần
        self.assertIsInstance(config, Config)
        self.assertIn("gpt-4", config.models)
        self.assertIn("gpt-3.5-turbo", config.models)
        self.assertIn("zero-shot", config.prompt_types)
        self.assertIn("few-shot", config.prompt_types)
        self.assertIn("cot", config.prompt_types)
    
    @patch('utils.config_utils.load_yaml')
    def test_load_config(self, mock_load_yaml):
        """Test hàm load_config đọc cấu hình đúng."""
        # Tạo dữ liệu cấu hình giả
        mock_config_data = {
            "version": "1.0.0",
            "models": {
                "test-model": {
                    "api_type": "openai",
                    "model_name": "test-model",
                    "max_tokens": 100
                }
            },
            "prompt_types": {
                "test-prompt": {
                    "template": "Test template"
                }
            },
            "evaluation": {
                "metrics": ["accuracy"],
                "output_dir": "test_outputs"
            }
        }
        
        # Giả lập load_yaml trả về dữ liệu cấu hình
        mock_load_yaml.return_value = mock_config_data
        
        # Giả lập tồn tại file
        with patch('os.path.exists', return_value=True):
            config = load_config("dummy.yaml")
        
        # Kiểm tra cấu hình đã được tải
        self.assertEqual(config.version, "1.0.0")
        self.assertIn("test-model", config.models)
        self.assertIn("test-prompt", config.prompt_types)
        self.assertEqual(config.evaluation.metrics, ["accuracy"])
    
    def test_validate_config(self):
        """Test hàm validate_config xác thực đúng cách."""
        # Cấu hình hợp lệ
        valid_config = {
            "version": "1.0.0",
            "models": {
                "test-model": {
                    "api_type": "openai",
                    "model_name": "test-model"
                }
            },
            "prompt_types": {
                "test-prompt": {
                    "template": "Test template"
                }
            },
            "evaluation": {
                "metrics": ["accuracy"]
            }
        }
        
        # Cấu hình thiếu trường bắt buộc
        invalid_config = {
            "version": "1.0.0",
            "models": {
                "test-model": {
                    "api_type": "openai"
                    # Thiếu model_name (bắt buộc)
                }
            },
            "prompt_types": {
                "test-prompt": {
                    "template": "Test template"
                }
            },
            "evaluation": {
                "metrics": ["accuracy"]
            }
        }
        
        # Kiểm tra cấu hình hợp lệ
        result = validate_config(valid_config)
        self.assertTrue(result)
        
        # Kiểm tra cấu hình không hợp lệ
        from jsonschema.exceptions import ValidationError
        with self.assertRaises(ValidationError):
            validate_config(invalid_config)

if __name__ == '__main__':
    unittest.main() 