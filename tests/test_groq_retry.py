import time
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import requests

# Thêm thư mục cha vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_evaluation.utils.logging_setup import setup_logging
from llm_evaluation.core.model_interface import ModelInterface

# Thiết lập logging
logger = setup_logging(log_file="test_groq_retry.log")

class MockResponse:
    """Giả lập phản hồi HTTP"""
    def __init__(self, status_code, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}
        self.text = text
        self.content = text.encode('utf-8')
        
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error: {self.text}",
                response=self
            )

class TestGroqRetry(unittest.TestCase):
    """Kiểm tra cơ chế retry cho Groq API"""
    
    def setUp(self):
        """Thiết lập trước mỗi test case"""
        self.model_interface = ModelInterface()
        
    def test_service_unavailable_retry(self):
        """Kiểm tra xử lý lỗi 503 Service Unavailable"""
        
        # Đếm số lần gọi API
        call_count = 0
        
        # Mock client.chat.completions.create
        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Lần gọi đầu tiên và thứ hai trả về lỗi 503
            if call_count <= 2:
                error_response = {
                    "error": {
                        "message": "Service Unavailable",
                        "type": "internal_server_error"
                    }
                }
                mock_resp = MockResponse(503, json_data=error_response, 
                                      text=json.dumps(error_response))
                mock_resp.raise_for_status()
            
            # Lần gọi thứ ba trở đi trả về thành công
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Đây là phản hồi thành công sau khi retry"
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]
            
            return mock_completion
        
        # Patch _get_groq_client để trả về mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        
        with patch.object(self.model_interface, '_get_groq_client', return_value=mock_client):
            # Gọi hàm cần kiểm tra
            response, stats = self.model_interface._generate_with_groq(
                "Prompt test", {"model": "llama3-70b-8192"}
            )
            
            # Kiểm tra kết quả
            self.assertEqual(call_count, 3, "Phải có đúng 3 lần gọi API (2 lỗi + 1 thành công)")
            self.assertEqual(response, "Đây là phản hồi thành công sau khi retry")
            self.assertFalse(stats.get("has_error", False), "Kết quả không được có lỗi")
    
    def test_max_retries_exceeded(self):
        """Kiểm tra khi vượt quá số lần retry tối đa"""
        
        # Mock client.chat.completions.create để luôn trả về lỗi 503
        def mock_create(*args, **kwargs):
            error_response = {
                "error": {
                    "message": "Service Unavailable",
                    "type": "internal_server_error"
                }
            }
            mock_resp = MockResponse(503, json_data=error_response, 
                                   text=json.dumps(error_response))
            mock_resp.raise_for_status()
        
        # Patch _get_groq_client để trả về mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        
        with patch.object(self.model_interface, '_get_groq_client', return_value=mock_client):
            # Gọi hàm cần kiểm tra
            response, stats = self.model_interface._generate_with_groq(
                "Prompt test", {"model": "llama3-70b-8192"}
            )
            
            # Kiểm tra kết quả
            self.assertTrue(stats.get("has_error", False), "Kết quả phải có lỗi")
            self.assertIn("[Error:", response, "Phản hồi phải chứa thông báo lỗi")
            
    def test_rate_limit_error(self):
        """Kiểm tra xử lý lỗi 429 Rate Limit"""
        
        # Đếm số lần gọi API
        call_count = 0
        
        # Mock client.chat.completions.create
        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Lần gọi đầu tiên trả về lỗi 429
            if call_count == 1:
                error_response = {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error"
                    }
                }
                mock_resp = MockResponse(429, json_data=error_response, 
                                      text=json.dumps(error_response),
                                      headers={"Retry-After": "1"})
                mock_resp.raise_for_status()
            
            # Lần gọi thứ hai trở đi trả về thành công
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Đây là phản hồi thành công sau khi retry"
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]
            
            return mock_completion
        
        # Patch _get_groq_client để trả về mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        
        with patch.object(self.model_interface, '_get_groq_client', return_value=mock_client):
            # Gọi hàm cần kiểm tra
            response, stats = self.model_interface._generate_with_groq(
                "Prompt test", {"model": "llama3-70b-8192"}
            )
            
            # Kiểm tra kết quả
            self.assertEqual(call_count, 2, "Phải có đúng 2 lần gọi API (1 lỗi + 1 thành công)")
            self.assertEqual(response, "Đây là phản hồi thành công sau khi retry")
            self.assertFalse(stats.get("has_error", False), "Kết quả không được có lỗi")

if __name__ == "__main__":
    unittest.main() 