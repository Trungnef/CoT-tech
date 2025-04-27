import unittest
import time
import os
import sys
from unittest.mock import patch, MagicMock
import json
import requests

# Thêm thư mục cha vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thiết lập logging
from llm_evaluation.utils.logging_setup import setup_logging
logger = setup_logging(log_file="test_system_resilience.log")

# Import các module cần thiết
from llm_evaluation.core.model_interface import ModelInterface, generate_text
from llm_evaluation.core.evaluator import Evaluator

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

class TestSystemResilience(unittest.TestCase):
    """Kiểm tra khả năng chống chịu lỗi của hệ thống"""
    
    def setUp(self):
        """Thiết lập cho mỗi test case"""
        self.model_interface = ModelInterface()
        
        # Tạo dữ liệu mẫu cho Evaluator
        self.sample_questions = [
            {"id": "q1", "question": "2 + 2 = ?", "answer": "4", "type": "math"},
            {"id": "q2", "question": "Thủ đô của Việt Nam là gì?", "answer": "Hà Nội", "type": "fact"}
        ]
        
    def test_service_unavailable_retry_in_generate_text(self):
        """Kiểm tra xử lý lỗi 503 trong hàm generate_text"""
        
        # Đếm số lần gọi
        call_count = 0
        
        # Mock completion.create để trả về lỗi 503 và sau đó thành công
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
        
        # Tạo mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        
        # Patch _get_groq_client để trả về mock client
        with patch.object(ModelInterface, '_get_groq_client', return_value=mock_client):
            # Gọi hàm cần kiểm tra
            result = generate_text("groq", "Prompt test")
            
            # Kiểm tra kết quả
            self.assertEqual(call_count, 3, "Phải có đúng 3 lần gọi API (2 lỗi + 1 thành công)")
            self.assertEqual(result[0], "Đây là phản hồi thành công sau khi retry")
            self.assertFalse(result[1].get("has_error", False))
    
    def test_evaluator_retry_on_503(self):
        """Kiểm tra xử lý lỗi 503 trong evaluator"""
        
        # Tạo evaluator với cấu hình tối thiểu
        evaluator = Evaluator(
            models_to_evaluate=["groq"],
            prompts_to_evaluate=["basic"],
            questions=self.sample_questions,
            batch_size=1,
            checkpoint_frequency=10,
            reasoning_evaluation_enabled=True
        )
        
        # Mock phương thức _evaluate_single_reasoning 
        original_method = evaluator._evaluate_single_reasoning
        
        # Counter để theo dõi số lần gọi
        call_count = 0
        service_unavailable_count = 0
        success_count = 0
        
        def mock_evaluate_single_reasoning(*args, **kwargs):
            nonlocal call_count, service_unavailable_count, success_count
            call_count += 1
            
            # Giả lập lỗi 503 trong 2 lần gọi đầu
            if call_count <= 2:
                service_unavailable_count += 1
                # Bắt chước lỗi 503 được lan truyền
                raise Exception("UNKNOWN_ERROR: Lỗi không rõ. Đang thử lại.... Original error: Error code: 503")
            
            # Sau đó trả về kết quả thành công
            success_count += 1
            return {
                'accuracy': 5,
                'reasoning': 5,
                'completeness': 5,
                'explanation': 5,
                'cultural_context': 5,
                'average': 5,
                'comment': "Phản hồi thành công sau khi retry"
            }
        
        # Patch phương thức để sử dụng phiên bản mock
        with patch.object(evaluator, '_evaluate_single_reasoning', side_effect=mock_evaluate_single_reasoning):
            # Gọi phương thức đánh giá một cặp câu hỏi-câu trả lời
            result = evaluator._evaluate_single_reasoning(
                question="2 + 2 = ?",
                correct_answer="4",
                model_answer="2 + 2 = 4"
            )
            
            # Kiểm tra kết quả
            self.assertEqual(service_unavailable_count, 2, "Phải có đúng 2 lần xuất hiện lỗi 503")
            self.assertEqual(success_count, 1, "Phải có đúng 1 lần thành công")
            self.assertEqual(result['average'], 5, "Kết quả đánh giá phải thành công")
            self.assertEqual(result['comment'], "Phản hồi thành công sau khi retry")
    
    def test_end_to_end_error_handling(self):
        """Kiểm tra xử lý lỗi end-to-end từ model_interface đến evaluator"""
        
        # Counter để theo dõi
        api_call_count = 0
        
        # Mock API call để trả về lỗi 503 và sau đó thành công
        def mock_create(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            
            # Lần gọi đầu tiên và thứ hai trả về lỗi 503
            if api_call_count <= 2:
                error_response = {
                    "error": {
                        "message": "Service Unavailable",
                        "type": "internal_server_error"
                    }
                }
                mock_resp = MockResponse(503, json_data=error_response, 
                                      text=json.dumps(error_response))
                mock_resp.raise_for_status()
            
            # Lần gọi khác trả về thành công
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = """
            {
                "accuracy": 5,
                "reasoning": 5,
                "completeness": 5,
                "explanation": 5,
                "cultural_context": 5,
                "average": 5,
                "comment": "Phản hồi tốt"
            }
            """
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]
            
            return mock_completion
        
        # Tạo mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        
        # Tạo evaluator
        evaluator = Evaluator(
            models_to_evaluate=["groq"],
            prompts_to_evaluate=["basic"],
            questions=self.sample_questions,
            batch_size=1,
            checkpoint_frequency=10,
            reasoning_evaluation_enabled=True
        )
        
        # Patch cả ModelInterface._get_groq_client
        with patch.object(ModelInterface, '_get_groq_client', return_value=mock_client):
            # Patch generate_text để sử dụng ModelInterface đã được mock
            with patch('llm_evaluation.core.evaluator.generate_text', 
                      side_effect=lambda *args, **kwargs: self.model_interface._generate_with_groq(*args[1:], **kwargs)):
                # Gọi phương thức đánh giá reasoning
                result = evaluator._evaluate_single_reasoning(
                    question="2 + 2 = ?",
                    correct_answer="4",
                    model_answer="2 + 2 = 4"
                )
                
                # Kiểm tra kết quả
                self.assertEqual(api_call_count, 3, "Phải có đúng 3 lần gọi API (2 lỗi + 1 thành công)")
                self.assertEqual(result['average'], 5, "Kết quả đánh giá phải thành công")

if __name__ == "__main__":
    unittest.main() 