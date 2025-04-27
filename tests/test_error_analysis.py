import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần test
from llm_evaluation.core.result_analyzer import ResultAnalyzer

class TestErrorAnalysis(unittest.TestCase):
    """Các test case cho chức năng phân tích lỗi"""
    
    def setUp(self):
        """Thiết lập dữ liệu test"""
        # Tạo dữ liệu mẫu
        data = []
        models = ["model_A", "model_B"]
        question_types = ["math", "reasoning", "factual"]
        prompt_types = ["zero_shot", "few_shot", "cot"]
        
        # Tạo 10 câu hỏi
        for i in range(10):
            # Mỗi câu hỏi được trả lời bởi cả hai model với tất cả các prompt type
            question_text = f"Câu hỏi #{i+1}"
            question_id = f"q{i+1}"
            q_type = question_types[i % len(question_types)]
            correct_answer = f"Đáp án đúng cho câu hỏi #{i+1}"
            
            for model in models:
                for prompt in prompt_types:
                    # Xác định xem câu trả lời có đúng không (một số đúng, một số sai)
                    is_correct = (i + ord(model[-1]) + len(prompt)) % 3 != 0
                    
                    # Tạo câu trả lời mẫu
                    if is_correct:
                        response = f"Đáp án đúng cho câu hỏi #{i+1}"
                    else:
                        # Tạo các loại lỗi khác nhau
                        error_types = [
                            f"Tôi không biết câu trả lời của câu hỏi #{i+1}",  # Knowledge Error
                            f"Kết quả là {(i+5) * 2}, vì {i+1} cộng {i+1} bằng {2*(i+1)}",  # Reasoning Error
                            f"Kết quả là {i*i} vì {i} nhân {i} bằng {i*i}",  # Calculation Error
                            "Tôi không thể trả lời câu hỏi này",  # Non-answer
                            "Chủ đề này rất thú vị. Nhưng tôi muốn nói về chủ đề khác.",  # Off-topic
                            f"Câu hỏi này đang hỏi về {['lịch sử', 'địa lý', 'thể thao'][i%3]}"  # Misunderstanding
                        ]
                        response = error_types[(i + ord(model[-1])) % len(error_types)]
                    
                    # Thêm vào dataset
                    data.append({
                        "model_name": model,
                        "prompt_type": prompt,
                        "question_id": question_id,
                        "question_type": q_type,
                        "question_text": question_text,
                        "response": response,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    })
        
        self.test_df = pd.DataFrame(data)
        
        # Mock cho _analyze_single_error để tránh gọi API thực tế
        self.error_types = [
            "Knowledge Error",
            "Reasoning Error", 
            "Calculation Error",
            "Non-answer",
            "Off-topic",
            "Misunderstanding",
            "Other"
        ]
    
    @patch('llm_evaluation.core.result_analyzer.ResultAnalyzer._analyze_single_error')
    def test_analyze_errors(self, mock_analyze):
        """Test phương thức analyze_errors"""
        # Thiết lập mock cho _analyze_single_error
        def side_effect(question, model_answer, correct_answer):
            # Trả về một loại lỗi ngẫu nhiên nhưng xác định
            error_idx = (hash(question) + hash(model_answer)) % len(self.error_types)
            return {
                "error_type": self.error_types[error_idx],
                "explanation": f"Explanation for {self.error_types[error_idx]}"
            }
            
        mock_analyze.side_effect = side_effect
        
        # Khởi tạo ResultAnalyzer
        analyzer = ResultAnalyzer(results_df=self.test_df)
        
        # Lọc các hàng sai để có danh sách ngắn hơn
        error_df = self.test_df[self.test_df['is_correct'] == False]
        
        # Chạy phân tích lỗi
        result_df = analyzer.analyze_errors(error_df, sample_size=10)
        
        # Kiểm tra kết quả
        # 1. Kiểm tra xem các cột mới đã được thêm chưa
        self.assertIn('error_type', result_df.columns)
        self.assertIn('error_explanation', result_df.columns)
        
        # 2. Kiểm tra xem có bao nhiêu hàng đã được phân tích
        error_analyzed = sum(result_df['error_type'] != '')
        # Số lượng phân tích không nên vượt quá sample_size và số lượng lỗi
        self.assertLessEqual(error_analyzed, 10)
        self.assertLessEqual(error_analyzed, len(error_df))
        self.assertGreater(error_analyzed, 0)  # Đảm bảo ít nhất 1 hàng được phân tích
        
        # 3. Kiểm tra các loại lỗi đã được gán
        error_types_used = result_df['error_type'].unique()
        for error_type in error_types_used:
            if error_type:  # Bỏ qua chuỗi rỗng
                self.assertIn(error_type, self.error_types)
    
    @patch('llm_evaluation.core.result_analyzer.ResultAnalyzer._analyze_single_error')
    def test_compute_error_metrics(self, mock_analyze):
        """Test phương thức _compute_error_metrics"""
        # Thiết lập mock cho _analyze_single_error
        def side_effect(question, model_answer, correct_answer):
            # Trả về một loại lỗi ngẫu nhiên nhưng xác định
            error_idx = (hash(question) + hash(model_answer)) % len(self.error_types)
            return {
                "error_type": self.error_types[error_idx],
                "explanation": f"Explanation for {self.error_types[error_idx]}"
            }
            
        mock_analyze.side_effect = side_effect
        
        # Khởi tạo ResultAnalyzer
        analyzer = ResultAnalyzer(results_df=self.test_df)
        
        # Lọc các hàng sai và phân tích lỗi
        error_df = self.test_df[self.test_df['is_correct'] == False]
        result_df = analyzer.analyze_errors(error_df, sample_size=20)
        
        # Tính toán metrics
        metrics = analyzer._compute_error_metrics(result_df)
        
        # Kiểm tra cấu trúc của metrics
        self.assertIn('overall', metrics)
        self.assertIn('error_counts', metrics['overall'])
        self.assertIn('error_percentages', metrics['overall'])
        
        # Kiểm tra metrics theo model
        self.assertIn('by_model', metrics)
        for model in self.test_df['model_name'].unique():
            self.assertIn(model, metrics['by_model'])
            model_metrics = metrics['by_model'][model]
            self.assertIn('error_counts', model_metrics)
            self.assertIn('error_percentages', model_metrics)
        
        # Kiểm tra metrics theo prompt type
        self.assertIn('by_prompt_type', metrics)
        for prompt in self.test_df['prompt_type'].unique():
            self.assertIn(prompt, metrics['by_prompt_type'])
            prompt_metrics = metrics['by_prompt_type'][prompt]
            self.assertIn('error_counts', prompt_metrics)
            self.assertIn('error_percentages', prompt_metrics)
            
        # Kiểm tra tổng error_percentages
        total_pct = sum(metrics['overall']['error_percentages'].values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Kiểm tra giá trị trong error_counts
        total_counts = sum(metrics['overall']['error_counts'].values())
        self.assertEqual(total_counts, sum(result_df['error_type'] != ''))
    
    def test_parse_error_analysis(self):
        """Test phương thức _parse_error_analysis"""
        analyzer = ResultAnalyzer(results_df=self.test_df)
        
        # Test đầu vào khác nhau
        test_cases = [
            {
                "response": "Error Type: Knowledge Error\nGiải thích: Model không có đủ kiến thức",
                "expected": {"error_type": "Knowledge Error", "explanation": "Model không có đủ kiến thức"}
            },
            {
                "response": "Loại lỗi: Reasoning Error\nBrief Explanation: Lỗi trong quá trình suy luận",
                "expected": {"error_type": "Reasoning Error", "explanation": "Lỗi trong quá trình suy luận"}
            },
            {
                "response": "Model đã mắc lỗi loại: Calculation Error\ngiải thích: Sai phép tính",
                "expected": {"error_type": "Calculation Error", "explanation": "Model đã mắc lỗi loại: Calculation Error\ngiải thích: Sai phép tính"}
            },
            {
                "response": "1. Knowledge Error\n\nGiải thích ngắn gọn: Model thiếu thông tin cần thiết",
                "expected": {"error_type": "Knowledge Error", "explanation": "Model thiếu thông tin cần thiết"}
            }
        ]
        
        for case in test_cases:
            result = analyzer._parse_error_analysis(case["response"])
            self.assertEqual(result["error_type"], case["expected"]["error_type"])
            if "explanation" in result and "explanation" in case["expected"]:
                self.assertIn(case["expected"]["explanation"], result["explanation"])
    
    def test_normalize_error_type(self):
        """Test phương thức _normalize_error_type"""
        analyzer = ResultAnalyzer(results_df=self.test_df)
        
        # Test các trường hợp khác nhau
        test_cases = [
            {"input": "Knowledge Error", "expected": "Knowledge Error"},
            {"input": "Lỗi kiến thức", "expected": "Knowledge Error"},
            {"input": "thiếu thông tin cần thiết", "expected": "Knowledge Error"},
            {"input": "Reasoning Error", "expected": "Reasoning Error"},
            {"input": "lỗi suy luận logic", "expected": "Reasoning Error"},
            {"input": "Calculation Error", "expected": "Calculation Error"},
            {"input": "lỗi tính toán số học", "expected": "Calculation Error"},
            {"input": "không trả lời được", "expected": "Non-answer"},
            {"input": "từ chối trả lời", "expected": "Non-answer"},
            {"input": "lạc đề hoàn toàn", "expected": "Off-topic"},
            {"input": "Off-topic", "expected": "Off-topic"},
            {"input": "Hiểu nhầm câu hỏi", "expected": "Misunderstanding"},
            {"input": "misunderstanding", "expected": "Misunderstanding"},
            {"input": "1. Knowledge Error", "expected": "Knowledge Error"},
            {"input": "2: Reasoning Error", "expected": "Reasoning Error"},
            {"input": "3. Calculation Error", "expected": "Calculation Error"},
            {"input": "4. Non-answer", "expected": "Non-answer"},
            {"input": "5: Off-topic", "expected": "Off-topic"},
            {"input": "6: Misunderstanding", "expected": "Misunderstanding"},
            {"input": "7. Other", "expected": "Other"},
            {"input": "Lỗi khác", "expected": "Other"},
            {"input": "Không xác định", "expected": "Other"}
        ]
        
        for case in test_cases:
            result = analyzer._normalize_error_type(case["input"])
            self.assertEqual(result, case["expected"], f"Input: {case['input']}, Expected: {case['expected']}, Got: {result}")

if __name__ == '__main__':
    unittest.main() 