import sys
import os
import unittest
import pandas as pd
import numpy as np
from collections import Counter

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần test
from llm_evaluation.utils.metrics_utils import calculate_consistency_metrics
from llm_evaluation.core.result_analyzer import ResultAnalyzer

class TestConsistencyMetrics(unittest.TestCase):
    """Test cases cho hàm tính toán metrics về tính nhất quán"""
    
    def setUp(self):
        """Thiết lập dữ liệu test"""
        # Tạo dữ liệu mẫu cho việc test
        self.test_responses = [
            ["Đáp án là 42", "Đáp án là 42", "Đáp án là 42", "Đáp án là 42"],  # 100% nhất quán
            ["Kết quả là 10", "Kết quả là 15", "Kết quả là 10", "Kết quả là 10"],  # 75% nhất quán
            ["A", "B", "C", "A", "D", "A"],  # 50% nhất quán
            ["X"]  # Trường hợp đặc biệt: chỉ 1 câu trả lời
        ]
        
        self.test_group_keys = ["group_full_consistent", "group_mostly_consistent", "group_diverse", "group_single"]
        
        # Tạo DataFrame mẫu để test phương thức evaluate_consistency
        data = []
        models = ["model_A", "model_B"]
        question_ids = ["q1", "q2"]
        prompt_types = ["cot_self_consistency_3", "cot_self_consistency_5"]
        
        # Tạo kết quả mẫu cho model_A, q1, cot_self_consistency_3
        for i in range(3):
            data.append({
                "model_name": "model_A", 
                "question_id": "q1", 
                "prompt_type": "cot_self_consistency_3",
                "response": f"Đáp án là 42 (run {i+1})",
                "final_answer": "42"
            })
        
        # Tạo kết quả mẫu cho model_A, q2, cot_self_consistency_3
        for i in range(3):
            data.append({
                "model_name": "model_A", 
                "question_id": "q2", 
                "prompt_type": "cot_self_consistency_3",
                "response": f"Kết quả là {10 + i*5} (run {i+1})",
                "final_answer": "10" if i != 1 else "15"
            })
        
        # Tạo kết quả mẫu cho model_B, q1, cot_self_consistency_5
        answers = ["A", "B", "C", "A", "D"]
        for i in range(5):
            data.append({
                "model_name": "model_B", 
                "question_id": "q1", 
                "prompt_type": "cot_self_consistency_5",
                "response": f"Lựa chọn {answers[i]} (run {i+1})",
                "final_answer": answers[i]
            })
        
        # Tạo kết quả mẫu cho model_B, q2, cot_self_consistency_5
        for i in range(5):
            data.append({
                "model_name": "model_B", 
                "question_id": "q2", 
                "prompt_type": "cot_self_consistency_5",
                "response": f"Trả lời X (run {i+1})",
                "final_answer": "X"
            })
        
        self.test_df = pd.DataFrame(data)
    
    def test_calculate_consistency_metrics(self):
        """Test tính toán consistency metrics cơ bản"""
        # Test với dữ liệu mẫu
        result = calculate_consistency_metrics(
            responses=self.test_responses,
            groupby_keys=self.test_group_keys
        )
        
        # Kiểm tra kết quả
        self.assertIn("group_full_consistent", result)
        self.assertIn("group_mostly_consistent", result)
        self.assertIn("group_diverse", result)
        self.assertIn("overall", result)
        
        # Kiểm tra giá trị cụ thể
        self.assertEqual(result["group_full_consistent"]["consistency_score"], 1.0)
        self.assertEqual(result["group_full_consistent"]["agreement_rate"], 1.0)
        self.assertEqual(result["group_full_consistent"]["unique_answers"], 1)
        
        self.assertEqual(result["group_mostly_consistent"]["consistency_score"], 0.75)
        self.assertEqual(result["group_mostly_consistent"]["agreement_rate"], 0.75)
        self.assertEqual(result["group_mostly_consistent"]["unique_answers"], 2)
        
        self.assertEqual(result["group_diverse"]["consistency_score"], 0.5)
        self.assertEqual(result["group_diverse"]["agreement_rate"], 0.5)
        self.assertEqual(result["group_diverse"]["unique_answers"], 4)
        
        # group_single không nên có trong kết quả vì bỏ qua các nhóm có <=1 câu trả lời
        self.assertNotIn("group_single", result)
        
        # Kiểm tra tính toán giá trị tổng thể
        expected_avg_consistency = (1.0 + 0.75 + 0.5) / 3
        self.assertAlmostEqual(result["overall"]["avg_consistency_score"], expected_avg_consistency)
    
    def test_evaluate_consistency_method(self):
        """Test phương thức evaluate_consistency trong ResultAnalyzer"""
        # Khởi tạo ResultAnalyzer với DataFrame mẫu
        analyzer = ResultAnalyzer(results_df=self.test_df)
        
        # Gọi phương thức evaluate_consistency
        result_df = analyzer.evaluate_consistency(self.test_df)
        
        # Kiểm tra xem các cột consistency đã được thêm vào chưa
        self.assertIn('consistency_score', result_df.columns)
        self.assertIn('consistency_agreement_rate', result_df.columns)
        self.assertIn('consistency_most_common', result_df.columns)
        self.assertIn('consistency_unique_answers', result_df.columns)
        
        # Kiểm tra kết quả cho từng nhóm
        # Group 1: model_A, q1, cot_self_consistency_3 (hoàn toàn nhất quán)
        group1 = result_df[
            (result_df['model_name'] == 'model_A') & 
            (result_df['question_id'] == 'q1') & 
            (result_df['prompt_type'] == 'cot_self_consistency_3')
        ]
        self.assertEqual(len(group1), 3)
        self.assertEqual(group1['consistency_score'].iloc[0], 1.0)
        self.assertEqual(group1['consistency_agreement_rate'].iloc[0], 1.0)
        self.assertEqual(group1['consistency_most_common'].iloc[0], '42')
        
        # Group 2: model_A, q2, cot_self_consistency_3 (2/3 nhất quán)
        group2 = result_df[
            (result_df['model_name'] == 'model_A') & 
            (result_df['question_id'] == 'q2') & 
            (result_df['prompt_type'] == 'cot_self_consistency_3')
        ]
        self.assertEqual(len(group2), 3)
        self.assertAlmostEqual(group2['consistency_score'].iloc[0], 2/3)
        self.assertAlmostEqual(group2['consistency_agreement_rate'].iloc[0], 2/3)
        self.assertEqual(group2['consistency_most_common'].iloc[0], '10')
        
        # Group 3: model_B, q1, cot_self_consistency_5 (đa dạng, 2/5 là A)
        group3 = result_df[
            (result_df['model_name'] == 'model_B') & 
            (result_df['question_id'] == 'q1') & 
            (result_df['prompt_type'] == 'cot_self_consistency_5')
        ]
        self.assertEqual(len(group3), 5)
        self.assertAlmostEqual(group3['consistency_score'].iloc[0], 2/5)
        self.assertAlmostEqual(group3['consistency_agreement_rate'].iloc[0], 2/5)
        self.assertEqual(group3['consistency_most_common'].iloc[0], 'A')
        
        # Group 4: model_B, q2, cot_self_consistency_5 (hoàn toàn nhất quán)
        group4 = result_df[
            (result_df['model_name'] == 'model_B') & 
            (result_df['question_id'] == 'q2') & 
            (result_df['prompt_type'] == 'cot_self_consistency_5')
        ]
        self.assertEqual(len(group4), 5)
        self.assertEqual(group4['consistency_score'].iloc[0], 1.0)
        self.assertEqual(group4['consistency_agreement_rate'].iloc[0], 1.0)
        self.assertEqual(group4['consistency_most_common'].iloc[0], 'X')

if __name__ == '__main__':
    unittest.main() 