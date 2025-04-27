#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script hiển thị cách sử dụng chức năng phân tích lỗi (Error Analysis)
trong llm_evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import các module cần thiết
from llm_evaluation.core.result_analyzer import ResultAnalyzer
from llm_evaluation.core.reporting import Reporting

def create_sample_data():
    """Tạo dữ liệu mẫu để minh họa chức năng phân tích lỗi"""
    # Tạo dữ liệu mẫu
    data = []
    models = ["llama", "qwen", "gemini"]
    question_types = ["math", "reasoning", "factual"]
    question_ids = [f"q{i}" for i in range(1, 21)]
    
    # Tạo các loại prompt khác nhau
    prompt_types = ["zero_shot", "few_shot", "cot"]
    
    # Tạo các loại lỗi khác nhau cho mỗi model
    error_responses = {
        "llama": [
            "Tôi không biết câu trả lời này vì thiếu thông tin.",  # Knowledge Error
            "Kết luận là 42 vì đó là câu trả lời cho mọi vấn đề.",  # Reasoning Error
            "Kết quả là 1024 vì 2 mũ 8 bằng 256.",  # Calculation Error
            "Tôi xin từ chối trả lời câu hỏi này.",  # Non-answer
        ],
        "qwen": [
            "Câu hỏi này rất thú vị, nhưng tôi muốn nói về chim cánh cụt.",  # Off-topic
            "Câu hỏi đang hỏi về lịch sử Việt Nam, nên câu trả lời là 1945.",  # Misunderstanding
            "Không đủ ngữ cảnh để trả lời chính xác.",  # Knowledge Error
            "Đáp án là A vì các phương án khác không hợp lý.",  # Reasoning Error
        ],
        "gemini": [
            "Tôi thấy câu hỏi này liên quan đến một chủ đề khác.",  # Misunderstanding
            "2 + 2 = 5 vì chúng ta đang làm việc trong modulo 1.",  # Calculation Error
            "Tôi không thể trả lời được câu hỏi này.",  # Non-answer
            "Câu hỏi này nằm ngoài phạm vi kiến thức của tôi.",  # Knowledge Error
        ]
    }
    
    # Tạo các câu trả lời đúng và sai cho mỗi model, câu hỏi và prompt
    for model in models:
        for q_id in question_ids:
            for q_type in question_types:
                # Tạo câu hỏi và đáp án đúng
                question_text = f"Câu hỏi {q_type} {q_id}"
                correct_answer = f"Đáp án đúng cho {q_type} {q_id}"
                
                for pt in prompt_types:
                    # Xác định xác suất câu trả lời đúng dựa vào model
                    # llama: 70%, qwen: 80%, gemini: 60% 
                    correct_probs = {"llama": 0.7, "qwen": 0.8, "gemini": 0.6}
                    is_correct = np.random.random() < correct_probs[model]
                    
                    if is_correct:
                        # Câu trả lời đúng
                        response = f"Đáp án đúng cho {q_type} {q_id}"
                    else:
                        # Chọn một loại lỗi ngẫu nhiên
                        response = np.random.choice(error_responses[model])
                    
                    # Thêm vào dataset
                    data.append({
                        "model_name": model,
                        "question_id": q_id,
                        "question_type": q_type,
                        "question_text": question_text,
                        "prompt_type": pt,
                        "response": response,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct,
                        "latency": np.random.uniform(0.5, 3.0)
                    })
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    return df

class MockErrorAnalyzer:
    """Mock cho phương thức phân tích lỗi để tránh gọi API thật"""
    
    def __init__(self):
        # Từ điển ánh xạ mẫu câu trả lời với loại lỗi
        self.error_patterns = {
            "không biết": "Knowledge Error",
            "thiếu thông tin": "Knowledge Error",
            "nằm ngoài phạm vi": "Knowledge Error",
            "kết luận là": "Reasoning Error",
            "vì đó là": "Reasoning Error",
            "không hợp lý": "Reasoning Error",
            "kết quả là": "Calculation Error",
            "bằng": "Calculation Error",
            "modulo": "Calculation Error",
            "từ chối": "Non-answer",
            "không thể trả lời": "Non-answer",
            "thú vị, nhưng": "Off-topic",
            "muốn nói về": "Off-topic",
            "liên quan đến một chủ đề khác": "Off-topic",
            "đang hỏi về": "Misunderstanding",
            "câu hỏi này liên quan đến": "Misunderstanding"
        }
    
    def analyze_error(self, question, response, correct_answer):
        """Phân tích loại lỗi dựa trên mẫu"""
        # Mặc định là Other
        error_type = "Other"
        explanation = "Không thể xác định loại lỗi cụ thể"
        
        # Tìm pattern phù hợp nhất
        response_lower = response.lower()
        for pattern, err_type in self.error_patterns.items():
            if pattern.lower() in response_lower:
                error_type = err_type
                explanation = f"Phát hiện dấu hiệu của lỗi {err_type} trong câu trả lời."
                break
        
        return {"error_type": error_type, "explanation": explanation}

def run_error_analysis_demo():
    """Thực hiện phân tích lỗi và tạo báo cáo"""
    print("Tạo dữ liệu mẫu cho phân tích lỗi...")
    df = create_sample_data()
    
    # Thống kê cơ bản
    print(f"Đã tạo DataFrame với {len(df)} dòng.")
    print(f"Các model: {df['model_name'].unique()}")
    print(f"Độ chính xác tổng thể: {df['is_correct'].mean():.2%}")
    
    # Đếm số lượng câu trả lời sai
    error_count = (~df['is_correct']).sum()
    print(f"Số lượng câu trả lời sai: {error_count} ({error_count/len(df):.2%})")
    
    # Tạo timestamp cho output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_dir, "examples", "outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo ResultAnalyzer
    print("\nĐang phân tích lỗi...")
    analyzer = ResultAnalyzer(results_df=df, verbose=True)
    
    # Patch phương thức _analyze_single_error để tránh gọi API thật
    mock_analyzer = MockErrorAnalyzer()
    original_method = analyzer._analyze_single_error
    
    # Thay thế phương thức bằng mock
    analyzer._analyze_single_error = lambda q, a, c: mock_analyzer.analyze_error(q, a, c)
    
    # Phân tích lỗi
    df = analyzer.analyze_errors(df, sample_size=50)
    
    # Khôi phục phương thức gốc
    analyzer._analyze_single_error = original_method
    
    # Hiển thị phân bố loại lỗi
    error_types = df[df['error_type'] != '']['error_type'].value_counts()
    print("\nPhân bố các loại lỗi:")
    for error_type, count in error_types.items():
        print(f"  - {error_type}: {count} lỗi ({count/len(error_types):.2%})")
    
    # Hiển thị phân bố lỗi theo model
    print("\nPhân bố lỗi theo model:")
    for model in df['model_name'].unique():
        model_errors = df[(df['model_name'] == model) & (df['error_type'] != '')]
        if len(model_errors) > 0:
            print(f"  Model {model}:")
            for error_type, count in model_errors['error_type'].value_counts().items():
                print(f"    - {error_type}: {count} lỗi ({count/len(model_errors):.2%})")
    
    # Lưu DataFrame với thông tin phân tích lỗi
    csv_path = os.path.join(output_dir, f"error_analysis_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu kết quả phân tích lỗi vào: {csv_path}")
    
    # Tạo báo cáo và biểu đồ
    print("\nĐang tạo báo cáo và biểu đồ...")
    reporter = Reporting(results_df=df, output_dir=output_dir, timestamp=timestamp)
    report_files = reporter.generate_reports()
    
    print("\nĐã tạo các báo cáo:")
    for file_type, file_path in report_files.items():
        print(f"  - {file_type}: {file_path}")
    
    # Hiển thị đường dẫn tới biểu đồ phân tích lỗi
    plots_dir = os.path.join(output_dir, "plots")
    error_plots = [f for f in os.listdir(plots_dir) if "error_analysis" in f]
    
    print("\nCác biểu đồ phân tích lỗi:")
    for plot in error_plots:
        print(f"  - {os.path.join(plots_dir, plot)}")
    
    return df, output_dir

if __name__ == "__main__":
    df, output_dir = run_error_analysis_demo()
    print(f"\nHoàn tất! Kiểm tra thư mục {output_dir} để xem tất cả các báo cáo và biểu đồ phân tích lỗi.") 