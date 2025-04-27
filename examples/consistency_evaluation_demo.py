#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script hiển thị cách sử dụng hàm đánh giá tính nhất quán (consistency)
trong llm_evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"Đã thêm {parent_dir} vào sys.path")

# Hiển thị sys.path để debug
print(f"sys.path: {sys.path}")

# Import các module cần thiết
try:
    from llm_evaluation.utils.metrics_utils import calculate_consistency_metrics
    from llm_evaluation.core.result_analyzer import ResultAnalyzer
    from llm_evaluation.core.reporting import Reporting
except ImportError as e:
    print(f"Lỗi khi import các module cần thiết: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

def create_sample_data():
    """Tạo dữ liệu mẫu để minh họa"""
    # Tạo dữ liệu mẫu
    data = []
    models = ["llama", "qwen", "gemini"]
    question_types = ["math", "reasoning", "factual"]
    question_ids = [f"q{i}" for i in range(1, 11)]
    
    # Cột prompt_type chứa cả normal prompts và self-consistency prompts
    prompt_types = [
        "zero_shot", "few_shot", "cot", 
        "cot_self_consistency_3", "cot_self_consistency_5", "cot_self_consistency_7"
    ]
    
    # Tạo câu trả lời có độ nhất quán khác nhau cho các model khác nhau
    # Mỗi mô hình sẽ có độ nhất quán khác nhau
    for model in models:
        for q_id in question_ids:
            for q_type in question_types:
                # Tạo câu hỏi
                question_text = f"Câu hỏi {q_type} {q_id}"
                
                # Tạo câu trả lời cho các prompt thông thường
                for pt in prompt_types[:3]:  # zero_shot, few_shot, cot
                    # Mức độ chính xác tùy thuộc vào model
                    is_correct = np.random.random() < {
                        "llama": 0.95,
                        "qwen": 0.98,
                        "gemini": 0.93
                    }[model]
                    
                    data.append({
                        "model_name": model,
                        "question_id": q_id,
                        "question_type": q_type,
                        "question_text": question_text,
                        "prompt_type": pt,
                        "response": f"Câu trả lời của {model} cho {q_id} sử dụng {pt}",
                        "is_correct": is_correct,
                        "latency": np.random.uniform(0.5, 3.0)
                    })
                
                # Tạo câu trả lời cho các prompt self-consistency
                for pt in prompt_types[3:]:  # cot_self_consistency_*
                    # Số lần chạy dựa trên prompt type
                    runs = int(pt.split("_")[-1])
                    
                    # Mức độ nhất quán tùy thuộc vào model
                    consistency_level = {
                        "llama": 0.9,  # 90% consistency
                        "qwen": 0.8,   # 80% consistency
                        "gemini": 0.7  # 70% consistency
                    }[model]
                    
                    # Tạo các câu trả lời cho mỗi lần chạy
                    for run in range(runs):
                        # Quyết định xem câu trả lời này có nhất quán với câu trả lời phổ biến nhất không
                        is_consistent = np.random.random() < consistency_level
                        
                        # Tạo câu trả lời khác nhau tùy thuộc vào tính nhất quán
                        if is_consistent:
                            answer = f"Đáp án {q_id} là A"
                            final_answer = "A"
                        else:
                            # Chọn một câu trả lời khác ngẫu nhiên
                            other_answer = np.random.choice(["B", "C", "D"])
                            answer = f"Đáp án {q_id} là {other_answer}"
                            final_answer = other_answer
                        
                        # Đánh dấu là đúng nếu là A (để minh họa)
                        is_correct = (final_answer == "A")
                        
                        data.append({
                            "model_name": model,
                            "question_id": q_id,
                            "question_type": q_type,
                            "question_text": question_text,
                            "prompt_type": pt,
                            "response": f"{answer} (run {run+1})",
                            "final_answer": final_answer,
                            "is_correct": is_correct,
                            "latency": np.random.uniform(0.5, 3.0)
                        })
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    return df

def run_consistency_evaluation():
    """Thực hiện đánh giá tính nhất quán và tạo báo cáo"""
    print("Tạo dữ liệu mẫu...")
    df = create_sample_data()
    
    print(f"Đã tạo DataFrame với {len(df)} dòng.")
    print(f"Các model: {df['model_name'].unique()}")
    print(f"Các loại prompt: {df['prompt_type'].unique()}")
    
    # Khởi tạo ResultAnalyzer
    print("\nĐánh giá tính nhất quán...")
    analyzer = ResultAnalyzer(results_df=df, verbose=True)
    
    # Đánh giá tính nhất quán
    df = analyzer.evaluate_consistency(df)
    
    # Hiển thị một số thống kê về tính nhất quán
    consistency_rows = df[~df['consistency_score'].isna()]
    print(f"\nĐã tính toán tính nhất quán cho {len(consistency_rows)} dòng.")
    
    # Hiển thị điểm nhất quán trung bình theo model
    consistency_by_model = consistency_rows.groupby('model_name')['consistency_score'].mean()
    print("\nĐiểm nhất quán trung bình theo model:")
    for model, score in consistency_by_model.items():
        print(f"  - {model}: {score:.4f}")
    
    # Hiển thị agreement rate trung bình theo model
    agreement_by_model = consistency_rows.groupby('model_name')['consistency_agreement_rate'].mean()
    print("\nAgreement rate trung bình theo model:")
    for model, rate in agreement_by_model.items():
        print(f"  - {model}: {rate:.4f}")
    
    # Tính toán accuracy cho mỗi model và hiển thị
    accuracy_by_model = df.groupby('model_name')['is_correct'].mean()
    print("\nAccuracy theo model:")
    for model, acc in accuracy_by_model.items():
        print(f"  - {model}: {acc:.4f}")
    
    # Lưu DataFrame với thông tin consistency
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_dir, "examples", "outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"consistency_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu kết quả vào: {csv_path}")
    
    # Tạo báo cáo sử dụng Reporting
    print("\nTạo báo cáo và biểu đồ...")
    reporter = Reporting(results_df=df, output_dir=output_dir, timestamp=timestamp)
    report_files = reporter.generate_reports()
    
    print("\nĐã tạo các báo cáo:")
    for file_type, file_path in report_files.items():
        print(f"  - {file_type}: {file_path}")
    
    # Hiển thị đường dẫn tới biểu đồ consistency
    plots_dir = os.path.join(output_dir, "plots")
    consistency_plots = [f for f in os.listdir(plots_dir) if "consistency" in f]
    
    print("\nCác biểu đồ về consistency:")
    for plot in consistency_plots:
        print(f"  - {os.path.join(plots_dir, plot)}")
    
    return df, output_dir

if __name__ == "__main__":
    df, output_dir = run_consistency_evaluation()
    print(f"\nHoàn tất! Kiểm tra thư mục {output_dir} để xem tất cả các báo cáo và biểu đồ.") 