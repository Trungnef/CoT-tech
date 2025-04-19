"""
Kiểm tra các hàm tính metrics trong metrics_utils.py
"""

import sys
import os
import pandas as pd
from utils.metrics_utils import calculate_exact_match_accuracy, calculate_rouge_scores

def test_exact_match():
    print("===== TEST EXACT MATCH =====")
    # Test cases cho exact match
    predictions = [
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là TP. HCM.",
        "Thủ đô của việt nam là hà nội",
        "Thủ đô VN: Hà Nội!",
        "Không biết thủ đô của Việt Nam",
        None
    ]
    
    references = [
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội."
    ]
    
    # Tính và in kết quả
    print("\nChế độ mặc định (normalize=True, case_sensitive=False, remove_punctuation=True):")
    em = calculate_exact_match_accuracy(predictions, references)
    print(f"Exact match accuracy: {em}")
    
    print("\nChế độ case_sensitive=True:")
    em = calculate_exact_match_accuracy(predictions, references, case_sensitive=True)
    print(f"Exact match accuracy: {em}")
    
    print("\nChế độ remove_punctuation=False:")
    em = calculate_exact_match_accuracy(predictions, references, remove_punctuation=False)
    print(f"Exact match accuracy: {em}")
    
    print("\nChế độ relaxed_match=False:")
    em = calculate_exact_match_accuracy(predictions, references, relaxed_match=False)
    print(f"Exact match accuracy: {em}")
    
    # Test với dữ liệu gần với thực tế
    print("\n===== TEST EXACT MATCH VỚI DỮ LIỆU THỰC TẾ =====")
    real_predictions = [
        "Số kẹo người thứ nhất là 17 viên.",
        "Người thứ nhất nhận được 17 viên kẹo.",
        "Người thứ 1 được 17 viên, người thứ 2 được 34 viên...",
        "Số kẹo là x=17, 2x=34, 4x=68, tổng = 17+34+68=119 (thiếu 2 viên)",
        "Đáp án: x = 17"
    ]
    
    real_references = [
        "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên.",
        "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên.",
        "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên.",
        "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên.",
        "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
    ]
    
    print("\nDữ liệu thực tế với relaxed_match=True (mặc định):")
    em = calculate_exact_match_accuracy(real_predictions, real_references)
    print(f"Exact match accuracy: {em}")
    
    print("\nDữ liệu thực tế với relaxed_match=False (nghiêm ngặt):")
    em = calculate_exact_match_accuracy(real_predictions, real_references, relaxed_match=False)
    print(f"Exact match accuracy: {em}")

def test_rouge():
    print("\n===== TEST ROUGE SCORES =====")
    # Test cases cho ROUGE
    predictions = [
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là thành phố Hà Nội.",
        "Hà Nội là thủ đô của Việt Nam từ lâu đời.",
        "Không biết thủ đô của Việt Nam",
        None
    ]
    
    references = [
        "Thủ đô của Việt Nam là Hà Nội.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Thủ đô lâu đời của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội.",
        "Thủ đô của Việt Nam là Hà Nội."
    ]
    
    # Tính và in kết quả
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # In kết quả
    print("ROUGE scores:")
    for metric, value in rouge_scores.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    # Thực thi các bài test
    test_exact_match()
    test_rouge() 