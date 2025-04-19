"""
Kiểm tra chi tiết cho Exact Match trên dữ liệu thực tế
"""

import sys
import os
import pandas as pd
from utils.metrics_utils import calculate_exact_match_accuracy

def test_examples():
    print("===== TEST EXACT MATCH VỚI CÁC VÍ DỤ CỤ THỂ =====")
    
    examples = [
        {
            "id": 1,
            "prediction": "Theo đề bài, tổng số viên kẹo là 121 viên. Gọi số kẹo người thứ nhất là x.\nTa có: x + 2x + 4x = 121\n=> 7x = 121\n=> x = 121/7 = 17.28 (không hợp lý vì số kẹo phải là số nguyên)\nVậy x = 17 viên, 2x = 34 viên, 4x = 68 viên.\nTổng số kẹo là: 17 + 34 + 68 = 119 viên, sai với đề (121 viên).\n\nKết luận: Người thứ nhất được 17 viên kẹo, người thứ hai được 34 viên kẹo, người thứ ba được 68 viên kẹo.",
            "reference": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
        },
        {
            "id": 2,
            "prediction": "Để giải bài toán này, ta gọi x là số viên kẹo mà người thứ nhất nhận được.\nTheo đề bài:\n- Người thứ hai được gấp 2 lần người thứ nhất, tức là 2x viên\n- Người thứ ba được gấp 2 lần người thứ hai, tức là 2(2x) = 4x viên\n- Người thứ tư = ?\n\nTổng số viên kẹo = 121 viên\n=> x + 2x + 4x + ? = 121\nĐề bài chỉ nhắc đến 3 người, nên:\n=> x + 2x + 4x = 121\n=> 7x = 121\n=> x = 17.28...\n\nVì số kẹo phải là số nguyên nên x = 17\nNgười thứ nhất: 17 viên\nNgười thứ hai: 2×17 = 34 viên\nNgười thứ ba: 4×17 = 68 viên\nTổng: 17 + 34 + 68 = 119 viên (khác 121 viên trong đề bài)",
            "reference": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
        },
        {
            "id": 3,
            "prediction": "Số kẹo người thứ nhất là 17 viên kẹo.",
            "reference": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
        },
        {
            "id": 4,
            "prediction": "121/7 = 17.28, làm tròn xuống ta có x = 17.",
            "reference": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
        },
        {
            "id": 5,
            "prediction": "x = 16, 2x = 32, 4x = 64; Tổng: 16 + 32 + 64 = 112 viên kẹo.",
            "reference": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 2^2x = 121. Giải ra: x = 17 viên."
        }
    ]
    
    # Test từng ví dụ
    for i, example in enumerate(examples, 1):
        prediction = example["prediction"]
        reference = example["reference"]
        
        print(f"\nVÍ DỤ {i}:")
        print(f"Dự đoán: {prediction[:50]}...")
        print(f"Tham chiếu: {reference}")
        
        # Kiểm tra với relaxed_match=True
        em_relaxed = calculate_exact_match_accuracy([prediction], [reference], relaxed_match=True)
        print(f"Exact match (relaxed): {em_relaxed}")
        
        # Kiểm tra với relaxed_match=False
        em_strict = calculate_exact_match_accuracy([prediction], [reference], relaxed_match=False)
        print(f"Exact match (strict): {em_strict}")

if __name__ == "__main__":
    test_examples() 