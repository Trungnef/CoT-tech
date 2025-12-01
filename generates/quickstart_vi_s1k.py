#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Guide for Vi-S1K
Hướng dẫn nhanh để bắt đầu sử dụng hệ thống dịch Vi-S1K
"""

import os
import sys
from pathlib import Path
import io

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print_header("Kiểm tra Dependencies")
    
    required_packages = {
        "datasets": "Tải dataset từ Hugging Face",
        "transformers": "LLM models",
        "google.generativeai": "Google Gemini API (tùy chọn)",
        "openai": "OpenAI API (tùy chọn)",
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package:20s} - {description}")
        except ImportError:
            print(f"✗ {package:20s} - {description}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nCài đặt với:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ Tất cả dependencies đã cài đặt!")
    return True

def check_env_setup():
    """Kiểm tra cấu hình environment"""
    print_header("Kiểm tra Environment Setup")
    
    env_vars = {
        "GEMINI_API_KEY": "Google Gemini (khuyến nghị)",
        "OPENAI_API_KEY": "OpenAI (tùy chọn)",
        "HF_TOKEN": "Hugging Face (tùy chọn, để tải dataset)",
    }
    
    configured = []
    for var, description in env_vars.items():
        if os.getenv(var):
            print(f"✓ {var:20s} - {description}")
            configured.append(var)
        else:
            print(f"✗ {var:20s} - {description}")
    
    if not configured:
        print("\n⚠️  Chưa cấu hình API keys!")
        print("\nTạo file .env trong thư mục gốc:")
        print("""
# .env file
GEMINI_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
HF_TOKEN=your_token_here
        """)
        return False
    
    print(f"\n✓ Đã cấu hình {len(configured)}/3 API keys")
    return True

def quick_demo():
    """Chạy demo nhanh"""
    print_header("Demo Nhanh")
    
    print("Bắt đầu dịch 3 câu hỏi toán học sang tiếng Việt...\n")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "llm_evaluation"))
        from llm_evaluation.utils.translation_utils import get_translator
        
        translator = get_translator("gemini")
        
        questions = [
            "What is 2 + 2?",
            "How many sides does a triangle have?",
            "What is 10 multiplied by 5?"
        ]
        
        for q in questions:
            print(f"English: {q}")
            translated = translator.translate(q, "English", "Vietnamese", "mathematics")
            print(f"Vietnamese: {translated}\n")
        
        print("✓ Demo thành công!")
        return True
    
    except Exception as e:
        print(f"✗ Demo thất bại: {e}")
        return False

def show_basic_usage():
    """Hiển thị cách sử dụng cơ bản"""
    print_header("Cách Sử Dụng Cơ Bản")
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║           Cách 1: Sử dụng Command Line (Dễ nhất)         ║
╚═══════════════════════════════════════════════════════════╝

# Chạy test với 5 mẫu
python generates/vi_s1k_builder.py --translator gemini --test-run

# Dịch 100 câu hỏi
python generates/vi_s1k_builder.py --translator gemini --max-samples 100

# Dịch toàn bộ dataset
python generates/vi_s1k_builder.py --translator gemini


╔═══════════════════════════════════════════════════════════╗
║            Cách 2: Sử dụng Python Code                   ║
╚═══════════════════════════════════════════════════════════╝

from generates.s1k_translator import S1KDatasetLoader, VietnameseBenchmarkBuilder
from llm_evaluation.utils.translation_utils import get_translator

# Khởi tạo translator
translator = get_translator("gemini")

# Tải dataset
loader = S1KDatasetLoader()
dataset = [loader.extract_question_fields(item) 
           for item in loader.load_s1k(max_samples=100)]

# Xây dựng benchmark
builder = VietnameseBenchmarkBuilder(translator)
items = builder.build_benchmark(dataset)

# Lưu kết quả
builder.save_benchmark("results/vi_s1k.json", format="json")


╔═══════════════════════════════════════════════════════════╗
║            Cách 3: Xem Ví Dụ Chi Tiết                    ║
╚═══════════════════════════════════════════════════════════╝

python generates/examples_vi_s1k.py

    """)

def show_next_steps():
    """Hiển thị các bước tiếp theo"""
    print_header("Các Bước Tiếp Theo")
    
    print("""
1. Đọc tài liệu chi tiết:
   - generates/README_VI_S1K.md

2. Chạy các ví dụ:
   - python generates/examples_vi_s1k.py

3. Tích hợp với LLM Evaluator:
   - python llm_evaluation/main.py --questions-file results/vi_s1k/vi_s1k_benchmark.json

4. Tùy chỉnh theo nhu cầu:
   - Sửa translation prompts trong translation_utils.py
   - Thêm domain-specific logic trong s1k_translator.py
   - Tạo custom quality checker

5. Xử lý dữ liệu lớn:
   - Chia thành batches
   - Sử dụng cache hiệu quả
   - Chọn translator backend phù hợp

    """)

def interactive_mode():
    """Chế độ interactive"""
    print_header("Menu Interactive")
    
    while True:
        print("""
1. Chạy demo nhanh
2. Xem cách sử dụng
3. Xem ví dụ chi tiết
4. Kiểm tra setup lại
5. Thoát
        """)
        
        choice = input("Chọn (1-5): ").strip()
        
        if choice == "1":
            quick_demo()
        elif choice == "2":
            show_basic_usage()
        elif choice == "3":
            print("\nMở file: generates/examples_vi_s1k.py")
            print("Chạy: python generates/examples_vi_s1k.py")
        elif choice == "4":
            check_dependencies()
            check_env_setup()
        elif choice == "5":
            print("\nTạm biệt! 👋")
            break
        else:
            print("Lựa chọn không hợp lệ")

def main():
    """Main entry point"""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║            🌍 Vi-S1K Quick Start Guide 🚀                  ║
║                                                            ║
║   Vietnamese S1K Benchmark Translation System              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Kiểm tra dependencies
    deps_ok = check_dependencies()
    
    # Kiểm tra env setup
    env_ok = check_env_setup()
    
    if deps_ok and env_ok:
        # Chạy demo
        if quick_demo():
            # Hiển thị cách sử dụng
            show_basic_usage()
            
            # Hỏi user tiếp theo
            print("\nBạn muốn làm gì tiếp theo?")
            print("1. Chạy demo chi tiết (examples)")
            print("2. Dịch dataset (100 câu hỏi)")
            print("3. Xem tài liệu đầy đủ (README)")
            print("4. Thoát")
            
            choice = input("\nChọn (1-4): ").strip()
            
            if choice == "1":
                print("\nChạy: python generates/examples_vi_s1k.py")
                os.system("python generates/examples_vi_s1k.py")
            elif choice == "2":
                print("\nChạy: python generates/vi_s1k_builder.py --translator gemini --max-samples 100")
                os.system("python generates/vi_s1k_builder.py --translator gemini --max-samples 100")
            elif choice == "3":
                print("\nXem file: generates/README_VI_S1K.md")
            elif choice == "4":
                print("\nTạm biệt! 👋")
            else:
                interactive_mode()
    else:
        print("\n⚠️  Vui lòng cài đặt các dependencies cần thiết trước!")
        print("Xem hướng dẫn cài đặt trong generates/README_VI_S1K.md")


if __name__ == "__main__":
    main()
