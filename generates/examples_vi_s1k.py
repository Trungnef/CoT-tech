#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example scripts showing different use cases of the Vi-S1K translator
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))  # generates folder
sys.path.insert(0, str(Path(__file__).parent.parent))  # root folder
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_evaluation"))  # llm_evaluation folder

from s1k_translator import (
    S1KDatasetLoader,
    VietnameseBenchmarkBuilder,
    QualityChecker,
    TranslatedQuestion
)
from llm_evaluation.utils.translation_utils import get_translator
import json


def example_1_basic_usage():
    """
    Example 1: Dịch toàn bộ dataset S1K sang tiếng Việt
    Sử dụng Gemini với 10 mẫu test
    """
    print("=" * 60)
    print("Example 1: Basic Vi-S1K Translation")
    print("=" * 60)
    
    # Khởi tạo translator
    translator = get_translator("gemini")
    
    # Tải dataset từ Hugging Face
    loader = S1KDatasetLoader()
    raw_data = loader.load_s1k(max_samples=10)
    dataset = [loader.extract_question_fields(item) for item in raw_data]
    
    # Xây dựng benchmark
    builder = VietnameseBenchmarkBuilder(translator)
    translated_items = builder.build_benchmark(
        dataset,
        quality_checker=QualityChecker.simple_check,
        show_progress=True
    )
    
    # Lưu kết quả
    output_dir = "./results/vi_s1k_example1"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    builder.save_benchmark(f"{output_dir}/vi_s1k.json", format="json")
    
    # Hiển thị một vài ví dụ
    print("\nSample translations:")
    for item in translated_items[:2]:
        print(f"\nOriginal: {item.original_question}")
        print(f"Vietnamese: {item.vietnamese_question}")
        print(f"Quality Score: {item.quality_score:.2f}")


def example_2_streaming_translation():
    """
    Example 2: Dịch từng câu một (streaming)
    Hữu ích khi xử lý live data
    """
    print("\n" + "=" * 60)
    print("Example 2: Streaming Translation")
    print("=" * 60)
    
    translator = get_translator("gemini")
    
    # Danh sách câu hỏi test
    questions = [
        "What is the sum of 5 and 3?",
        "How many sides does a triangle have?",
        "What is 10 minus 4?"
    ]
    
    print("\nTranslating questions one by one:\n")
    for q in questions:
        translated = translator.translate(q, "English", "Vietnamese", "mathematics")
        print(f"English: {q}")
        print(f"Vietnamese: {translated}\n")


def example_3_custom_quality_check():
    """
    Example 3: Sử dụng quality checker tùy chỉnh
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Quality Checker")
    print("=" * 60)
    
    def custom_quality_checker(translated_item: TranslatedQuestion) -> float:
        """Tùy chỉnh kiểm tra chất lượng"""
        score = 0.0
        
        # Kiểm tra câu hỏi tiếng Việt không rỗng
        if translated_item.vietnamese_question:
            score += 0.3
        
        # Kiểm tra có dịch câu trả lời
        if translated_item.vietnamese_answer:
            score += 0.3
        
        # Kiểm tra độ dài hợp lý
        if len(translated_item.vietnamese_question) > 5:
            score += 0.4
        
        return score
    
    translator = get_translator("gemini")
    
    loader = S1KDatasetLoader()
    raw_data = loader.load_s1k(max_samples=5)
    dataset = [loader.extract_question_fields(item) for item in raw_data]
    
    builder = VietnameseBenchmarkBuilder(translator)
    translated_items = builder.build_benchmark(
        dataset,
        quality_checker=custom_quality_checker,
        show_progress=True
    )
    
    print("\nCustom Quality Scores:")
    for item in translated_items:
        print(f"ID: {item.id}, Score: {item.quality_score:.2f}")


def example_4_multilingual():
    """
    Example 4: Dịch sang nhiều ngôn ngữ khác nhau
    """
    print("\n" + "=" * 60)
    print("Example 4: Multilingual Translation")
    print("=" * 60)
    
    translator = get_translator("gemini")
    
    question = "What is the area of a square with side length 5?"
    
    languages = [
        ("English", "Vietnamese"),
        ("English", "French"),
        ("English", "Spanish"),
        ("English", "Japanese"),
    ]
    
    print(f"\nOriginal question: {question}\n")
    
    for source_lang, target_lang in languages:
        translated = translator.translate(
            question, 
            source_lang=source_lang,
            target_lang=target_lang,
            domain="mathematics"
        )
        print(f"{target_lang}: {translated}")


def example_5_batch_processing():
    """
    Example 5: Xử lý batch lớn với tối ưu hóa bộ nhớ
    """
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing with Memory Optimization")
    print("=" * 60)
    
    translator = get_translator("gemini")
    
    loader = S1KDatasetLoader()
    raw_data = loader.load_s1k(max_samples=100)
    
    # Xử lý từng batch
    batch_size = 10
    all_translated = []
    
    for batch_start in range(0, len(raw_data), batch_size):
        batch_end = min(batch_start + batch_size, len(raw_data))
        batch = raw_data[batch_start:batch_end]
        dataset = [loader.extract_question_fields(item) for item in batch]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} "
              f"({batch_start}-{batch_end})")
        
        builder = VietnameseBenchmarkBuilder(translator)
        translated_items = builder.build_benchmark(
            dataset,
            quality_checker=QualityChecker.simple_check,
            show_progress=False  # Tắt progress bar để output sạch
        )
        
        all_translated.extend(translated_items)
        
        # Lưu batch
        output_file = f"./results/batch_{batch_start//batch_size + 1}.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        builder.save_benchmark(output_file, format="json")
    
    print(f"\nTotal items translated: {len(all_translated)}")
    stats = translator.get_stats()
    print(f"Translator stats: {stats}")


def example_6_caching_efficiency():
    """
    Example 6: Kiểm tra hiệu quả của caching
    """
    print("\n" + "=" * 60)
    print("Example 6: Caching Efficiency Demonstration")
    print("=" * 60)
    
    # Sử dụng cache
    translator_with_cache = get_translator("gemini", use_cache=True)
    
    # Dịch cùng một danh sách câu hỏi
    questions = [
        "What is 2 + 2?",
        "What is 5 * 3?",
        "What is 2 + 2?",  # Duplicate
        "What is 5 * 3?",  # Duplicate
    ]
    
    print(f"\nTranslating {len(questions)} questions...")
    print("(2 duplicates to test cache)\n")
    
    for q in questions:
        translator_with_cache.translate(q, "English", "Vietnamese", "mathematics")
    
    stats = translator_with_cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total translations attempted: {stats['total_translations']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"\nThis means {stats['cache_hits']} API calls were saved!")


def example_7_error_handling():
    """
    Example 7: Xử lý lỗi và retry logic
    """
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)
    
    translator = get_translator("gemini")
    
    # Test cases với các loại input khác nhau
    test_cases = [
        ("Normal question", "What is 2 + 2?"),
        ("Empty string", ""),
        ("Very long text", "This is a very long question " * 50),
        ("Special characters", "What is 10² + 5³?"),
    ]
    
    print("\nTesting various input cases:\n")
    
    for test_name, question in test_cases:
        try:
            translated = translator.translate(
                question,
                "English",
                "Vietnamese",
                "mathematics"
            )
            status = "✓ Success" if translated else "✗ Empty result"
            print(f"{test_name}: {status}")
            if translated and len(translated) < 100:
                print(f"  → {translated}")
        except Exception as e:
            print(f"{test_name}: ✗ Error - {str(e)[:50]}")


def example_8_integration_with_evaluator():
    """
    Example 8: Tích hợp Vi-S1K với LLM Evaluator của project
    """
    print("\n" + "=" * 60)
    print("Example 8: Integration with LLM Evaluator")
    print("=" * 60)
    
    print("""
    Để tích hợp Vi-S1K với evaluator của project:
    
    1. Dịch dataset:
       python generates/vi_s1k_builder.py --translator gemini --max-samples 100
    
    2. Sử dụng Vi-S1K trong evaluator:
       python llm_evaluation/main.py \\
           --questions-file results/vi_s1k/vi_s1k_benchmark.json \\
           --models gemini llama \\
           --prompts zero_shot few_shot_3
    
    3. Kết quả sẽ được lưu trong results/
    """)


def main():
    """Chạy các ví dụ"""
    examples = {
        "1": ("Basic Usage", example_1_basic_usage),
        "2": ("Streaming Translation", example_2_streaming_translation),
        "3": ("Custom Quality Check", example_3_custom_quality_check),
        "4": ("Multilingual", example_4_multilingual),
        "5": ("Batch Processing", example_5_batch_processing),
        "6": ("Caching Efficiency", example_6_caching_efficiency),
        "7": ("Error Handling", example_7_error_handling),
        "8": ("Integration with Evaluator", example_8_integration_with_evaluator),
    }
    
    print("""
╔════════════════════════════════════════════════════════════╗
║         Vi-S1K Translation System - Example Scripts        ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print(f"  0. Run all examples")
    print()
    
    choice = input("Select example (0-8) or press Enter for basic demo: ").strip()
    
    if choice == "0":
        for key, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n✗ Example {key} failed: {e}")
    elif choice in examples:
        examples[choice][1]()
    else:
        example_2_streaming_translation()


if __name__ == "__main__":
    main()
