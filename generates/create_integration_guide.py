"""
Integration Guide: Vi-S1K with LLM Evaluator
Hướng dẫn tích hợp Vi-S1K với LLM Evaluator hiện tại
"""

import json
from pathlib import Path


def create_integration_guide():
    """Tạo hướng dẫn tích hợp"""
    
    guide = """
# Integration Guide: Vi-S1K with LLM Evaluator

## Tổng Quan (Overview)

Hướng dẫn này giải thích cách sử dụng Vi-S1K (Vietnamese S1K Benchmark) 
với hệ thống LLM Evaluator hiện tại của CoT-tech.

## Kiến Trúc (Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    S1K Dataset (Hugging Face)               │
│                    (simplescaling/s1K-1.1)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   Vi-S1K Translation System      │
        │  (generates/vi_s1k_translator.py)│
        │                                  │
        │  • Dịch sang tiếng Việt         │
        │  • Kiểm chất lượng               │
        │  • Xuất nhiều định dạng          │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   Vi-S1K Benchmark (JSON)       │
        │  results/vi_s1k/                │
        │  vi_s1k_benchmark.json          │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │   LLM Evaluator                 │
        │  (llm_evaluation/main.py)       │
        │                                  │
        │  • Evaluate models              │
        │  • Generate reports             │
        │  • Analyze results              │
        └─────────────────────────────────┘
```

## Bước 1: Xây Dựng Vi-S1K Benchmark

### 1.1 Cài đặt (Setup)

```bash
# Cài đặt dependencies
pip install datasets google-generativeai transformers pandas

# Tạo file .env
# .env
GEMINI_API_KEY=your_key_here
```

### 1.2 Tạo Vi-S1K Benchmark

```bash
# Option A: Dịch dataset đầy đủ
python generates/vi_s1k_builder.py \\
    --translator gemini \\
    --output-dir ./results/vi_s1k

# Option B: Test với 5 mẫu trước
python generates/vi_s1k_builder.py \\
    --translator gemini \\
    --test-run

# Option C: Dịch 100 mẫu
python generates/vi_s1k_builder.py \\
    --translator gemini \\
    --max-samples 100
```

Output sẽ tạo ra:
```
results/vi_s1k/
├── vi_s1k_benchmark.json    # Main benchmark file
├── vi_s1k_benchmark.jsonl   # JSONL format
├── vi_s1k_benchmark.csv     # CSV format
└── statistics.json          # Translation statistics
```

### 1.3 Kiểm Tra Output

```python
import json

with open("results/vi_s1k/vi_s1k_benchmark.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items: {len(data['items'])}")
print(f"First item:")
print(json.dumps(data['items'][0], ensure_ascii=False, indent=2))
```

## Bước 2: Sử Dụng Vi-S1K với LLM Evaluator

### 2.1 Chạy Evaluation

```bash
# Sử dụng Vi-S1K benchmark
python llm_evaluation/main.py \\
    --questions-file results/vi_s1k/vi_s1k_benchmark.json \\
    --models gemini llama \\
    --prompts zero_shot few_shot_3 cot_self_consistency_3 \\
    --max-questions 50 \\
    --results-dir results/vi_s1k_evaluation
```

### 2.2 Các Tùy Chọn Khác

```bash
# Chỉ test với 2 mẫu
python llm_evaluation/main.py \\
    --questions-file results/vi_s1k/vi_s1k_benchmark.json \\
    --models gemini \\
    --prompts zero_shot \\
    --max-questions 2 \\
    --test-run

# Evaluate lại từ checkpoint trước đó
python llm_evaluation/main.py \\
    --questions-file results/vi_s1k/vi_s1k_benchmark.json \\
    --models gemini \\
    --prompts zero_shot \\
    --resume
```

### 2.3 Xem Kết Quả

```bash
# HTML report sẽ được tạo tại:
results/vi_s1k_evaluation/reports/evaluation_report_*.html

# Raw results:
results/vi_s1k_evaluation/raw_results/evaluation_results_*.csv
results/vi_s1k_evaluation/raw_results/evaluation_results_*.json
```

## Bước 3: Python API Integration

### 3.1 Sử Dụng Direct Python

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".") / "llm_evaluation"))

from core.evaluator import Evaluator
from utils.config_utils import EvaluationConfig

# Tạo config
config = EvaluationConfig(
    models=["gemini", "llama"],
    prompt_types=["zero_shot", "few_shot_3", "cot_self_consistency_3"],
    questions_file="results/vi_s1k/vi_s1k_benchmark.json",
    results_dir="results/vi_s1k_evaluation",
    max_questions=100,
    batch_size=5
)

# Chạy evaluation
evaluator = Evaluator(config)
evaluator.run()

# Xem results
results = evaluator.get_results()
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Average latency: {results['latency']:.2f}ms")
```

### 3.2 Xử Lý Custom Data Format

Nếu bạn có format dữ liệu khác, convert sang format của evaluator:

```python
import json

def convert_to_evaluator_format(vi_s1k_file, output_file):
    """Convert Vi-S1K format to LLM Evaluator format"""
    
    with open(vi_s1k_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data['items']:
        question = {
            "id": item['id'],
            "question": item['vietnamese_question'],  # Sử dụng câu hỏi tiếng Việt
            "correct_answer": item.get('vietnamese_answer', ''),
            "category": item.get('domain', 'mathematics'),
            "difficulty": item.get('difficulty', 'medium'),
            "task_type": "reasoning",  # hoặc "math", "logic", etc.
            "examples": item.get('metadata', {}).get('examples', [])
        }
        questions.append(question)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

# Sử dụng
convert_to_evaluator_format(
    "results/vi_s1k/vi_s1k_benchmark.json",
    "results/vi_s1k_evaluator_format.json"
)

# Sau đó dùng với evaluator
# python llm_evaluation/main.py --questions-file results/vi_s1k_evaluator_format.json
```

## Bước 4: Advanced Usage

### 4.1 Comparing Translations Quality

```python
from generates.s1k_translator import QualityChecker
import json

# Load benchmark
with open("results/vi_s1k/vi_s1k_benchmark.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Analyze quality scores
items = data['items']
scores = [item['quality_score'] for item in items]

print(f"Quality statistics:")
print(f"  Min: {min(scores):.2f}")
print(f"  Max: {max(scores):.2f}")
print(f"  Average: {sum(scores)/len(scores):.2f}")
print(f"  Median: {sorted(scores)[len(scores)//2]:.2f}")

# Find low-quality translations
low_quality = [item for item in items if item['quality_score'] < 0.5]
print(f"\nLow-quality translations: {len(low_quality)}")
for item in low_quality[:3]:
    print(f"  - {item['id']}: {item['quality_score']:.2f}")
```

### 4.2 Analyzing Evaluation Results

```python
import pandas as pd
import json

# Load evaluation results
df = pd.read_csv("results/vi_s1k_evaluation/raw_results/evaluation_results_*.csv")

# Group by model
print("Results by Model:")
print(df.groupby('model')[['accuracy', 'latency']].mean())

# Group by prompt type
print("\nResults by Prompt Type:")
print(df.groupby('prompt_type')[['accuracy', 'latency']].mean())

# Find best model-prompt combination
best = df.loc[df['accuracy'].idxmax()]
print(f"\nBest combination:")
print(f"  Model: {best['model']}")
print(f"  Prompt: {best['prompt_type']}")
print(f"  Accuracy: {best['accuracy']:.2%}")
```

### 4.3 Batch Processing with Resource Management

```python
from generates.s1k_translator import S1KDatasetLoader, VietnameseBenchmarkBuilder
from llm_evaluation.utils.translation_utils import get_translator
import time

# Translate in batches to manage resources
loader = S1KDatasetLoader()
all_data = loader.load_s1k()

batch_size = 500
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    dataset = [loader.extract_question_fields(item) for item in batch]
    
    translator = get_translator("gemini")
    builder = VietnameseBenchmarkBuilder(translator)
    
    items = builder.build_benchmark(dataset)
    
    # Save each batch
    output_file = f"results/vi_s1k/batch_{i//batch_size}.json"
    builder.save_benchmark(output_file, format="json")
    
    print(f"Processed batch {i//batch_size + 1}")
    time.sleep(5)  # Rate limiting
```

## Comparison: Original S1K vs Vi-S1K

| Aspect | Original S1K | Vi-S1K |
|--------|-------------|--------|
| Language | English | Vietnamese |
| Domain | Elementary Math | Elementary Math |
| Format | standardized | Same as S1K |
| Quality | High | High (translated + validated) |
| Use Case | English learners | Vietnamese learners |
| Integration | Direct | Via LLM Evaluator |

## Troubleshooting

### Issue: "questions_file not found"
```bash
# Solution: Kiểm tra đường dẫn đúng
python llm_evaluation/main.py \\
    --questions-file $(realpath results/vi_s1k/vi_s1k_benchmark.json) \\
    --models gemini
```

### Issue: "Translation too slow"
```python
# Solution: Sử dụng cache hiệu quả
# Cache được lưu tự động ở ./translation_cache
# Chạy lần 2 sẽ nhanh hơn

# Hoặc giảm số mẫu
python generates/vi_s1k_builder.py --max-samples 100
```

### Issue: "Out of memory"
```bash
# Solution: Xử lý batch nhỏ hơn
python llm_evaluation/main.py \\
    --questions-file results/vi_s1k/vi_s1k_benchmark.json \\
    --batch-size 2
```

## Best Practices

1. **Caching**: Luôn bật cache khi dịch để tiết kiệm chi phí API
2. **Testing**: Test với --test-run hoặc small batches trước khi chạy full
3. **Quality**: Kiểm tra quality scores để đảm bảo chất lượng bản dịch
4. **Monitoring**: Theo dõi translation statistics và evaluation metrics
5. **Versioning**: Lưu metadata để biết phiên bản translator nào được dùng

## Performance Tips

- **Translation**: Sử dụng Gemini API (balance cost/quality)
- **Evaluation**: Sử dụng local models cho phía evaluator (nếu có GPU)
- **Caching**: Tối tối ưu hóa cache hit rate > 50%
- **Batching**: Dùng batch size 5-10 cho balanced speed/quality

## References

- Vi-S1K Documentation: generates/README_VI_S1K.md
- LLM Evaluator: llm_evaluation/README.md
- S1K Dataset: https://huggingface.co/datasets/simplescaling/s1K-1.1
- Examples: generates/examples_vi_s1k.py

## Tài Liệu (Documentation)

- **Vi-S1K README**: generates/README_VI_S1K.md
- **Setup Guide**: generates/setup_vi_s1k.py
- **Examples**: generates/examples_vi_s1k.py
- **Quick Start**: generates/quickstart_vi_s1k.py

---

**Chúc bạn thành công! 🚀**
"""
    
    return guide


def save_integration_guide():
    """Lưu hướng dẫn tích hợp"""
    
    guide = create_integration_guide()
    
    output_path = Path("generates/INTEGRATION_GUIDE.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"Integration guide saved to: {output_path}")
    return output_path


def create_example_config():
    """Tạo example configuration file"""
    
    example_config = {
        "vi_s1k_translation": {
            "description": "Configuration for Vi-S1K translation",
            "translator": {
                "backend": "gemini",
                "use_cache": True,
                "cache_dir": "./translation_cache"
            },
            "dataset": {
                "s1k_config": "default",
                "max_samples": 100,
                "split": "train"
            },
            "output": {
                "output_dir": "./results/vi_s1k",
                "formats": ["json", "jsonl", "csv"]
            }
        },
        "llm_evaluation": {
            "description": "Configuration for LLM evaluation on Vi-S1K",
            "models": ["gemini", "llama"],
            "prompts": ["zero_shot", "few_shot_3", "cot_self_consistency_3"],
            "questions_file": "results/vi_s1k/vi_s1k_benchmark.json",
            "results_dir": "results/vi_s1k_evaluation",
            "max_questions": 100,
            "batch_size": 5,
            "checkpoint_frequency": 10
        }
    }
    
    output_path = Path("generates/vi_s1k_config_example.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, ensure_ascii=False, indent=2)
    
    print(f"Example config saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Creating integration guide...")
    save_integration_guide()
    print("Creating example config...")
    create_example_config()
    print("\nDone! ✓")
