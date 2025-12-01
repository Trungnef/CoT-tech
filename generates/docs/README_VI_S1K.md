# Vi-S1K: Vietnamese S1K Benchmark Builder

Hệ thống dịch và xây dựng **Vi-S1K** - một benchmark tiếng Việt chất lượng cao cho bài toán tiểu học, dựa trên dataset S1K từ Hugging Face.

## 🎯 Mục tiêu

Xây dựng một bộ dữ liệu toán tiểu học tiếng Việt chất lượng cao bằng cách:
- Dịch dataset S1K (simplescaling/s1K-1.1) từ tiếng Anh sang tiếng Việt
- Sử dụng các LLM mạnh (Gemini, OpenAI, Local models) để dịch
- Kiểm chất lượng bản dịch tự động
- Hỗ trợ caching để tối ưu hóa chi phí API
- Xuất kết quả ở nhiều định dạng (JSON, JSONL, CSV)

## 📋 Tính năng chính

### 1. **Hỗ trợ nhiều LLM backend**
- **Gemini (Google)**: Giải pháp tốt với chi phí hợp lý
- **OpenAI**: Chất lượng cao (GPT-3.5, GPT-4)
- **Local Models**: Llama, Qwen - chạy offline

### 2. **Translation Cache**
- Tránh dịch lại cùng một đoạn text
- Lưu cache tự động trên đĩa
- Giảm chi phí API tới 90% khi xử lý dữ liệu lớn

### 3. **Domain-Specific Translation**
- Xây dựng prompt dịch riêng cho toán học
- Bảo toàn các ký hiệu toán học và con số
- Giữ ngữ điệu và ý nghĩa chính xác

### 4. **Quality Control**
- Kiểm tra chất lượng bản dịch tự động
- Tính điểm chất lượng dựa trên:
  - Độ dài (length ratio check)
  - Sự có mặt của ký tự tiếng Việt
  - LLM evaluation (tùy chọn)

### 5. **Multiple Output Formats**
- **JSON**: Định dạng chi tiết với metadata
- **JSONL**: Một JSON mỗi dòng, tiết kiệm bộ nhớ
- **CSV**: Dễ phân tích trong Excel/Pandas

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
# Cài dependencies cơ bản
pip install datasets transformers google-generativeai openai pandas tqdm

# Hoặc cài từ requirements.txt
pip install -r requirements.txt
```

### 2. Cấu hình API keys

Tạo file `.env` trong thư mục dự án:

```bash
# Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face (để tải dataset)
HF_TOKEN=your_huggingface_token_here
```

## 💻 Sử dụng

### Cách 1: Sử dụng Gemini (Khuyến nghị cho người mới)

```bash
# Chạy với 100 mẫu
python generates/vi_s1k_builder.py \
    --translator gemini \
    --max-samples 100 \
    --output-dir ./results/vi_s1k

# Chạy test với 5 mẫu
python generates/vi_s1k_builder.py \
    --translator gemini \
    --test-run \
    --output-dir ./results/vi_s1k_test
```

### Cách 2: Sử dụng OpenAI

```bash
python generates/vi_s1k_builder.py \
    --translator openai \
    --max-samples 100 \
    --output-dir ./results/vi_s1k
```

### Cách 3: Sử dụng Local Model (Llama/Qwen)

```bash
# Đầu tiên cần download model
# Ví dụ với Llama 2
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf ./models/llama2

# Sau đó chạy
python generates/vi_s1k_builder.py \
    --translator local \
    --max-samples 100 \
    --output-dir ./results/vi_s1k
```

### Cách 4: Sử dụng trong Python code

```python
from generates.s1k_translator import S1KDatasetLoader, VietnameseBenchmarkBuilder, QualityChecker
from llm_evaluation.utils.translation_utils import get_translator

# Khởi tạo translator
translator = get_translator("gemini")

# Tải dataset
loader = S1KDatasetLoader()
raw_data = loader.load_s1k(max_samples=100)
dataset = [loader.extract_question_fields(item) for item in raw_data]

# Xây dựng benchmark
builder = VietnameseBenchmarkBuilder(translator)
translated_items = builder.build_benchmark(
    dataset,
    quality_checker=QualityChecker.simple_check,
    show_progress=True
)

# Lưu kết quả
builder.save_benchmark("./results/vi_s1k.json", format="json")

# In thống kê
stats = builder.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average quality: {stats['average_quality_score']:.2f}")
```

## 📊 Kết quả

### Cấu trúc output

```
results/vi_s1k/
├── vi_s1k_benchmark.json      # Dữ liệu chi tiết
├── vi_s1k_benchmark.jsonl     # Dạng JSONL
├── vi_s1k_benchmark.csv       # Dạng CSV
└── statistics.json             # Thống kê
```

### Ví dụ dữ liệu trong JSON

```json
{
  "metadata": {
    "source": "simplescaling/s1K-1.1",
    "target_language": "Vietnamese",
    "benchmark_name": "Vi-S1K",
    "creation_date": "2024-11-30T10:30:00",
    "total_items": 100,
    "translation_model": "gemini-pro",
    "translator_stats": {
      "total_translations": 100,
      "cache_hits": 5,
      "cache_hit_rate": 0.05
    }
  },
  "items": [
    {
      "id": "q001",
      "original_question": "What is 2 + 2?",
      "vietnamese_question": "2 cộng 2 bằng bao nhiêu?",
      "original_answer": "4",
      "vietnamese_answer": "4",
      "domain": "mathematics",
      "difficulty": "easy",
      "tags": ["arithmetic", "addition"],
      "metadata": {},
      "translation_model": "gemini-pro",
      "quality_score": 0.95
    }
  ]
}
```

## ⚙️ Các tùy chọn command line

```bash
python generates/vi_s1k_builder.py --help

Options:
  --translator {gemini,openai,local}  Backend dịch (mặc định: gemini)
  --max-samples N                      Số lượng mẫu tối đa
  --output-dir PATH                    Thư mục output
  --split {train,test,validation}      Dataset split
  --formats FORMAT1 FORMAT2 ...         Định dạng output
  --no-quality-check                   Bỏ qua kiểm tra chất lượng
  --cache-dir PATH                     Thư mục cache
  --test-run                           Chạy test với 5 mẫu
```

## 🔧 Cấu trúc code

```
llm_evaluation/utils/
└── translation_utils.py          # Các lớp translator
    ├── TranslationCache          # Quản lý cache
    ├── TranslationPromptBuilder  # Xây dựng prompt
    ├── MultilingualTranslator    # Base class
    ├── GeminiTranslator          # Gemini backend
    ├── OpenAITranslator          # OpenAI backend
    └── LocalLLMTranslator        # Local model backend

generates/
├── s1k_translator.py             # S1K dataset builder
│   ├── S1KDatasetLoader          # Tải S1K dataset
│   ├── VietnameseBenchmarkBuilder# Xây dựng benchmark
│   ├── TranslatedQuestion        # Dataclass cho câu hỏi dịch
│   └── QualityChecker            # Kiểm tra chất lượng
│
└── vi_s1k_builder.py             # Main script
```

## 💡 Một số tips tối ưu

### 1. **Sử dụng cache hiệu quả**
```python
# Cache được lưu ở ./translation_cache
# Nếu chạy lại cùng dataset, tự động sử dụng cache
translator = get_translator("gemini", cache_dir="./translation_cache")
```

### 2. **Giảm chi phí API**
```bash
# Chạy test trước với --test-run
python generates/vi_s1k_builder.py --translator gemini --test-run

# Sau đó chạy full dataset
python generates/vi_s1k_builder.py --translator gemini --max-samples 10000
```

### 3. **Xử lý dataset lớn**
```bash
# Chia nhỏ thành batches nếu cần
# Chạy lần 1: 1000 mẫu
python generates/vi_s1k_builder.py --max-samples 1000 --output-dir ./results/batch1

# Chạy lần 2: 1000 mẫu tiếp
python generates/vi_s1k_builder.py --max-samples 1000 --output-dir ./results/batch2
```

### 4. **Tích hợp với evaluator hiện tại**
```python
from llm_evaluation.core.evaluator import Evaluator

# Sử dụng Vi-S1K với evaluator của project
evaluator = Evaluator(
    models=["gemini", "llama"],
    questions_file="./results/vi_s1k/vi_s1k_benchmark.json"
)
evaluator.run()
```

## 📈 Thống kê

Ví dụ output thống kê:

```
============================================================
Build Complete!
============================================================

Vi-S1K Benchmark Statistics:
  Total items: 100
  Successfully translated: 98
  Failed: 2
  Success rate: 98.0%
  Average quality score: 0.87/1.0

Translator Statistics:
  Total translations: 198 (100 questions + 98 answers)
  Cache hits: 12
  Cache hit rate: 6.1%
```

## 🐛 Troubleshooting

### 1. **Lỗi: "datasets module not found"**
```bash
pip install datasets
```

### 2. **Lỗi: "GEMINI_API_KEY not found"**
- Kiểm tra file `.env` có chứa `GEMINI_API_KEY`
- Hoặc set environment variable: `export GEMINI_API_KEY=your_key`

### 3. **Dịch quá chậm**
- Cache được tự động lưu - lần thứ 2 sẽ nhanh hơn
- Cân nhắc sử dụng `--max-samples` để test trước

### 4. **Lỗi encoding trên Windows**
```python
# Đảm bảo UTF-8 encoding
export PYTHONIOENCODING=utf-8
```

## 📚 Tham khảo

- **S1K Dataset**: https://huggingface.co/datasets/simplescaling/s1K-1.1
- **Google Gemini API**: https://ai.google.dev/
- **OpenAI API**: https://platform.openai.com/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/

## 📝 License

Dự án này tuân theo cùng license với CoT-tech project.

## 👤 Tác giả

TRUNE - CoT-tech project

---

**Câu hỏi hoặc góp ý?** Vui lòng tạo Issue hoặc Pull Request trên GitHub.
