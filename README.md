# LLM Evaluation for Classical Problems

Hệ thống đánh giá và so sánh hiệu năng của các mô hình ngôn ngữ lớn (LLM) khi giải quyết các bài toán cổ điển được trích xuất từ tài liệu PDF.

## Tổng quan

Dự án này cung cấp một framework toàn diện để:

1. Trích xuất các câu hỏi và bài toán từ tài liệu PDF "Những bài toán cổ điển"
2. Áp dụng nhiều loại prompt khác nhau (Standard, Chain-of-Thought, Hybrid-CoT, v.v.) 
3. Đánh giá hiệu năng của các mô hình ngôn ngữ lớn khác nhau (Llama-3.3-70B, Qwen-2.5-72B, Gemini)
4. Tối ưu hóa việc sử dụng GPU để xử lý các mô hình nặng (>70B tham số)
5. Tạo báo cáo và biểu đồ so sánh chi tiết

## Yêu cầu hệ thống

- Python 3.8+
- 3 GPU AMD 6000 (mỗi card 47.5GB VRAM, tổng 142.5GB)
- CUDA toolkit phiên bản mới nhất
- Tối thiểu 32GB RAM hệ thống
- Ổ cứng SSD với ít nhất 200GB trống

## Cấu hình GPU

Hệ thống được tối ưu hóa để tận dụng tối đa 3 GPU AMD 6000:
- Mỗi GPU có 47.5GB VRAM
- Để lại 2.5GB mỗi GPU cho system operations
- Phân bổ model tự động dựa trên kích thước:
  - Model ≤45GB: Sử dụng 1 GPU
  - Model ≤90GB: Sử dụng 2 GPU
  - Model >90GB: Sử dụng cả 3 GPU

## Xử lý song song

Hệ thống sử dụng nhiều cơ chế xử lý song song:
1. **Model Parallelism**: Phân chia mô hình lớn trên nhiều GPU
2. **Data Parallelism**: Xử lý nhiều câu hỏi cùng lúc
3. **Batch Processing**: Xử lý câu hỏi theo batch để tối ưu hiệu suất
4. **Multi-threading**: Sử dụng ThreadPoolExecutor để quản lý tác vụ

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Cấu hình file `.env`:

Tạo file `.env` trong thư mục gốc và thêm các thông tin sau:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Model Paths
QWEN_MODEL_PATH=cache/model/Qwen_Qwen2.5-72B-Instruct/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
QWEN_TOKENIZER_PATH=cache/tokenizer/Qwen_Qwen2.5-72B-Instruct/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31

LLAMA_MODEL_PATH=cache/model/meta-llama_Llama-3.3-70B-Instruct/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
LLAMA_TOKENIZER_PATH=cache/tokenizer/meta-llama_Llama-3.3-70B-Instruct/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b

# GPU Configuration
MAX_GPU_MEMORY_GB=47.5
SYSTEM_RESERVE_MEMORY_GB=2.5
CPU_OFFLOAD_GB=24
```

> **Lưu ý**: File `.env` chứa thông tin nhạy cảm, đảm bảo không commit file này lên git.

## Cấu trúc dự án

```
.
├── db/                        # Thư mục chứa dữ liệu
│   ├── Nhung bai toan co.pdf  # File PDF chứa các bài toán cổ điển
│   └── questions/             # Thư mục chứa các câu hỏi đã trích xuất
├── cache/                     # Cache cho mô hình đã tải
├── offload/                   # Thư mục để offload mô hình khi cần
├── results/                   # Kết quả đánh giá và báo cáo
├── extract_questions.py       # Module trích xuất câu hỏi từ PDF
├── prompts.py                 # Các mẫu prompt (Standard, CoT, Hybrid-CoT)
├── model_manager.py           # Quản lý việc tải và tối ưu hóa mô hình
├── model_evaluator.py         # Đánh giá và phân tích hiệu năng mô hình
├── evaluate_models.py         # Script chính để chạy đánh giá
├── requirements.txt           # Các thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn
```

## Cách sử dụng

### Trích xuất câu hỏi từ PDF

```bash
python extract_questions.py
```

### Đánh giá các mô hình

Chạy đánh giá với cấu hình mặc định:

```bash
python evaluate_models.py
```

Tùy chỉnh các tham số:

```bash
python evaluate_models.py \
    --models llama qwen gemini \
    --prompt_types standard hybrid_cot \
    --batch_size 10 \
    --max_workers 3
```

### Các tham số quan trọng

- `--batch_size`: Số câu hỏi xử lý cùng lúc (mặc định: 10)
- `--max_workers`: Số luồng xử lý song song (mặc định: 3, một cho mỗi GPU)
- `--models`: Chọn mô hình để đánh giá (mặc định: tất cả)
- `--prompt_types`: Loại prompt để sử dụng (mặc định: tất cả)
- `--use_4bit`: Sử dụng lượng tử hóa 4-bit để tiết kiệm VRAM (mặc định: True)

## Tối ưu hóa hiệu năng

1. **Quản lý bộ nhớ GPU**:
   - Lượng tử hóa 4-bit giảm 75% dung lượng VRAM cần thiết
   - Tự động dọn dẹp bộ nhớ sau mỗi batch
   - Phân phối model tối ưu trên nhiều GPU

2. **Xử lý song song**:
   - Chia batch thành các phần nhỏ cho mỗi GPU
   - Sử dụng thread pool để quản lý tác vụ
   - Cân bằng tải giữa các GPU

3. **Khôi phục lỗi**:
   - Tự động thử lại khi gặp lỗi API
   - Lưu kết quả thường xuyên
   - Có thể tiếp tục từ điểm dừng

## Kết quả đánh giá

Kết quả được lưu trong thư mục `results` với cấu trúc:
```
results/
  ├── YYYYMMDD_HHMMSS/           # Timestamp của lần chạy
  │   ├── evaluation_results.json # Kết quả chi tiết
  │   ├── plots/                 # Các biểu đồ so sánh
  │   └── report.html            # Báo cáo tổng hợp
```

Báo cáo HTML bao gồm:
- So sánh thời gian xử lý
- So sánh độ chính xác
- Biểu đồ hiệu năng
- Mẫu câu trả lời

## Giấy phép

MIT

## Tác giả

[Your Name]