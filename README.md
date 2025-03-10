# LLM Evaluation for Classical Problems

Hệ thống đánh giá và so sánh hiệu năng của các mô hình ngôn ngữ lớn (LLM) khi giải quyết các bài toán cổ điển được trích xuất từ tài liệu PDF hoặc được tạo tự động.

## Tổng quan

Dự án này cung cấp một framework toàn diện để:

1. Tạo tự động các bài toán cổ điển đa dạng với nhiều loại và độ khó khác nhau
2. Trích xuất các câu hỏi và bài toán từ tài liệu PDF "Những bài toán cổ điển"
3. Áp dụng nhiều loại prompt khác nhau (Standard, Chain-of-Thought, Hybrid-CoT, v.v.) 
4. Đánh giá hiệu năng của các mô hình ngôn ngữ lớn khác nhau (Llama-3.3-70B, Qwen-2.5-72B, Gemini)
5. Tối ưu hóa việc sử dụng GPU để xử lý các mô hình nặng (>70B tham số)
6. Tạo báo cáo và biểu đồ so sánh chi tiết

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
│   ├── Nhung_bai_toan_co_dien.pdf  # File PDF chứa các bài toán cổ điển đã tạo
│   └── questions/             # Thư mục chứa các câu hỏi đã trích xuất
│       └── problems.json      # File JSON chứa tất cả các bài toán đã tạo
├── cache/                     # Cache cho mô hình đã tải
├── offload/                   # Thư mục để offload mô hình khi cần
├── results/                   # Kết quả đánh giá và báo cáo
├── generate_problems.py       # Module tạo các bài toán cơ bản
├── generate_classical_problems.py # Module tạo các bài toán cổ điển
├── generate_all_problems.py   # Module tạo tất cả các loại bài toán
├── extract_questions.py       # Module trích xuất câu hỏi từ PDF
├── prompts.py                 # Các mẫu prompt (Standard, CoT, Hybrid-CoT)
├── model_manager.py           # Quản lý việc tải và tối ưu hóa mô hình
├── model_evaluator.py         # Đánh giá và phân tích hiệu năng mô hình
├── evaluate_models.py         # Script chính để chạy đánh giá
├── analyze_problems.py        # Script phân tích các bài toán đã tạo
├── requirements.txt           # Các thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn
```

## Cách sử dụng

### Tạo bài toán tự động

Tạo tất cả các loại bài toán (khoảng 2345 bài):

```bash
python generate_all_problems.py
```

Tạo các bài toán cơ bản:

```bash
python generate_problems.py
```

Tạo các bài toán cổ điển:

```bash
python generate_classical_problems.py
```

Phân tích các bài toán đã tạo:

```bash
python analyze_problems.py
```

### Trích xuất câu hỏi từ PDF (trong trường hợp đã load thì không cần)

```bash
python extract_questions.py
```

### Đánh giá các mô hình

Chạy đánh giá với cấu hình mặc định (sử dụng problems.json):

```bash
python evaluate_models.py
```

Tùy chỉnh các tham số:

```bash
python evaluate_models.py \
    --questions_json db/questions/problems.json \
    --models llama qwen gemini \
    --prompt_types standard hybrid_cot \
    --batch_size 10 \
    --max_workers 3
```

### Các tham số quan trọng

- `--questions_json`: Đường dẫn đến file JSON chứa các câu hỏi (mặc định: db/questions/problems.json)
- `--batch_size`: Số câu hỏi xử lý cùng lúc (mặc định: 10)
- `--max_workers`: Số luồng xử lý song song (mặc định: 3, một cho mỗi GPU)
- `--models`: Chọn mô hình để đánh giá (mặc định: tất cả)
- `--prompt_types`: Loại prompt để sử dụng (mặc định: tất cả)
- `--use_4bit`: Sử dụng lượng tử hóa 4-bit để tiết kiệm VRAM (mặc định: True)

## Tạo bài toán tự động

Hệ thống có thể tạo tự động nhiều loại bài toán khác nhau:

1. **Bài toán logic**: Các câu đố logic đơn giản
2. **Bài toán kiểu luận**: Các bài toán yêu cầu giải thích chi tiết
3. **Thơ toán học**: Các bài toán được trình bày dưới dạng thơ
4. **Bài toán từ vựng**: Các bài toán liên quan đến từ vựng toán học
5. **Bài toán trắc nghiệm**: Các câu hỏi trắc nghiệm
6. **Bài toán chuyển động**: Các bài toán về vận tốc, thời gian, quãng đường
7. **Bài toán về tuổi**: Các bài toán liên quan đến tuổi tác
8. **Bài toán chia kẹo**: Các bài toán về chia đều
9. **Bài toán hồ bơi**: Các bài toán về thời gian đổ đầy/xả cạn
10. **Bài toán phân số**: Các bài toán về phân số
11. **Bài toán công việc**: Các bài toán về thời gian hoàn thành công việc
12. **Bài toán hỗn hợp**: Các bài toán về nồng độ, hỗn hợp
13. **Bài toán số học**: Các bài toán về số học
14. **Bài toán hình học**: Các bài toán về hình học
15. **Bài toán tỷ lệ**: Các bài toán về tỷ lệ

Mỗi loại bài toán có nhiều mẫu (templates) khác nhau và được tạo với các tham số ngẫu nhiên để đảm bảo tính đa dạng. Các bài toán được phân loại theo độ khó (Dễ, Trung bình, Khó) và được gắn các thẻ (tags) phù hợp.

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


## Tác giả

Truneee