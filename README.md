# LLM Evaluation Framework

Một framework toàn diện để đánh giá và so sánh hiệu suất của các mô hình ngôn ngữ lớn (LLM) trên các bài toán cổ điển.

## Tính năng

- Hỗ trợ nhiều mô hình: Llama, Qwen, Gemini và dễ dàng mở rộng cho các mô hình khác
- Đánh giá với nhiều loại prompt khác nhau: standard, chain-of-thought, hybrid-cot, zero-shot-cot
- Đánh giá tuần tự hoặc song song sử dụng nhiều GPU
- Phân tích và đánh giá chi tiết với các chỉ số về độ chính xác, chất lượng suy luận, độ tự tin
- Tạo báo cáo HTML tương tác với biểu đồ và trực quan hóa so sánh
- Tối ưu hóa sử dụng GPU với cơ chế phân phối và lưu trữ bộ nhớ

## Cấu trúc thư mục

```
├── evaluate.py               # Entry point script
├── src/                      # Thư mục mã nguồn chính
│   ├── core/                 # Chức năng cốt lõi
│   │   ├── main.py           # Điểm vào chính của ứng dụng
│   │   └── __init__.py
│   ├── evaluators/           # Các lớp đánh giá mô hình
│   │   ├── evaluator.py      # Lớp đánh giá tuần tự
│   │   ├── parallel_evaluator.py # Lớp đánh giá song song
│   │   ├── metrics.py        # Các hàm tính toán chỉ số đánh giá
│   │   └── __init__.py
│   ├── models/               # Quản lý mô hình
│   │   ├── model_loader.py   # Tải và tối ưu mô hình
│   │   ├── model_generator.py # Tạo text từ mô hình
│   │   └── __init__.py
│   ├── prompts/              # Các mẫu prompt khác nhau
│   │   ├── base_prompts.py   # Các prompt cơ bản
│   │   ├── advanced_prompts.py # Các prompt nâng cao
│   │   └── __init__.py
│   ├── utils/                # Tiện ích
│   │   ├── logging_utils.py  # Ghi log và hiển thị trạng thái
│   │   ├── file_utils.py     # Thao tác file
│   │   └── __init__.py
│   ├── visualization/        # Trực quan hóa
│   │   ├── report_generator.py # Tạo báo cáo HTML
│   │   └── __init__.py
│   └── __init__.py
├── results/                  # Kết quả đánh giá
├── model_cache/              # Cache của mô hình
├── sample_questions.json     # Dữ liệu câu hỏi mẫu
└── requirements.txt          # Các phụ thuộc
```

## Cài đặt

1. Clone repository:

```bash
git clone https://github.com/username/llm-evaluation.git
cd llm-evaluation
```

2. Cài đặt các phụ thuộc:

```bash
pip install -r requirements.txt
```

3. Thiết lập biến môi trường trong file `.env`:

```
LLAMA_MODEL_PATH=/path/to/llama/model
LLAMA_TOKENIZER_PATH=/path/to/llama/tokenizer

QWEN_MODEL_PATH=/path/to/qwen/model
QWEN_TOKENIZER_PATH=/path/to/qwen/tokenizer

GEMINI_API_KEY=your_gemini_api_key
```

## Sử dụng

### Đánh giá tuần tự

Đánh giá tuần tự với một GPU:

```bash
python evaluate.py --models llama gemini --prompt_types standard cot hybrid_cot --batch_size 5 --max_questions 100
```

### Đánh giá song song

Đánh giá song song với nhiều GPU:

```bash
python evaluate.py --parallel --gpu_ids "0,1,2" --gpu_allocation "llama:0" "llama:1" "qwen:2" --models llama qwen gemini --prompt_types standard cot hybrid_cot zero_shot_cot --batch_size 5 --max_questions 100
```

### Các tùy chọn khác

- `--questions_file`: Đường dẫn đến file JSON chứa câu hỏi (mặc định: sample_questions.json)
- `--max_questions`: Số lượng câu hỏi tối đa để đánh giá (mặc định: tất cả)
- `--models`: Danh sách mô hình để đánh giá, có thể chọn từ [llama, qwen, gemini]
- `--prompt_types`: Các loại prompt, có thể chọn từ [standard, cot, hybrid_cot, zero_shot_cot]
- `--parallel`: Chạy đánh giá song song
- `--gpu_ids`: Danh sách các ID GPU, phân tách bằng dấu phẩy
- `--gpu_allocation`: Phân phối GPU cho từng mô hình, vd: "llama:0" "qwen:1"
- `--batch_size`: Kích thước batch cho xử lý câu hỏi
- `--output_dir`: Thư mục để lưu kết quả
- `--verbose`: Bật output chi tiết

## Xem báo cáo

Sau khi đánh giá hoàn tất, một báo cáo HTML sẽ được tạo trong thư mục kết quả. Báo cáo này chứa:

- Tổng quan về cấu hình đánh giá
- Thống kê tổng hợp
- So sánh hiệu suất giữa các mô hình
- So sánh hiệu suất giữa các loại prompt
- Trực quan hóa tương tác (biểu đồ radar, heatmap, bubble chart)

Mở file `evaluation_report.html` trong thư mục kết quả để xem báo cáo.

## Mở rộng

### Thêm mô hình mới

1. Bổ sung hỗ trợ cho mô hình mới trong `src/models/model_loader.py`
2. Thêm logic để tải và cấu hình mô hình
3. Cập nhật câu lệnh trợ giúp trong `src/core/main.py` để thêm mô hình mới vào lựa chọn

### Thêm loại prompt mới

1. Tạo hàm prompt mới trong `src/prompts/base_prompts.py` hoặc `src/prompts/advanced_prompts.py`
2. Đăng ký prompt trong `src/prompts/__init__.py`
3. Cập nhật lớp `ModelEvaluator` để hỗ trợ loại prompt mới

## Đóng góp

Đóng góp luôn được chào đón! Vui lòng tạo issue hoặc pull request để cải thiện dự án.

# Hệ thống đánh giá toàn diện cho Prompt Engineering

Hệ thống đánh giá này giúp so sánh hiệu suất của các loại prompt khác nhau (zero-shot, few-shot, Chain of Thought, self-consistency, ReAct) trên các model ngôn ngữ lớn.

## Sử dụng cơ bản

```bash
# Đánh giá toàn diện cho các prompts
python evaluate_models.py --comprehensive_prompt_eval --ground_truth data/ground_truth.json

# Tùy chọn về model và prompt type
python evaluate_models.py --comprehensive_prompt_eval --models llama qwen gemini --prompt_types zero_shot few_shot_3 cot react

# Đánh giá với custom examples cho few-shot
python evaluate_models.py --comprehensive_prompt_eval --examples_file data/custom_examples.json
```

## Tham số command line

| Tham số | Mô tả |
|---------|-------|
| `--comprehensive_prompt_eval` | Chạy đánh giá toàn diện cho các prompt với báo cáo chi tiết |
| `--models` | Danh sách các model muốn đánh giá (mặc định: llama, qwen, gemini) |
| `--prompt_types` | Danh sách các loại prompt muốn đánh giá |
| `--ground_truth` | Đường dẫn đến file JSON chứa câu trả lời chính xác |
| `--questions_json` | Đường dẫn đến file JSON chứa câu hỏi (mặc định: db/questions/problems.json) |
| `--results_dir` | Thư mục lưu kết quả (mặc định: results) |
| `--max_questions` | Số lượng câu hỏi tối đa cần đánh giá |

## Đánh giá Few-shot với các examples tùy chỉnh

Để sử dụng custom examples cho few-shot, tạo file JSON với cấu trúc:

```json
{
  "math": [
    {
      "problem": "Tính diện tích hình tròn có bán kính 5cm.",
      "solution": "Diện tích = π × r² = π × 5² = 25π ≈ 78.54 cm²",
      "reasoning": "Diện tích hình tròn được tính bằng công thức π × r² với r là bán kính. Thay r = 5cm, ta có diện tích = π × 5² = 25π ≈ 78.54 cm²."
    },
    // Thêm các examples khác
  ],
  "general": [
    // Examples cho các câu hỏi tổng quát
  ]
}
```

Sau đó chỉ định file này khi chạy đánh giá:

```bash
python evaluate_models.py --comprehensive_prompt_eval --examples_file path/to/examples.json
```

## Cấu trúc báo cáo đánh giá

Báo cáo đánh giá toàn diện bao gồm:

1. **Đánh giá độ chính xác (Accuracy)**: So sánh câu trả lời với ground truth
2. **Đánh giá chất lượng suy luận (Reasoning)**: Phân tích quá trình suy luận
3. **Phân tích tính nhất quán (Consistency)**: Đánh giá độ ổn định của câu trả lời
4. **Đánh giá hiệu suất (Performance)**: Thời gian phản hồi, độ dài câu trả lời
5. **Xếp hạng tổng hợp (Overall ranking)**: Xếp hạng các prompt dựa trên kết hợp các tiêu chí

## Các biểu đồ được tạo ra

- **Bar chart độ chính xác theo prompt type**
- **Bar chart chất lượng suy luận theo prompt type**
- **Heatmap độ chính xác theo model và prompt type**
- **Scatter plot so sánh độ chính xác và chất lượng suy luận**
- **Biểu đồ xếp hạng tổng hợp các prompt**
- **Radar chart cho top prompt types**

## Mở rộng

Bạn có thể mở rộng hệ thống bằng cách:

1. Thêm model mới trong `model_config` trong file `evaluate_models.py`
2. Thêm kiểu prompt mới trong module `prompts.py`
3. Tùy chỉnh trọng số đánh giá trong `metrics_weights` 