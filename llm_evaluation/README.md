# Framework Đánh Giá Mô Hình Ngôn Ngữ Lớn (LLM)

Framework này cung cấp một bộ công cụ để đánh giá hiệu suất của các Mô hình Ngôn ngữ Lớn (LLM), đặc biệt tập trung vào các mô hình và bài toán liên quan đến tiếng Việt. Nó cho phép so sánh các mô hình khác nhau dựa trên nhiều loại prompt kỹ thuật và các metrics đánh giá đa dạng.

## Tính Năng Nổi Bật

*   **Hỗ trợ Đa Mô Hình**: Dễ dàng tích hợp và đánh giá các LLM phổ biến, bao gồm:
    *   **Mô hình Cục bộ (Local)**: Llama, Qwen (yêu cầu tải model về máy).
    *   **Mô hình API**: Gemini (Google), Groq.
*   **Kỹ Thuật Prompt Đa Dạng**: Hỗ trợ nhiều phương pháp prompt tiên tiến để đánh giá khả năng của mô hình trong các tình huống khác nhau:
    *   Zero-shot
    *   Few-shot (với số lượng ví dụ tùy chỉnh, ví dụ: `few_shot_3`, `few_shot_5`)
    *   Chain-of-Thought (`chain_of_thought`)
    *   Self-Consistency (`cot_self_consistency_3`, `cot_self_consistency_5`)
    *   ReAct (`react`)
*   **Đánh Giá Toàn Diện**: Cung cấp nhiều loại metrics:
    *   **Metrics Cơ Bản**: Accuracy, Latency (thời gian xử lý), Token Count, Tokens Per Second.
    *   **Đánh Giá Suy Luận (Reasoning)**: Sử dụng LLM (mặc định là Groq `llama3-70b`) để đánh giá chất lượng suy luận logic, toán học, độ rõ ràng, đầy đủ và liên quan của câu trả lời (có thể cấu hình).
    *   **Đánh Giá Tính Nhất Quán (Consistency)**: Phân tích sự nhất quán trong các câu trả lời của mô hình khi sử dụng kỹ thuật self-consistency.
    *   **Đánh Giá Tính Đầy Đủ (Completeness)**: Đánh giá xem câu trả lời có bao phủ hết các khía cạnh của câu hỏi không.
    *   **Đánh Giá Độ Tương Đồng (Similarity)**: Tính toán ROUGE, BLEU và tùy chọn cosine similarity của embeddings (nếu cung cấp mô hình) để so sánh câu trả lời với đáp án chuẩn.
    *   **Phân Tích Lỗi (Error Analysis)**: Tự động phân loại các câu trả lời sai vào các nhóm lỗi (Kiến thức, Suy luận, Tính toán, Lạc đề, v.v.) sử dụng LLM.
*   **Tối Ưu Hóa Hiệu Suất**:
    *   **Quantization**: Tự động áp dụng quantization 4-bit (BitsAndBytes) cho model cục bộ để giảm yêu cầu bộ nhớ GPU.
    *   **Memory Management**: Tự động tính toán và phân bổ bộ nhớ GPU/CPU (`device_map="auto"`), hỗ trợ CPU offload.
    *   **Model Caching**:
        *   *Memory Cache*: Giữ các model thường dùng trong RAM/VRAM để truy cập nhanh.
        *   *Disk Cache*: Lưu trữ model đã tải và quantized trên đĩa để khởi động nhanh hơn trong các lần chạy sau (có thể bật/tắt).
    *   **API Resilience**: Tự động retry khi gọi API lỗi với cơ chế exponential backoff, quản lý rate limiting.
*   **Checkpointing**: Tự động lưu trạng thái đánh giá định kỳ hoặc khi bị ngắt, cho phép tiếp tục (`--resume`) từ lần chạy trước.
*   **Báo Cáo Chi Tiết**: Tự động tạo báo cáo tổng hợp kết quả:
    *   **HTML**: Báo cáo tương tác với bảng biểu, thống kê và các biểu đồ trực quan (accuracy, latency, reasoning scores, heatmap, v.v.).
    *   **CSV/JSON**: Dữ liệu tổng hợp và kết quả thô để phân tích sâu hơn.
*   **Linh Hoạt**: Dễ dàng mở rộng để hỗ trợ thêm mô hình, loại prompt, metrics đánh giá, hoặc nguồn dữ liệu câu hỏi mới.

## Cài Đặt

1.  **Clone Repository**:
    ```bash
    git clone <URL_repository_cua_ban>
    cd llm_evaluation
    ```

2.  **Tạo Môi Trường (Khuyến nghị)**:
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **Cài Đặt Dependencies**:
    *   **Cài đặt**:
        ```bash
        pip install -r requirements.txt
        # Cài đặt torch phù hợp với hệ thống CUDA của bạn nếu cần
        # Ví dụ: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **Download dữ liệu NLTK (nếu chưa có)**: Cần cho tính BLEU score.
        ```python
        import nltk
        nltk.download('punkt')
        ```

4.  **Cấu Hình Môi Trường (`.env`)**:
    *   Sao chép file `.env.example` (nếu có) thành `.env` hoặc tạo file `.env` mới trong thư mục gốc của dự án.
    *   Điền các thông tin cần thiết:
        ```dotenv
        # --- API Keys ---
        # Bắt buộc nếu sử dụng model tương ứng
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        GROQ_API_KEY="YOUR_GROQ_API_KEY"
        # OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # Nếu tích hợp OpenAI

        # --- Local Model Paths ---
        # Đường dẫn tuyệt đối hoặc tương đối đến thư mục chứa model và tokenizer đã tải về
        # Bắt buộc nếu sử dụng model local tương ứng
        LLAMA_MODEL_PATH="/path/to/your/llama/model"
        LLAMA_TOKENIZER_PATH="/path/to/your/llama/tokenizer" # Thường cùng đường dẫn với model
        QWEN_MODEL_PATH="/path/to/your/qwen/model"
        QWEN_TOKENIZER_PATH="/path/to/your/qwen/tokenizer" # Thường cùng đường dẫn với model

        # --- GPU & Memory Configuration (Optional - Điều chỉnh nếu cần) ---
        # Dung lượng GPU tối đa (GB) cho mỗi card, framework sẽ tự tính toán phần còn lại
        # MAX_GPU_MEMORY_GB=140 # Ví dụ cho A100 80GB x 2
        # Dung lượng GPU (GB) dự trữ cho hệ thống/OS
        SYSTEM_RESERVE_MEMORY_GB=2.5
        # Dung lượng RAM (GB) sử dụng cho CPU offloading khi GPU không đủ
        CPU_OFFLOAD_GB=24

        # --- Disk Cache Configuration (Optional) ---
        # Thư mục lưu cache model trên đĩa
        MODEL_CACHE_DIR="./model_cache"
        # Bật/tắt disk cache (true/false)
        ENABLE_DISK_CACHE=true
        # Số lượng model tối đa lưu trong disk cache (LRU)
        MAX_CACHED_MODELS=2
        ```

5.  **Chuẩn Bị Dữ Liệu Câu Hỏi**:
    *   Đảm bảo file câu hỏi (`data/questions/problems.json` theo mặc định trong `config.py`) tồn tại và có định dạng JSON hợp lệ. Mỗi câu hỏi nên là một object JSON chứa ít nhất trường `id` (duy nhất) và `question`.
    *   Các trường tùy chọn khác có thể bao gồm: `correct_answer` (đáp án đúng), `category` (chủ đề), `difficulty` (độ khó), `task_type` (loại tác vụ), `examples` (ví dụ cho few-shot).

## Cách Sử Dụng

### Chạy Đánh Giá

Sử dụng script `main.py` từ dòng lệnh.

**Cú pháp cơ bản**:

```bash
python main.py [--models MODEL1 MODEL2 ...] [--prompts PROMPT1 PROMPT2 ...] [OPTIONS]
```

**Các tham số chính**:

*   `--models`: (Bắt buộc nếu không dùng mặc định) Danh sách các model cần đánh giá (ví dụ: `llama qwen gemini`). Tên model phải khớp với key trong `config.MODEL_CONFIGS` hoặc `.env`.
*   `--prompts`: (Bắt buộc nếu không dùng mặc định) Danh sách các loại prompt cần đánh giá (ví dụ: `zero_shot few_shot_3 cot_self_consistency_5`).
*   `--questions-file`: Đường dẫn đến file JSON chứa câu hỏi (mặc định: `data/questions/problems.json`).
*   `--results-dir`: Thư mục lưu kết quả (mặc định: `results`).
*   `--max-questions`: Số lượng câu hỏi tối đa cần đánh giá từ file (mặc định: tất cả).
*   `--batch-size`: Kích thước batch khi xử lý câu hỏi (ảnh hưởng đến việc sử dụng bộ nhớ, mặc định: 5).
*   `--checkpoint-frequency`: Tần suất lưu checkpoint (số câu hỏi, mặc định: 5).
*   `--resume`: Tiếp tục từ checkpoint tự động gần nhất trong `results/checkpoints`.
*   `--checkpoint <path>`: Tiếp tục từ một file checkpoint cụ thể.
*   `--test-run`: Chạy thử nghiệm nhanh với 1 model, 1 prompt, và 2 câu hỏi.
*   `--skip-reasoning-eval`: Bỏ qua bước đánh giá suy luận (ngay cả khi được bật trong `config.py`).
*   `--no-cache`: Vô hiệu hóa memory cache (luôn tải lại model).
*   `--question-ids ID1 ID2 ...`: Chỉ đánh giá các câu hỏi có ID cụ thể.
*   `--parallel`: **(Chưa triển khai)** Bật chế độ đánh giá song song.
*   `--gpu-ids ID1 ID2 ...`: Chỉ định các GPU ID để sử dụng (mặc định: 0).
*   `--report-only`: Chỉ tạo báo cáo từ file kết quả đã có mà không chạy đánh giá mới. Yêu cầu dùng kèm `--results-file`.
*   `--results-file <path>`: Đường dẫn đến file kết quả (`.csv` hoặc `.json`) để tạo báo cáo khi dùng `--report-only`.
*   `--log-level`: Mức độ log cho console (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, mặc định: `INFO`).
*   `--log-file <name>`: Tên file log cụ thể (mặc định: tự tạo tên với timestamp trong `results/logs`).
*   `--debug`: Bật chế độ debug (tương đương `--log-level DEBUG`).

**Ví dụ**:

```bash
# Đánh giá Llama và Gemini với prompt zero_shot và few_shot_3 cho 50 câu hỏi đầu tiên
python main.py --models llama gemini --prompts zero_shot few_shot_3 --max-questions 50

# Tiếp tục đánh giá từ lần chạy trước
python main.py --resume

# Chỉ tạo báo cáo từ file kết quả đã lưu
python main.py --report-only --results-file results/raw_results/evaluation_results_xxxx.csv
```

### Xem Kết Quả

Kết quả đánh giá và báo cáo sẽ được lưu trong thư mục `--results-dir` (mặc định là `results/`) với cấu trúc con:

*   `raw_results/`: Chứa các file kết quả thô dạng CSV và JSON.
*   `reports/`: Chứa báo cáo tổng hợp dạng HTML.
*   `plots/`: Chứa các file ảnh biểu đồ được tạo cho báo cáo HTML.
*   `checkpoints/`: Chứa các file checkpoint (`.json`).
*   `logs/`: Chứa các file log (`.log`).

Mở file `.html` trong thư mục `reports/` bằng trình duyệt để xem báo cáo tương tác.

## Cấu Trúc Dự Án

```
llm_evaluation/
├── core/                  # Logic cốt lõi
│   ├── evaluator.py         # Điều phối đánh giá
│   ├── model_interface.py   # Giao diện tương tác LLM
│   ├── prompt_builder.py    # Xây dựng prompt
│   ├── result_analyzer.py   # Phân tích kết quả, metrics
│   ├── reporting.py         # Tạo báo cáo
│   ├── checkpoint_manager.py# Quản lý checkpoint
│   └── __init__.py
├── data/                  # Dữ liệu đầu vào
│   └── questions/
│       └── problems.json    # File câu hỏi mẫu
├── results/               # Thư mục lưu kết quả (mặc định)
│   ├── raw_results/
│   ├── reports/
│   ├── plots/
│   ├── checkpoints/
│   └── logs/
├── tests/                 # Unit tests
│   ├── test_checkpoint_manager.py
│   └── test_utils.py
├── utils/                 # Các module tiện ích
│   ├── config_utils.py    # Tiện ích cấu hình (dataclasses)
│   ├── data_loader.py     # Tải dữ liệu (CẦN KIỂM TRA/TRIỂN KHAI)
│   ├── file_utils.py      # Tiện ích file I/O
│   ├── logging_setup.py   # Thiết lập logging
│   ├── logging_utils.py   # Các hàm log chuyên biệt
│   ├── memory_utils.py    # Tiện ích quản lý bộ nhớ
│   ├── metrics_utils.py   # Tính toán metrics cụ thể
│   ├── text_utils.py      # Xử lý văn bản
│   ├── visualization_utils.py # Tạo biểu đồ (có thể tích hợp vào reporting)
│   └── __init__.py
├── model_cache/           # Cache model trên đĩa (nếu bật)
├── main.py                # Điểm vào chính
├── config.py              # Cấu hình mặc định và tải .env
├── requirements.txt       # Danh sách dependencies (CẦN CẬP NHẬT)
├── README.md              # Tài liệu hướng dẫn (file này)
└── .env                   # File cấu hình môi trường (cần tự tạo)
```

## Đóng Góp

Chào mừng các đóng góp! Vui lòng tạo Pull Request hoặc Issue trên repository.

## Tác Giả

Trune

## Giấy Phép

Alonee
