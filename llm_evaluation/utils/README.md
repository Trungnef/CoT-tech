# Tiện ích (Utils) cho Framework Đánh giá LLM

Thư mục này chứa các module tiện ích độc lập với logic core của framework đánh giá LLM. Các module này cung cấp các chức năng phổ biến và có thể tái sử dụng trong nhiều phần khác nhau của framework.

## Cấu trúc thư mục

- `__init__.py`: File khởi tạo module
- `file_utils.py`: Các tiện ích xử lý file và I/O
- `logging_utils.py`: Tiện ích quản lý logging
- `text_utils.py`: Tiện ích xử lý văn bản, đặc biệt hỗ trợ tiếng Việt
- `metrics_utils.py`: Tiện ích tính toán các metrics đánh giá
- `visualization_utils.py`: Tiện ích tạo biểu đồ và trực quan hóa
- `config_utils.py`: Tiện ích quản lý cấu hình
- `memory_utils.py`: Tiện ích quản lý bộ nhớ

## Mô tả các module

### `file_utils.py`

Module này cung cấp các tiện ích xử lý file và I/O, bao gồm:

- Đọc/ghi file với nhiều định dạng (JSON, YAML, CSV, Pickle)
- Quản lý đường dẫn và thư mục
- Xử lý an toàn khi ghi file
- Các tiện ích làm việc với timestamp và kích thước file

### `logging_utils.py`

Module này cung cấp các tiện ích quản lý logging, bao gồm:

- Cấu hình logging nhất quán cho toàn bộ framework
- Hỗ trợ màu sắc cho các log level khác nhau
- Hỗ trợ log ra console và file
- Cung cấp decorator để log function calls

### `text_utils.py`

Module này cung cấp các tiện ích xử lý văn bản, đặc biệt hỗ trợ tiếng Việt:

- Làm sạch văn bản: loại bỏ dấu câu, số, khoảng trắng
- Loại bỏ dấu thanh/dấu phụ trong tiếng Việt
- Chuẩn hóa văn bản tiếng Việt
- Tokenize đơn giản cho văn bản tiếng Việt
- Tính toán độ tương đồng giữa các văn bản
- Trích xuất từ khóa, câu và số từ văn bản

### `metrics_utils.py`

Module này cung cấp các tiện ích tính toán metrics đánh giá:

- Metrics cho bài toán phân loại nhị phân: accuracy, precision, recall, F1, AUC
- Metrics cho bài toán phân loại đa lớp
- Metrics cho bài toán regression: MSE, RMSE, MAE, R², MAPE
- Metrics đặc thù cho đánh giá LLM: exact match, token overlap
- Metrics đánh giá khả năng lập luận của LLM
- Metrics về độ trễ (latency)

### `visualization_utils.py`

Module này cung cấp các tiện ích tạo biểu đồ và trực quan hóa:

- Biểu đồ so sánh độ chính xác giữa các mô hình
- Biểu đồ heatmap cho các metrics
- Biểu đồ phân tích độ trễ
- Biểu đồ radar đánh giá nhiều tiêu chí
- Biểu đồ so sánh hiệu suất theo loại prompt
- Biểu đồ hiển thị số lượng mẫu
- Biểu đồ ma trận nhầm lẫn (confusion matrix)

### `config_utils.py`

Module này cung cấp các tiện ích quản lý cấu hình:

- Đọc/ghi cấu hình từ/vào file YAML/JSON
- Xác thực cấu hình theo schema
- Gộp và cập nhật cấu hình
- Cập nhật cấu hình từ biến môi trường
- Cung cấp cấu trúc dữ liệu mạnh cho cấu hình

### `memory_utils.py`

Module này cung cấp các tiện ích quản lý bộ nhớ:

- Giám sát sử dụng bộ nhớ
- Tự động giải phóng bộ nhớ khi gần hết
- Theo dõi và quản lý các đối tượng lớn
- Ước tính kích thước đối tượng trong bộ nhớ
- Cung cấp decorator để theo dõi sử dụng bộ nhớ của hàm

## Cách sử dụng

```python
# Sử dụng logging
from utils.logging_utils import get_logger, setup_logging

setup_logging(log_level="INFO", log_file="logs/app.log")
logger = get_logger(__name__)
logger.info("Ứng dụng đã khởi động")

# Sử dụng file_utils
from utils.file_utils import load_json, save_json, ensure_dir

data = load_json("config.json")
data["new_setting"] = "value"
ensure_dir("outputs/results")
save_json(data, "outputs/results/updated_config.json")

# Sử dụng text_utils
from utils.text_utils import clean_text, normalize_vietnamese_text

text = "  Đây Là Một Câu Văn Bản. 123  "
cleaned = clean_text(text, remove_punctuation=True, remove_whitespace=True)
normalized = normalize_vietnamese_text("Đây  là  câu . Có dấu  câu  không chuẩn !")

# Sử dụng metrics_utils
from utils.metrics_utils import calculate_binary_metrics

y_true = [1, 0, 1, 0, 1]
y_pred = [1, 0, 0, 1, 1]
metrics = calculate_binary_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")

# Sử dụng config_utils
from utils.config_utils import load_config, create_default_config

# Tạo cấu hình mặc định
default_config = create_default_config()
# Hoặc tải từ file
config = load_config("config.yaml") 