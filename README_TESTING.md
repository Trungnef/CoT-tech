# Cải Tiến Hiệu Suất Đánh Giá LLM

Tài liệu này mô tả các cải tiến được thực hiện để tối ưu hiệu suất hệ thống đánh giá LLM, đặc biệt tập trung vào việc tăng batch size và tận dụng tối đa tài nguyên đa GPU.

## Cải Tiến Đã Thực Hiện

### 1. Tăng Batch Size và Quản Lý Bộ Nhớ

- **Tăng batch size mặc định từ 5 lên 15**: Cho phép xử lý nhiều câu hỏi hơn trong mỗi lần đánh giá.
- **Cập nhật tổng dung lượng GPU**: Điều chỉnh cấu hình `MAX_GPU_MEMORY_GB` lên 144GB để phản ánh chính xác tổng dung lượng của 3 GPU.
- **Điều chỉnh ngưỡng cảnh báo bộ nhớ**: Tăng ngưỡng cảnh báo từ 75% lên 85% và ngưỡng tới hạn từ 85% lên 90%.

### 2. Tối Ưu Hóa Sử Dụng GPU

- **Phân bổ bộ nhớ nhiều GPU hiệu quả hơn**: Cải tiến hàm `_get_max_memory_config()` để tối ưu việc phân bổ bộ nhớ cho hệ thống 3 GPU.
- **Thiết lập `CUDA_VISIBLE_DEVICES`**: Tự động cấu hình để sử dụng cả 3 GPU cho một mô hình đơn.
- **Giảm offload CPU**: Giảm lượng bộ nhớ dành cho CPU offload khi có 3 GPU để tăng hiệu quả.

### 3. Tăng Tốc Inference

- **Tích hợp `torch.compile()`**: Tự động sử dụng torch.compile() nếu phiên bản PyTorch hỗ trợ để tăng tốc inference.
- **Xử lý theo batch**: Cải tiến phương thức đánh giá tuần tự để xử lý câu hỏi theo batch, giúp tăng hiệu suất và hiển thị ước tính thời gian còn lại.
- **Lưu checkpoint theo batch**: Lưu kết quả sau mỗi batch thay vì mỗi `checkpoint_frequency` câu hỏi.

### 4. Quản lý Lỗi Rate Limit và Quotas

- **Circuit Breaker Pattern**: Thêm cơ chế "circuit breaker" để tạm ngưng gọi API khi gặp quá nhiều lỗi rate limit liên tiếp, giúp tránh các request vô ích và giảm áp lực lên API.
- **Adaptive Rate Limiting**: Tự động điều chỉnh khoảng thời gian giữa các request dựa trên phản hồi từ API, tăng khoảng cách khi gặp lỗi và giảm dần về bình thường khi thành công.
- **Phân tích Thông minh Lỗi API**: Trích xuất thời gian retry từ thông báo lỗi của API để chờ đợi một cách chính xác.
- **Cấu hình Theo API**: Thiết lập các tham số rate limit và circuit breaker khác nhau cho từng API (Groq, Gemini) dựa trên đặc điểm của từng dịch vụ.

## Sử Dụng

### Chạy Đánh Giá Với Tối Ưu Mới

```bash
# Sử dụng mặc định batch size 15
python main.py --models llama --prompts zero_shot few_shot_3 --gpu-ids 0 1 2

# Chỉ định batch size cao hơn (nếu bộ nhớ cho phép)
python main.py --models llama --prompts zero_shot few_shot_3 --batch-size 20 --gpu-ids 0 1 2
```

### Theo Dõi Hiệu Suất

Để theo dõi việc sử dụng bộ nhớ và phát hiện vấn đề sớm, hãy sử dụng flag `--debug`:

```bash
python main.py --models llama --prompts zero_shot --batch-size 15 --gpu-ids 0 1 2 --debug
```

## Khắc phục sự cố

Nếu gặp lỗi OOM (Out of Memory):

1. Giảm batch size
2. Kiểm tra với nvidia-smi trong quá trình chạy để theo dõi việc sử dụng bộ nhớ GPU
3. Chuyển về các mô hình nhỏ hơn hoặc sử dụng quantization mạnh hơn

Nếu gặp lỗi rate limit khi sử dụng API:

1. Hệ thống sẽ tự động xử lý với các cơ chế retry và circuit breaker
2. Nếu vẫn gặp vấn đề, bạn có thể điều chỉnh thông số trong `config.py`:
   ```python
   API_CONFIGS = {
       "groq": {
           "requests_per_minute": 20,  # Giảm số lượng request mỗi phút
           "circuit_breaker": {
               "failure_threshold": 3,  # Giảm ngưỡng lỗi để kích hoạt circuit breaker sớm hơn
               "cooldown_period": 90,  # Tăng thời gian nghỉ 
           }
       }
   }
   ```

## Lưu ý

- Các cải tiến này được tối ưu cho hệ thống có 3 GPU RTX 6000 Ada Generation, mỗi GPU có khoảng 48GB VRAM.
- Hiệu suất có thể khác nhau tùy thuộc vào mô hình, loại prompt, và độ phức tạp của câu hỏi.
- Đã thêm các log thông tin để dễ dàng theo dõi tiến trình và hiệu suất. 