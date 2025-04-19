# Hướng dẫn Quản lý Nhiều API Keys

Tài liệu này hướng dẫn cách cấu hình và sử dụng nhiều API keys để tăng khả năng xử lý và tránh gián đoạn khi các API keys riêng lẻ gặp vấn đề về giới hạn tỷ lệ hoặc hết quota.

## Cấu hình API Keys

### Định dạng trong File .env

API keys được cấu hình trong file `.env` với định dạng danh sách các keys cách nhau bởi dấu phẩy:

```
GEMINI_API_KEYS="key1, key2, key3"
GROQ_API_KEYS="key1, key2, key3, key4"
OPENAI_API_KEYS="key1, key2"
```

### Ví dụ Cấu Hình

```
# API KEYS
GEMINI_API_KEYS="AIzaSyD1234567890abcdef, AIzaSyD0987654321zyxwvu"
OPENAI_API_KEYS="sk-1234567890abcdef1234, sk-0987654321zyxwvu5678"  
GROQ_API_KEYS="gsk_abcdef1234567890, gsk_zyxwvu0987654321, gsk_qwerty1234567890"
```

### Biến Tương Thích Ngược

Hệ thống vẫn hỗ trợ các biến API key đơn lẻ để đảm bảo tương thích với code cũ:

```
GEMINI_API_KEY="AIzaSyD1234567890abcdef"
OPENAI_API_KEY="sk-1234567890abcdef1234"
GROQ_API_KEY="gsk_abcdef1234567890"
```

Nếu cả biến đơn lẻ và danh sách được thiết lập, hệ thống sẽ ưu tiên sử dụng danh sách.

## Cơ Chế Xử Lý Lỗi và Rotation

Hệ thống cung cấp các cơ chế tự động sau:

1. **Key Rotation**: Khi một key gặp lỗi quota hoặc rate limit, hệ thống tự động chuyển sang key tiếp theo.
2. **Theo dõi Key đã hết Quota**: Hệ thống theo dõi các key đã hết quota để tránh sử dụng lại.
3. **Auto-Reset**: Khi tất cả các key đều đã hết quota, hệ thống sẽ reset danh sách key đã thử để bắt đầu lại.
4. **Phát hiện loại lỗi quota**: Hệ thống phân biệt giữa lỗi rate limit tạm thời và lỗi hết quota theo ngày.
5. **Làm mới key theo ngày**: Keys bị đánh dấu là hết quota theo ngày sẽ tự động được làm mới sau khi sang ngày mới.

### Quản lý Quota theo Ngày

Nhiều API như Gemini và Groq áp dụng giới hạn quota theo ngày. Khi một key hết quota theo ngày:

1. Key đó được đánh dấu là `daily_quota_exceeded` với timestamp hiện tại
2. Hệ thống chuyển sang sử dụng key tiếp theo
3. Khi sang ngày mới (00:00 UTC), key sẽ tự động được làm mới và có thể sử dụng lại
4. Hệ thống kiểm tra định kỳ (mỗi 1 giờ) để làm mới các key đã hết quota theo ngày

## Ví Dụ Mã Xử Lý Lỗi

```python
# Khi gặp lỗi quota hoặc rate limit
if 'quota' in str(error).lower() or 'rate limit' in str(error).lower():
    # Xác định loại lỗi quota
    is_daily_quota = False
    reason = "rate_limit_exceeded"
    
    # Phân tích nội dung lỗi để xác định đúng loại lỗi quota
    error_str = str(error).lower()
    if 'daily' in error_str and 'quota' in error_str:
        is_daily_quota = True
        reason = "daily_quota_exceeded"
        
    # Lưu thông tin hết hạn với timestamp hiện tại
    self.exhausted_keys[current_key] = {
        'timestamp': time.time(),
        'reason': reason,
        'error': str(error)
    }
    
    # Chuyển sang key tiếp theo
    self.current_key_index = (self.current_key_index + 1) % len(keys)
```

## Cách Hệ thống Phát hiện Quota theo Ngày

Hệ thống sử dụng phân tích nội dung lỗi để phát hiện khi một key hết quota theo ngày:

1. Tìm các từ khóa như "daily quota", "quota exceeded", "quota limit"
2. Nếu phát hiện, đánh dấu key với lý do `daily_quota_exceeded`
3. Lưu timestamp để tính toán khi key có thể sử dụng lại

## Kiểm Tra Thành Công

Để kiểm tra chức năng key rotation và làm mới theo ngày:

```bash
python test_key_rotation.py
```

Output sẽ hiển thị danh sách các keys được cấu hình, quá trình chuyển đổi key khi cần, và mô phỏng việc làm mới key theo ngày.

## Best Practices

1. **Sử dụng Nhiều Keys khác nhau**: Nên sử dụng nhiều API keys tạo từ các tài khoản riêng biệt để tối đa hóa quota
2. **Quản lý Keys**: Giữ keys an toàn và định kỳ kiểm tra tính khả dụng
3. **Theo dõi Sử dụng**: Thêm monitoring để biết key nào đang được sử dụng và khi nào hết quota
4. **Đa dạng hóa Keys**: Sử dụng keys từ các tài khoản khác nhau để đảm bảo luân phiên hoạt động liên tục

## Xử Lý Sự Cố

- **Tất cả keys đều hết quota**: Hệ thống sẽ hiển thị thông báo và tiếp tục cố gắng với key đầu tiên
- **API key không hợp lệ**: Key đó sẽ bị đánh dấu và hệ thống chuyển sang key tiếp theo
- **Lỗi rate limit tạm thời**: Hệ thống sẽ chuyển sang key khác và quay lại key bị hạn chế sau một thời gian
- **Lỗi hết quota theo ngày**: Hệ thống sẽ chuyển sang key khác và chỉ quay lại key đó khi sang ngày mới

## Logs & Thông Báo

Hệ thống ghi log chi tiết các hành động:

```
[2025-04-19 05:36:33] [INFO] [model_interface] Key #1 đã hết quota theo ngày. Chuyển sang key #2/3
[2025-04-19 12:00:05] [INFO] [model_interface] Key AIzaSyD123... đã sang ngày mới, có thể sử dụng lại
```

Theo dõi logs để biết khi nào cần thêm API keys mới hoặc khi có các vấn đề khác xảy ra. 