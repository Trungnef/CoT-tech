# Phân tích lỗi trong LLM Evaluation

## Giới thiệu

Phân tích lỗi (Error Analysis) là một tính năng quan trọng giúp phân loại và hiểu rõ hơn về các loại lỗi mà các mô hình ngôn ngữ lớn (LLM) thường gặp phải. Thay vì chỉ biết mô hình trả lời sai, tính năng này giúp xác định *tại sao* mô hình trả lời sai và lỗi thuộc loại nào.

Tính năng này giúp:
- Phân loại lỗi thành các nhóm có ý nghĩa như: Lỗi kiến thức, Lỗi suy luận, Lỗi tính toán, v.v.
- Tạo báo cáo trực quan về phân bố các loại lỗi
- So sánh các mô hình khác nhau dựa trên loại lỗi chúng gặp phải
- Cung cấp thông tin chi tiết để cải thiện mô hình và prompt

## Các loại lỗi

Hệ thống phân loại lỗi mặc định bao gồm các loại sau:

| Loại lỗi | Mô tả |
|----------|-------|
| **Knowledge Error** | Mô hình không có kiến thức cần thiết để trả lời câu hỏi |
| **Reasoning Error** | Mô hình có kiến thức nhưng suy luận không đúng |
| **Calculation Error** | Lỗi trong các phép tính toán |
| **Misunderstanding** | Mô hình hiểu sai câu hỏi hoặc yêu cầu |
| **Off-topic** | Câu trả lời không liên quan đến câu hỏi |
| **Non-answer** | Mô hình từ chối trả lời hoặc đưa ra câu trả lời không rõ ràng |
| **Other** | Các lỗi khác không thuộc các nhóm trên |

## Cách sử dụng

### Sử dụng qua ResultAnalyzer

```python
from llm_evaluation.core.result_analyzer import ResultAnalyzer

# Khởi tạo analyzer với DataFrame chứa kết quả đánh giá
analyzer = ResultAnalyzer(results_df=results)

# Phân tích lỗi
# sample_size: Số lượng lỗi tối đa cần phân tích (để tránh quá nhiều API calls)
results_with_errors = analyzer.analyze_errors(results, sample_size=100)

# Kiểm tra kết quả phân tích
error_types = results_with_errors[results_with_errors['error_type'] != '']['error_type'].value_counts()
print("Phân bố các loại lỗi:")
print(error_types)
```

### Tạo báo cáo với kết quả phân tích lỗi

```python
from llm_evaluation.core.reporting import Reporting

# Khởi tạo reporting với DataFrame đã có kết quả phân tích lỗi
reporter = Reporting(results_df=results_with_errors, output_dir="./outputs")

# Tạo báo cáo
report_files = reporter.generate_reports()
```

## Cơ chế hoạt động

1. **Phát hiện lỗi**: Hệ thống xác định các câu trả lời sai dựa trên cột `is_correct`
2. **Phân tích lỗi**: Với mỗi câu trả lời sai, hệ thống sẽ:
   - Phân tích câu trả lời và câu hỏi để xác định loại lỗi
   - Thêm hai cột mới `error_type` và `error_explanation` vào DataFrame
3. **Tính toán số liệu thống kê**: Tính toán tỷ lệ phần trăm của từng loại lỗi
4. **Trực quan hóa**: Tạo các biểu đồ trực quan về phân bố lỗi

### Lưu ý về API sử dụng

Mặc định, tính năng phân tích lỗi sử dụng các mô hình LLM để phân loại lỗi. Điều này có thể dẫn đến chi phí API nếu số lượng lỗi cần phân tích là lớn. Vì vậy, luôn sử dụng tham số `sample_size` để giới hạn số lượng lỗi được phân tích.

## Ví dụ đầy đủ

Xem file ví dụ tại `examples/error_analysis_demo.py` để thấy tính năng phân tích lỗi hoạt động trong thực tế.

## Tùy chỉnh phân loại lỗi

Bạn có thể tùy chỉnh các loại lỗi hoặc cách phân loại bằng cách:

1. **Tùy chỉnh prompt phân tích**: Sửa đổi phương thức `_build_error_analysis_prompt` trong `ResultAnalyzer`
2. **Sử dụng mô hình phân loại của riêng bạn**: Ghi đè phương thức `_analyze_single_error`

Ví dụ tùy chỉnh prompt:

```python
class CustomErrorAnalyzer(ResultAnalyzer):
    def _build_error_analysis_prompt(self, question, response, correct_answer):
        # Tùy chỉnh prompt cho phân tích lỗi
        return f"""
        Phân tích lỗi trong câu trả lời sau:
        
        Câu hỏi: {question}
        Câu trả lời của mô hình: {response}
        Câu trả lời đúng: {correct_answer}
        
        Hãy phân loại lỗi này thành một trong các loại sau:
        - Technical Error: Lỗi về kỹ thuật hoặc liên quan đến công nghệ
        - Domain Error: Lỗi liên quan đến lĩnh vực cụ thể
        - Logical Error: Lỗi về logic hoặc suy luận
        - Factual Error: Lỗi về sự kiện hoặc dữ kiện
        - Context Error: Lỗi do thiếu ngữ cảnh
        
        Trả về kết quả dưới dạng JSON với cấu trúc:
        {{"error_type": "<loại lỗi>", "explanation": "<giải thích ngắn gọn>"}}
        """
```

## Tích hợp với các công cụ khác

Kết quả phân tích lỗi có thể được tích hợp với các công cụ khác như:

1. **Hệ thống theo dõi thử nghiệm**: Lưu kết quả phân tích lỗi vào cơ sở dữ liệu
2. **Công cụ cải thiện prompt**: Sử dụng thông tin về các loại lỗi để tối ưu hóa prompt
3. **Hệ thống huấn luyện**: Sử dụng kết quả phân tích để tạo dữ liệu huấn luyện tập trung vào các loại lỗi cụ thể 