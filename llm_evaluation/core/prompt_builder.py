"""
Prompt Builder cho hệ thống đánh giá LLM.
Cung cấp các hàm tạo prompt với nhiều chiến lược khác nhau:
- Zero-shot: Chỉ cung cấp câu hỏi, không có ví dụ
- Few-shot: Cung cấp một số ví dụ (3, 5, 7) trước câu hỏi
- Chain of Thought (CoT): Yêu cầu mô hình giải thích từng bước suy luận
- Self-consistency: Yêu cầu mô hình tạo nhiều lời giải khác nhau và chọn kết quả phổ biến nhất
- ReAct: Kết hợp Reasoning và Acting trong quá trình suy luận
"""

import re
import random
from typing import List, Dict, Any, Tuple, Optional, Union

# Thêm các ví dụ mặc định cho few-shot learning
DEFAULT_EXAMPLES = [
    # Bài toán công việc
    {
        "question": "Thợ A làm một công việc hết 6 giờ, thợ B làm hết 4 giờ. Hỏi nếu hai thợ cùng làm thì sau bao lâu sẽ hoàn thành công việc?",
        "answer": "Trong 1 giờ, thợ A làm được 1/6 công việc, thợ B làm được 1/4 công việc. Hai thợ cùng làm trong 1 giờ được: 1/6 + 1/4 = 5/12 công việc. Thời gian cần: 1 ÷ (5/12) = 12/5 = 2.4 giờ."
    },
    # Bài toán chia kẹo
    {
        "question": "Có 143 viên kẹo chia cho 5 người. Người thứ hai được gấp 2 lần người thứ nhất, người thứ ba được gấp 2 lần người thứ hai. Hỏi mỗi người được bao nhiêu viên kẹo?",
        "answer": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + 2x + 4x + người thứ 4 + người thứ 5 = 143. Ta chưa biết số kẹo của người 4 và 5, giả sử bằng nhau và bằng x. Khi đó: x + 2x + 4x + x + x = 9x = 143. Giải ra: x = 143/9 ≈ 15.89, không hợp lý. Giả sử người 4 và 5 đều nhận x viên như người 1: x + 2x + 4x + x + x = 9x = 143. Vậy x = 143/9 ≈ 15.89, làm tròn x = 16. Kiểm tra: 16 + 32 + 64 + 16 + 16 = 144 > 143, nên x = 15.89 không chính xác. Vậy người 1, 4, 5 mỗi người nhận ít hơn 16 viên."
    },
    # Bài toán hình học
    {
        "question": "Cho hình chữ nhật có chiều dài 17cm, chiều rộng 14cm. Tính diện tích và chu vi của hình chữ nhật.",
        "answer": "Diện tích = dài × rộng = 17 × 14 = 238cm². Chu vi = 2 × (dài + rộng) = 2 × (17 + 14) = 2 × 31 = 62cm."
    },
    # Bài toán số học
    {
        "question": "Tìm số tự nhiên có 3 chữ số, biết rằng tổng các chữ số là 6 và tích các chữ số là 6.",
        "answer": "Gọi các chữ số là a, b, c (với a ≠ 0). Ta có: a + b + c = 6 và a × b × c = 6. Xét các cách phân tích 6 thành tích của 3 số: 6 = 1×2×3 = 1×1×6 = 2×1×3. Thử với a, b, c = 1, 2, 3: 1 + 2 + 3 = 6 ✓. Số cần tìm: 123."
    },
    # Bài toán chuyển động
    {
        "question": "Hai xe xuất phát cùng lúc từ A và B cách nhau 276 km. Xe thứ nhất đi với vận tốc 55 km/h, xe thứ hai đi với vận tốc 43 km/h. Hỏi sau bao lâu hai xe gặp nhau?",
        "answer": "Gọi thời gian gặp nhau là t giờ. Khoảng cách hai xe đi được bằng khoảng cách AB: 55t + 43t = 276. Giải ra: t = 276/(55+43) = 276/98 = 2.82 giờ."
    },
    # Bài toán về tuổi
    {
        "question": "Hiện nay tuổi cha là 46 tuổi, tuổi con là 10 tuổi. Hỏi sau bao nhiêu năm nữa thì tuổi cha gấp 3 lần tuổi con?",
        "answer": "Gọi số năm cần tìm là x. Ta có: (46+x)/(10+x)=3. Giải ra: (46+x) = 3(10+x) → 46+x = 30+3x → x = 8 năm."
    },
    # Bài toán phân số
    {
        "question": "Cho hai phân số 5/7 và 3/4. Tìm tổng và tích của hai phân số này.",
        "answer": "Quy đồng mẫu số: 5/7 = 20/28, 3/4 = 21/28. Tổng = 20/28 + 21/28 = 41/28. Tích = (5/7) × (3/4) = 15/28."
    },
    # Bài toán hồ bơi
    {
        "question": "Một hồ bơi có 4588 m³ nước. Có 5 ống nước chảy vào với lưu lượng 74 m³/giờ và 1 ống thoát với lưu lượng 36 m³/giờ. Hỏi sau bao lâu hồ sẽ đầy?",
        "answer": "Lưu lượng nước vào mỗi giờ: 5×74=370 m³/giờ. Lưu lượng nước ra mỗi giờ: 1×36=36 m³/giờ. Lưu lượng thực tế: 370-36=334 m³/giờ. Thời gian cần: 4588/334=13.74 giờ."
    },
    # Bài toán hỗn hợp
    {
        "question": "Hòa tan 441g muối vào 1748ml nước được dung dịch 20.15%. Hỏi phải thêm bao nhiêu gam muối nữa để được dung dịch 27.15%?",
        "answer": "Khối lượng muối ban đầu: 441g. Khối lượng dung dịch: 441 + 1748 = 2189g. Gọi số gam muối cần thêm là x. Ta có: (441 + x)/(2189 + x) = 27.15/100. Giải ra: (441 + x)/(2189 + x) = 0.2715 → 441 + x = 0.2715(2189 + x) → 441 + x = 594.3 + 0.2715x → 0.7285x = 153.3 → x = 210.45g."
    },
    # Câu đố logic
    {
        "question": "Một người nông dân có 18 con vật gồm gà và thỏ. Đếm tổng số chân thì thấy có 48 cái chân. Hỏi người nông dân có bao nhiêu con gà và bao nhiêu con thỏ?",
        "answer": "Gọi số gà là x con. Khi đó số thỏ là 18 - x con. Số chân gà: 2x. Số chân thỏ: 4(18 - x). Theo đề bài: 2x + 4(18 - x) = 48. Giải ra: 2x + 72 - 4x = 48 → -2x = -24 → x = 12 con gà. Vậy có 6 con thỏ."
    },
    # Câu hỏi trắc nghiệm
    {
        "question": "Kết quả của phép tính 46 + 7 × 9 là bao nhiêu?\nA. 114\nB. 109\nC. 104\nD. 2898",
        "answer": "Theo thứ tự ưu tiên, ta thực hiện phép nhân trước: 7 × 9 = 63. Sau đó cộng với 46: 46 + 63 = 109. Đáp án đúng là B. 109."
    },
    # Thơ toán học
    {
        "question": "Có 88 quả cam ngon,\nChia đều cho 6 người em.\nMỗi người được mấy quả?\nCòn dư mấy quả cam?",
        "answer": "Số cam chia được: 88 ÷ 6 = 14 quả mỗi người. Số cam còn dư: 88 - (6 × 14) = 88 - 84 = 4 quả."
    },
    # Bài toán từ vựng toán học
    {
        "question": "Một cửa hàng bán thước với giá 5385đ một quyển. Nếu mua 49 quyển, sau đó được giảm 29%, hỏi phải trả bao nhiêu tiền?",
        "answer": "Giá gốc: 5385 × 49 = 263865đ. Số tiền được giảm: 263865 × 29/100 = 76520đ. Số tiền phải trả: 263865 - 76520 = 187345đ."
    },
    # Câu hỏi kiểu bài luận
    {
        "question": "Giải thích tại sao khi chia một số cho 9, nếu tổng các chữ số của số đó chia hết cho 9 thì số đó cũng chia hết cho 9?",
        "answer": "Để chứng minh điều này, ta xét một số N có n chữ số: N = a₁×10^(n-1) + a₂×10^(n-2) + ... + aₙ. Khi chia 10^k cho 9, ta luôn được dư 1. Do đó, N ≡ a₁ + a₂ + ... + aₙ (mod 9). Vì vậy, nếu tổng các chữ số chia hết cho 9 thì N cũng chia hết cho 9."
    }
]

class PromptBuilder:
    """
    Lớp xây dựng prompt cho việc đánh giá mô hình ngôn ngữ.
    Hỗ trợ nhiều chiến lược tạo prompt khác nhau cho bài toán Tiếng Việt.
    """
    
    def __init__(self, 
                 system_message: str = "Bạn là một trợ lý AI giỏi giải toán.",
                 language: str = "vietnamese"):
        """
        Khởi tạo PromptBuilder.
        
        Args:
            system_message (str): Thông điệp hệ thống để đặt vai trò của mô hình
            language (str): Ngôn ngữ sử dụng ("vietnamese" hoặc "english")
        """
        self.system_message = system_message
        self.language = language.lower()
        
        # Cấu trúc mở đầu và kết thúc prompt dựa trên ngôn ngữ
        self.prompt_frames = {
            "vietnamese": {
                "zero_shot": {
                    "prefix": "Hãy giải bài toán sau:\n\n",
                    "suffix": "\n\nĐáp án:"
                },
                "few_shot": {
                    "prefix": "Dưới đây là một số ví dụ về cách giải các bài toán tương tự. Hãy giải bài toán cuối cùng:\n\n",
                    "example_separator": "\n\n---\n\n",
                    "question_prefix": "Bài toán: ",
                    "answer_prefix": "Đáp án: ",
                    "suffix": "\n\nĐáp án:"
                },
                "cot": {
                    "prefix": "Hãy giải bài toán sau đây. Hãy lập luận từng bước để tìm ra đáp án chính xác:\n\n",
                    "suffix": "\n\nHãy lập luận từng bước:\n"
                },
                "self_consistency": {
                    "prefix": "Hãy giải bài toán sau đây bằng NHIỀU cách tiếp cận khác nhau để kiểm tra tính nhất quán của kết quả:\n\n",
                    "suffix": "\n\nHãy cung cấp {count} cách giải khác nhau, mỗi cách đều phải trình bày từng bước logic và kết quả cuối cùng."
                },
                "react": {
                    "prefix": "Hãy giải bài toán sau đây. Sử dụng phương pháp ReAct (Reasoning and Acting):\n\n",
                    "suffix": "\n\nHãy giải bài toán theo định dạng sau:\nSuy nghĩ 1: [Suy nghĩ của bạn về bài toán]\nHành động 1: [Hành động bạn thực hiện để giải quyết]\nKết quả 1: [Kết quả của hành động]\n... (lặp lại cho đến khi tìm ra đáp án)\nĐáp án cuối cùng: [Đáp án]"
                }
            },
            "english": {
                "zero_shot": {
                    "prefix": "Solve the following problem:\n\n",
                    "suffix": "\n\nAnswer:"
                },
                "few_shot": {
                    "prefix": "Here are some examples of how to solve similar problems. Please solve the final problem:\n\n",
                    "example_separator": "\n\n---\n\n",
                    "question_prefix": "Problem: ",
                    "answer_prefix": "Answer: ",
                    "suffix": "\n\nAnswer:"
                },
                "cot": {
                    "prefix": "Solve the following problem. Reason step-by-step to find the correct answer:\n\n",
                    "suffix": "\n\nLet's think step-by-step:\n"
                },
                "self_consistency": {
                    "prefix": "Solve the following problem using MULTIPLE different approaches to check the consistency of the result:\n\n",
                    "suffix": "\n\nPlease provide {count} different solution methods, each with step-by-step reasoning and a final result."
                },
                "react": {
                    "prefix": "Solve the following problem. Use the ReAct (Reasoning and Acting) approach:\n\n",
                    "suffix": "\n\nSolve the problem using this format:\nThought 1: [Your reasoning about the problem]\nAction 1: [The action you take to solve it]\nResult 1: [The result of your action]\n... (repeat until you find the answer)\nFinal Answer: [Answer]"
                }
            }
        }
    
    def create_prompt(self, 
                      question: str, 
                      prompt_type: str, 
                      examples: Optional[List[Dict[str, str]]] = None,
                      count: int = 3) -> str:
        """
        Tạo prompt dựa theo loại prompt được chỉ định.
        
        Args:
            question (str): Câu hỏi cần trả lời
            prompt_type (str): Loại prompt (zero_shot, few_shot_3, few_shot_5, few_shot_7, 
                                          cot, self_consistency_3, self_consistency_5, 
                                          self_consistency_7, react)
            examples (List[Dict]): Danh sách các ví dụ (mỗi ví dụ chứa 'question' và 'answer')
            count (int): Số lượng cách tiếp cận cho self-consistency
            
        Returns:
            str: Prompt đã được tạo
        """
        # Xác định ngôn ngữ sử dụng
        lang = "vietnamese" if self.language == "vietnamese" else "english"
        frames = self.prompt_frames[lang]
        
        # Xử lý các loại prompt khác nhau
        if prompt_type == "zero_shot":
            return self._create_zero_shot_prompt(question, frames["zero_shot"])
        
        elif prompt_type.startswith("few_shot_"):
            # Lấy số lượng ví dụ từ tên prompt
            num_examples = int(prompt_type.split("_")[-1])
            
            # Sử dụng ví dụ mặc định nếu không có ví dụ được cung cấp hoặc không đủ số lượng
            if not examples or len(examples) < num_examples:
                if not examples:
                    examples = DEFAULT_EXAMPLES.copy()
                elif len(examples) < num_examples:
                    # Bổ sung thêm các ví dụ mặc định nếu thiếu
                    additional_examples = [ex for ex in DEFAULT_EXAMPLES if all(e.get('question') != ex.get('question') for e in examples)]
                    examples = examples + additional_examples
                
                if len(examples) < num_examples:
                    # Nếu vẫn không đủ, đưa ra cảnh báo nhưng vẫn sử dụng số lượng hiện có
                    print(f"Cảnh báo: Chỉ có {len(examples)} ví dụ cho few-shot với {num_examples} shot")
                
            return self._create_few_shot_prompt(
                question, frames["few_shot"], examples, min(num_examples, len(examples))
            )
        
        elif prompt_type == "cot":
            return self._create_cot_prompt(question, frames["cot"])
        
        elif prompt_type.startswith("self_consistency_") or prompt_type.startswith("cot_self_consistency_"):
            # Lấy số lượng cách tiếp cận từ tên prompt
            count = int(prompt_type.split("_")[-1])
            return self._create_self_consistency_prompt(
                question, frames["self_consistency"], count
            )
        
        elif prompt_type == "react":
            return self._create_react_prompt(question, frames["react"])
        
        else:
            raise ValueError(f"Loại prompt không được hỗ trợ: {prompt_type}")
    
    def _create_zero_shot_prompt(self, question: str, frame: Dict[str, str]) -> str:
        """
        Tạo zero-shot prompt, chỉ bao gồm câu hỏi.
        
        Args:
            question (str): Câu hỏi cần trả lời
            frame (Dict): Cấu trúc prompt
            
        Returns:
            str: Zero-shot prompt
        """
        return f"{self.system_message}\n\n{frame['prefix']}{question}{frame['suffix']}"
    
    def _create_few_shot_prompt(self, 
                               question: str, 
                               frame: Dict[str, str],
                               examples: List[Dict[str, str]], 
                               num_examples: int) -> str:
        """
        Tạo few-shot prompt, bao gồm một số ví dụ và câu hỏi.
        
        Args:
            question (str): Câu hỏi cần trả lời
            frame (Dict): Cấu trúc prompt
            examples (List[Dict]): Danh sách các ví dụ (mỗi ví dụ chứa 'question' và 'answer')
            num_examples (int): Số lượng ví dụ cần sử dụng
            
        Returns:
            str: Few-shot prompt
        """
        # Chọn ngẫu nhiên số lượng ví dụ cần thiết
        selected_examples = random.sample(examples, min(num_examples, len(examples)))
        
        # Tạo phần ví dụ
        examples_text = []
        for ex in selected_examples:
            example_str = f"{frame['question_prefix']}{ex['question']}\n{frame['answer_prefix']}{ex['answer']}"
            examples_text.append(example_str)
        
        # Tạo phần câu hỏi cuối cùng cần trả lời
        final_question = f"{frame['question_prefix']}{question}"
        
        # Kết hợp tất cả
        prompt = (f"{self.system_message}\n\n{frame['prefix']}" + 
                  f"{frame['example_separator'].join(examples_text)}" +
                  f"{frame['example_separator']}{final_question}{frame['suffix']}")
        
        return prompt
    
    def _create_cot_prompt(self, question: str, frame: Dict[str, str]) -> str:
        """
        Tạo Chain-of-Thought prompt, yêu cầu mô hình lập luận từng bước.
        
        Args:
            question (str): Câu hỏi cần trả lời
            frame (Dict): Cấu trúc prompt
            
        Returns:
            str: Chain-of-Thought prompt
        """
        return f"{self.system_message}\n\n{frame['prefix']}{question}{frame['suffix']}"
    
    def _create_self_consistency_prompt(self, 
                                       question: str, 
                                       frame: Dict[str, str],
                                       count: int) -> str:
        """
        Tạo Self-Consistency prompt, yêu cầu mô hình đưa ra nhiều cách tiếp cận.
        
        Args:
            question (str): Câu hỏi cần trả lời
            frame (Dict): Cấu trúc prompt
            count (int): Số lượng cách tiếp cận
            
        Returns:
            str: Self-Consistency prompt
        """
        suffix = frame['suffix'].format(count=count)
        return f"{self.system_message}\n\n{frame['prefix']}{question}{suffix}"
    
    def _create_react_prompt(self, question: str, frame: Dict[str, str]) -> str:
        """
        Tạo ReAct prompt, kết hợp Reasoning và Acting.
        
        Args:
            question (str): Câu hỏi cần trả lời
            frame (Dict): Cấu trúc prompt
            
        Returns:
            str: ReAct prompt
        """
        return f"{self.system_message}\n\n{frame['prefix']}{question}{frame['suffix']}"
    
    def extract_final_answer(self, response: str, prompt_type: str) -> str:
        """
        Trích xuất câu trả lời cuối cùng từ phản hồi của mô hình.
        
        Args:
            response (str): Phản hồi từ mô hình
            prompt_type (str): Loại prompt đã sử dụng
            
        Returns:
            str: Câu trả lời cuối cùng
        """
        # Xử lý khác nhau dựa trên loại prompt
        if prompt_type == "zero_shot":
            # Đối với zero-shot, câu trả lời thường là toàn bộ phản hồi hoặc phần sau "Đáp án:"
            match = re.search(r"đáp án:?\s*(.*?)$", response.lower(), re.DOTALL)
            if match:
                return match.group(1).strip()
            return response.strip()
        
        elif prompt_type.startswith("few_shot_"):
            # Đối với few-shot, tìm câu trả lời sau "Đáp án:" ở cuối
            match = re.search(r"đáp án:?\s*(.*?)$", response.lower(), re.DOTALL)
            if match:
                return match.group(1).strip()
            return response.strip()
        
        elif prompt_type == "cot":
            # Đối với CoT, tìm câu trả lời cuối cùng sau các bước lập luận
            # Thường là phần sau "Vậy đáp án là" hoặc "Kết quả là" hoặc "Đáp án:"
            patterns = [
                r"vậy đáp án là:?\s*(.*?)$",
                r"kết quả là:?\s*(.*?)$", 
                r"đáp án:?\s*(.*?)$",
                r"vậy?[,\s]*kết quả:?\s*(.*?)$",
                r"vậy?[,\s]*kết luận:?\s*(.*?)$",
                r"kết luận:?\s*(.*?)$"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response.lower(), re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # Nếu không tìm thấy, lấy câu cuối cùng
            sentences = re.split(r'[.!?]', response)
            return sentences[-1].strip()
        
        elif prompt_type.startswith("self_consistency_"):
            # Đối với self-consistency, tìm kết quả phổ biến nhất
            # Tìm tất cả đáp án trong các cách tiếp cận khác nhau
            answers = []
            
            # Tìm các đáp án trong mỗi cách giải
            patterns = [
                r"đáp án:?\s*(.*?)(?:\n|$)",
                r"kết quả:?\s*(.*?)(?:\n|$)",
                r"vậy đáp án là:?\s*(.*?)(?:\n|$)"
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, response.lower(), re.DOTALL)
                for match in matches:
                    answers.append(match.group(1).strip())
            
            if answers:
                # Tìm đáp án phổ biến nhất
                answer_count = {}
                for ans in answers:
                    if ans in answer_count:
                        answer_count[ans] += 1
                    else:
                        answer_count[ans] = 1
                
                # Trả về đáp án có tần suất xuất hiện nhiều nhất
                return max(answer_count.items(), key=lambda x: x[1])[0]
            
            # Nếu không tìm thấy đáp án rõ ràng, trả về câu cuối cùng
            return response.strip().split('\n')[-1]
        
        elif prompt_type == "react":
            # Đối với ReAct, tìm "Đáp án cuối cùng"
            match = re.search(r"đáp án cuối cùng:?\s*(.*?)(?:\n|$)", response.lower(), re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Hoặc tìm kết quả cuối cùng
            match = re.search(r"kết quả cuối cùng:?\s*(.*?)(?:\n|$)", response.lower(), re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Nếu không tìm thấy, tìm bất kỳ đáp án nào
            match = re.search(r"đáp án:?\s*(.*?)(?:\n|$)", response.lower(), re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Nếu tất cả đều thất bại, trả về câu cuối cùng
            return response.strip().split('\n')[-1]
        
        else:
            # Trường hợp mặc định
            return response.strip()

# Hàm wrapper ở cấp module để tương thích với import statement trong evaluator.py
def create_prompt(query, prompt_type, task_type=None, custom_examples=None):
    """
    Hàm wrapper cho PromptBuilder.create_prompt để tương thích với các module khác.
    
    Args:
        query (str): Câu hỏi cần trả lời
        prompt_type (str): Loại prompt (zero_shot, few_shot_3, few_shot_5, few_shot_7, etc.)
        task_type (str, optional): Loại nhiệm vụ, không sử dụng trong PromptBuilder hiện tại
        custom_examples (List[Dict], optional): Các ví dụ tùy chỉnh cho few-shot prompts
        
    Returns:
        str: Prompt đã được tạo
    """
    builder = PromptBuilder()
    return builder.create_prompt(
        question=query,
        prompt_type=prompt_type,
        examples=custom_examples
    )
