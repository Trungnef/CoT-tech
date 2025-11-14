"""
Prompt Builder cho hệ thống đánh giá LLM.
Cung cấp các hàm tạo prompt với nhiều chiến lược khác nhau:
- Zero-shot: Chỉ cung cấp câu hỏi, không có ví dụ
- Few-shot: Cung cấp một số ví dụ (3, 5, 7) trước câu hỏi. **REQUIRED**: Phải cung cấp examples từ dữ liệu (problems.json)
- Chain of Thought (CoT): Yêu cầu mô hình giải thích từng bước suy luận
- Self-consistency: Yêu cầu mô hình tạo nhiều lời giải khác nhau và chọn kết quả phổ biến nhất
- ReAct: Kết hợp Reasoning và Acting trong quá trình suy luận

IMPROVEMENTS (Nov 15, 2025):
1. Removed DEFAULT_EXAMPLES - Few-shot prompts now REQUIRE explicit examples from user data
2. Fixed extract_final_answer to preserve original casing while doing case-insensitive matching
3. Improved self-consistency answer extraction: correctly identifies most common answer

See IMPROVEMENTS.md for detailed changelog.
"""

import re
import random
from typing import List, Dict, Any, Tuple, Optional, Union

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
            examples (List[Dict]): Danh sách các ví dụ cho few-shot prompts (mỗi ví dụ chứa 'question' và 'answer').
                                  Bắt buộc cho few-shot prompts. Ví dụ nên được cung cấp từ dữ liệu problems.json.
            count (int): Số lượng cách tiếp cận cho self-consistency
            
        Returns:
            str: Prompt đã được tạo
            
        Raises:
            ValueError: Nếu few-shot prompt được yêu cầu nhưng không có ví dụ được cung cấp
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
            
            # Kiểm tra xem có ví dụ được cung cấp không
            if not examples:
                raise ValueError(
                    f"few-shot prompt yêu cầu ví dụ được cung cấp qua tham số 'examples'. "
                    f"Vui lòng cung cấp danh sách ví dụ từ dữ liệu problems.json hoặc từ cơ sở dữ liệu."
                )
            
            # Kiểm tra nếu không đủ ví dụ
            actual_num_examples = min(num_examples, len(examples))
            if len(examples) < num_examples:
                print(f"Cảnh báo: Yêu cầu {num_examples} ví dụ nhưng chỉ có {len(examples)} ví dụ được cung cấp. "
                      f"Sử dụng {actual_num_examples} ví dụ.")
                
            return self._create_few_shot_prompt(
                question, frames["few_shot"], examples, actual_num_examples
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
        Giữ nguyên casing gốc của đáp án từ response.
        
        Args:
            response (str): Phản hồi từ mô hình
            prompt_type (str): Loại prompt đã sử dụng
            
        Returns:
            str: Câu trả lời cuối cùng (giữ nguyên casing gốc)
        """
        # Xử lý trường hợp không có phản hồi
        if not response:
            return ""
        
        # Hàm trợ giúp: trích xuất câu trả lời từ response dùng pattern, giữ nguyên casing
        def extract_with_pattern(response_text, patterns):
            """Tìm pattern và trả về match group giữ nguyên casing gốc"""
            response_lower = response_text.lower()
            for pattern in patterns:
                match = re.search(pattern, response_lower, re.DOTALL)
                if match:
                    # Lấy phần tương ứng từ response gốc dựa trên vị trí của match
                    start = match.start(1)
                    end = match.end(1)
                    # Tìm vị trí thực tế trong response gốc
                    answer_text = response_text[start:end].strip()
                    return answer_text
            return None
            
        # Xử lý khác nhau dựa trên loại prompt
        if prompt_type == "zero_shot":
            patterns = [
                r"đáp án:?\s*(.*?)$",
                r"vậy đáp án là:?\s*(.*?)$",
                r"kết quả là:?\s*(.*?)$", 
                r"kết luận:?\s*(.*?)$"
            ]
            
            answer = extract_with_pattern(response, patterns)
            if answer:
                return answer
                
            return response.strip()
        
        elif prompt_type.startswith("few_shot_"):
            patterns = [
                r"đáp án:?\s*(.*?)$",
                r"vậy đáp án là:?\s*(.*?)$",
                r"kết quả là:?\s*(.*?)$", 
                r"kết luận:?\s*(.*?)$"
            ]
            
            answer = extract_with_pattern(response, patterns)
            if answer:
                return answer
                
            # Nếu không tìm thấy pattern, trả về câu cuối cùng giữ nguyên casing
            sentences = re.split(r'[.!?]', response)
            return sentences[-1].strip()
        
        elif prompt_type == "cot" or prompt_type.startswith("cot_"):
            # Đối với CoT, tìm câu trả lời cuối cùng sau các bước lập luận
            patterns = [
                r"vậy đáp án là:?\s*(.*?)$",
                r"kết quả là:?\s*(.*?)$", 
                r"đáp án:?\s*(.*?)$",
                r"vậy?[,\s]*kết quả:?\s*(.*?)$",
                r"vậy?[,\s]*kết luận:?\s*(.*?)$",
                r"kết luận:?\s*(.*?)$",
                r"do đó[,\s]*đáp án:?\s*(.*?)$",
                r"do đó[,\s]*(.*?)$"
            ]
            
            answer = extract_with_pattern(response, patterns)
            if answer:
                return answer
            
            # Nếu không tìm thấy, lấy câu cuối cùng
            sentences = re.split(r'[.!?]', response)
            return sentences[-1].strip()
        
        elif prompt_type.startswith("self_consistency_") or prompt_type.startswith("cot_self_consistency_"):
            # Đối với self-consistency, tìm kết quả phổ biến nhất
            # Lưu trữ cả lowercase để so sánh và original để trả về
            answer_count = {}  # key: lowercase answer, value: count
            answers_map = {}   # key: lowercase answer, value: original answer (lần đầu tiên)
            
            # Tìm các đáp án trong mỗi cách giải
            # Pattern cần linh hoạt để match cả ở đầu dòng và giữa khoảng trắng
            patterns = [
                r"đáp án:?\s*(.*?)(?:\n|$)",
                r"kết quả:?\s*(.*?)(?:\n|$)",
                r"vậy đáp án là:?\s*(.*?)(?:\n|$)",
                r"vậy kết quả là:?\s*(.*?)(?:\n|$)",
                r"kết luận:?\s*(.*?)(?:\n|$)",
            ]
            
            # Áp dụng pattern và thu thập tất cả đáp án
            for pattern in patterns:
                # Tìm trên response gốc nhưng với regex case-insensitive
                matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    original_answer = match.group(1).strip()
                    if original_answer:  # Chỉ lấy nếu không rỗng
                        answer_lower = original_answer.lower().strip()
                        
                        # Đếm tần suất (dùng lowercase để so sánh)
                        if answer_lower not in answer_count:
                            answer_count[answer_lower] = 0
                            answers_map[answer_lower] = original_answer  # Lưu version gốc đầu tiên
                        answer_count[answer_lower] += 1
            
            if answer_count:
                # Trả về đáp án gốc của cái phổ biến nhất
                most_common_lower = max(answer_count.items(), key=lambda x: x[1])[0]
                return answers_map[most_common_lower]
            
            # Nếu không tìm thấy đáp án rõ ràng, tìm cách khác
            conclusions = re.findall(r"(?:vậy|do đó)[,\s]*(.*?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
            if conclusions:
                # Lấy kết luận cuối cùng
                return conclusions[-1].strip()
                
            # Nếu không tìm thấy kết luận, trả về câu cuối cùng
            return response.strip().split('\n')[-1]
        
        elif prompt_type == "react":
            # Đối với ReAct, tìm "Đáp án cuối cùng"
            patterns = [
                r"đáp án cuối cùng:?\s*(.*?)(?:\n|$)", 
                r"final answer:?\s*(.*?)(?:\n|$)",
                r"kết quả cuối cùng:?\s*(.*?)(?:\n|$)",
                r"kết luận cuối cùng:?\s*(.*?)(?:\n|$)"
            ]
            
            answer = extract_with_pattern(response, patterns)
            if answer:
                return answer
            
            # Nếu không tìm thấy, tìm bất kỳ đáp án nào
            patterns = [
                r"đáp án:?\s*(.*?)(?:\n|$)",
                r"kết quả:?\s*(.*?)(?:\n|$)",
                r"kết luận:?\s*(.*?)(?:\n|$)"
            ]
            
            answer = extract_with_pattern(response, patterns)
            if answer:
                return answer
            
            # Nếu tất cả đều thất bại, trả về câu cuối cùng
            return response.strip().split('\n')[-1]
        
        else:
            # Trường hợp mặc định
            return response.strip()

# Hàm wrapper ở cấp module để tương thích với import statement trong evaluator.py
def create_prompt(query, prompt_type, task_type=None, question_type=None, custom_examples=None):
    """
    Hàm wrapper cho PromptBuilder.create_prompt để tương thích với các module khác.
    
    Args:
        query (str): Câu hỏi cần trả lời
        prompt_type (str): Loại prompt (zero_shot, few_shot_3, few_shot_5, few_shot_7, etc.)
        task_type (str, optional): Loại nhiệm vụ, không sử dụng trong PromptBuilder hiện tại
        question_type (str, optional): Loại câu hỏi, không sử dụng trong PromptBuilder hiện tại
        custom_examples (List[Dict], optional): Các ví dụ tùy chỉnh cho few-shot prompts.
                                              Bắt buộc khi prompt_type là few_shot_* hoặc cot_few_shot_*.
            
    Returns:
        str: Prompt đã được tạo
        
    Raises:
        ValueError: Nếu few-shot prompt được yêu cầu nhưng không có ví dụ được cung cấp
    """
    builder = PromptBuilder()
    return builder.create_prompt(
        question=query,
        prompt_type=prompt_type,
        examples=custom_examples
    )
