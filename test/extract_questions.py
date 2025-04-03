import os
import fitz  # PyMuPDF
import re
import json
from tqdm import tqdm
from typing import List, Dict

class QuestionExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.questions_dir = 'db/questions'
        os.makedirs(self.questions_dir, exist_ok=True)
        
    def extract_questions(self):
        doc = fitz.open(self.pdf_path)
        current_section = None
        questions = []
        current_question = None
        question_text = []
        solution_text = []
        is_in_solution = False
        
        for page in doc:
            text = page.get_text()
            # Xử lý các ký tự xuống dòng đặc biệt và dấu câu
            text = text.replace('\r\n', ' ').replace('\n\n', '\n').strip()
            text = re.sub(r'\s+', ' ', text)  # Chuẩn hóa khoảng trắng
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                # Tìm phần mới
                if line.startswith('Phần'):
                    if current_section and questions:
                        self._save_questions(current_section, questions)
                        questions = []
                    current_section = line.split(': ')[1].strip()
                    continue
                
                # Tìm câu hỏi mới
                if line.startswith('Bài'):
                    if current_question:
                        if question_text:
                            current_question['question'] = ' '.join(question_text)
                        if solution_text:
                            current_question['solution'] = ' '.join(solution_text)
                        questions.append(current_question)
                    
                    # Reset cho câu hỏi mới
                    current_question = {
                        'id': len(questions) + 1,
                        'question': '',
                        'solution': '',
                        'type': current_section
                    }
                    question_text = []
                    solution_text = []
                    is_in_solution = False
                    continue
                
                # Xác định phần hướng dẫn giải
                if 'Hướng dẫn giải' in line or 'Lời giải' in line or 'Giải' in line:
                    is_in_solution = True
                    continue
                
                # Thêm nội dung vào câu hỏi hoặc lời giải
                if current_question:
                    if not is_in_solution:
                        question_text.append(line)
                    else:
                        solution_text.append(line)
        
        # Xử lý câu hỏi cuối cùng
        if current_question:
            if question_text:
                current_question['question'] = ' '.join(question_text)
            if solution_text:
                current_question['solution'] = ' '.join(solution_text)
            questions.append(current_question)
        
        # Lưu phần cuối cùng
        if current_section and questions:
            self._save_questions(current_section, questions)
        
        doc.close()
        print(f"✅ Đã trích xuất xong các câu hỏi từ file PDF")
    
    def _save_questions(self, section: str, questions: List[Dict]):
        # Chuẩn hóa tên file
        filename = section.lower().replace(' ', '_')
        filename = re.sub(r'[^a-z0-9_]', '', filename)
        filepath = os.path.join(self.questions_dir, f'{filename}.json')
        
        # Chuẩn hóa dữ liệu và xử lý văn bản
        for q in questions:
            # Chuẩn hóa câu hỏi
            q['question'] = self._normalize_text(q['question'])
            # Chuẩn hóa lời giải
            q['solution'] = self._normalize_text(q['solution'])
            # Thêm metadata
            q['difficulty'] = self._estimate_difficulty(q['question'], q['solution'])
            q['tags'] = self._extract_tags(q['question'])
        
        # Lưu file JSON với định dạng đẹp
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'section': section,
                'questions': questions
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Đã lưu {len(questions)} câu hỏi từ phần '{section}' vào {filepath}")
    
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        # Chuẩn hóa dấu câu
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        # Chuẩn hóa dấu ngoặc
        text = re.sub(r'\s*([()[\]])\s*', r'\1', text)
        return text.strip()
    
    def _estimate_difficulty(self, question: str, solution: str) -> str:
        # Ước lượng độ khó dựa trên độ dài và độ phức tạp
        complexity = len(solution.split()) / len(question.split()) if question else 1
        if complexity > 2:
            return "Khó"
        elif complexity > 1.5:
            return "Trung bình"
        return "Dễ"
    
    def _extract_tags(self, question: str) -> List[str]:
        # Trích xuất từ khóa quan trọng
        keywords = {
            'số học': ['số', 'chữ số', 'ước số', 'bội số', 'tổng', 'hiệu', 'tích', 'thương'],
            'hình học': ['tam giác', 'hình vuông', 'hình chữ nhật', 'diện tích', 'chu vi'],
            'đại số': ['phương trình', 'biểu thức', 'số x', 'nghiệm'],
            'logic': ['nếu', 'thì', 'hoặc', 'và', 'suy ra'],
            'thực tế': ['tiền', 'tuổi', 'giờ', 'ngày', 'tháng', 'năm']
        }
        
        tags = []
        text = question.lower()
        for category, words in keywords.items():
            if any(word in text for word in words):
                tags.append(category)
        return tags

    def get_statistics(self):
        total_questions = 0
        sections = {}
        
        for filename in os.listdir(self.questions_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.questions_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    count = len(data.get('questions', []))
                    sections[os.path.splitext(filename)[0]] = count
                    total_questions += count
        
        print("\n📊 Thống kê:")
        print(f"Tổng số câu hỏi: {total_questions}")
        print("\nPhân bố theo phần:")
        for section, count in sections.items():
            print(f"- {section}: {count} câu")

if __name__ == '__main__':
    extractor = QuestionExtractor('db/Nhung_bai_toan_co_dien.pdf')
    extractor.extract_questions()
    extractor.get_statistics() 