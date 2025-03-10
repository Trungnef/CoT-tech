import os
import fitz  # PyMuPDF
import re
import json
from tqdm import tqdm

def extract_questions_from_pdf(pdf_path, output_json_path=None):
    """
    Extract questions from a PDF document containing classical problems
    
    Args:
        pdf_path (str): Path to the PDF file
        output_json_path (str, optional): Path to save extracted questions
        
    Returns:
        list: List of extracted questions
    """
    print(f"📄 Attempting to extract questions from: {pdf_path}")
    print(f"📍 Full absolute path: {os.path.abspath(pdf_path)}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return []
    
    try:
        # Open the PDF file
        print("🔍 Opening PDF file...")
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)
        print(f"✅ Successfully opened PDF with {num_pages} pages")
        
        questions = []
        current_question = ""
        in_question = False
        
        # Các mẫu để nhận dạng câu hỏi
        question_patterns = [
            r'^\d+\.\s',           # Số + dấu chấm (vd: "1. ")
            r'Câu\s+\d+[:.]\s*',   # "Câu" + số (vd: "Câu 1:", "Câu 1.")
            r'Bài\s+\d+[:.]\s*',   # "Bài" + số (vd: "Bài 1:", "Bài 1.")
            r'Bài\s+toán\s+\d+[:.]\s*',  # "Bài toán" + số
            r'Câu\s+hỏi\s+\d+[:.]\s*',   # "Câu hỏi" + số
            r'Exercise\s+\d+[:.]\s*',     # Tiếng Anh
            r'Problem\s+\d+[:.]\s*'       # Tiếng Anh
        ]
        
        # Compile patterns for better performance
        patterns = [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in question_patterns]
        
        print(f"📖 Processing {num_pages} pages...")
        
        # Process each page
        for page_num in tqdm(range(num_pages)):
            try:
                page = pdf_document[page_num]
                text = page.get_text("text")  # Extract text with better formatting
                
                # Debug: Print first few characters of text
                if page_num == 0:
                    print(f"\n📝 Sample of extracted text from first page:")
                    print(text[:500] + "...")
                
                # Split text into lines and clean them
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    # Check if line contains any question marker
                    is_question_start = any(pattern.search(line) for pattern in patterns)
                    
                    if is_question_start:
                        # Debug: Print found question marker
                        print(f"\n🔍 Found question marker: {line[:100]}...")
                        
                        # If we were already processing a question, save it
                        if current_question:
                            questions.append(current_question.strip())
                            print(f"✅ Saved question: {current_question[:100]}...")
                        
                        # Start a new question
                        current_question = line
                        in_question = True
                    elif in_question and line:
                        # Continue adding to the current question if line is not empty
                        # and doesn't match any question pattern
                        if not any(pattern.search(line) for pattern in patterns):
                            current_question += " " + line
            
            except Exception as e:
                print(f"⚠️ Warning: Error processing page {page_num}: {e}")
                continue
        
        # Add the last question if there is one
        if current_question:
            questions.append(current_question.strip())
        
        # Remove duplicate questions and empty strings
        questions = [q for q in questions if q]
        questions = list(dict.fromkeys(questions))
        
        print(f"\n✅ Extracted {len(questions)} questions")
        
        # Print sample of extracted questions
        if questions:
            print("\n📋 Sample of extracted questions:")
            for i, q in enumerate(questions[:5], 1):
                print(f"\nQuestion {i}: {q[:200]}...")
        else:
            print("\n⚠️ No questions were extracted. Showing first page content for debugging:")
            first_page = pdf_document[0].get_text("text")
            print("\nFirst page content:")
            print(first_page[:1000])
        
        # Save to JSON if path provided
        if output_json_path and questions:
            try:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(questions, json_file, ensure_ascii=False, indent=2)
                print(f"✅ Saved {len(questions)} questions to {output_json_path}")
            except Exception as e:
                print(f"❌ Error saving to JSON: {e}")
        
        # Close the PDF document
        pdf_document.close()
        
        return questions
        
    except Exception as e:
        print(f"❌ Error extracting questions: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Create db/questions directory if it doesn't exist
    os.makedirs("db/questions", exist_ok=True)
    
    # Try different possible file names
    possible_paths = [
        "./db/Nhung_bai_toan_co2.pdf",
        # "./db/Nhung bai toan co.pdf",
        # "db/Nhung_bai_toan_co.pdf",
        # "db/Nhung bai toan co.pdf",
        # "./Nhung bai toan co.pdf",
        # "Nhung bai toan co.pdf"
    ]
    
    pdf_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pdf_path = path
            print(f"✅ Found PDF at: {path}")
            break
    
    if not pdf_path:
        print("❌ PDF not found. Tried following paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
    else:
        # Extract and save questions
        questions = extract_questions_from_pdf(
            pdf_path, 
            output_json_path="db/questions/classical_problems.json"
        ) 