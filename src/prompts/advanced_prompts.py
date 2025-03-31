"""
Advanced prompt templates for querying language models with more complex reasoning strategies.
"""

def hybrid_cot_prompt(query, task_type="math"):
    """
    Implement a Hybrid Chain-of-Thought prompt that combines:
    - Self-consistency: exploring multiple reasoning paths
    - Least-to-most: breaking complex problems into simpler subproblems
    - Reflection: evaluating the quality of each reasoning path
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', 'coding', 'writing', etc.)
    
    Returns:
    - str: Formatted Hybrid-CoT prompt ready to be sent to an LLM
    """
    prompts = {
        "general": f"""Answer the following question using advanced reasoning techniques:

Question: {query}

Step 1: Break down this problem into smaller, more manageable sub-problems.

Step 2: For each sub-problem, explore 2-3 different solution approaches.

Step 3: Solve each sub-problem step by step using the best approach.

Step 4: Combine the solutions to the sub-problems to form a complete solution.

Step 5: Reflect on your solution. Check for errors, verify your calculations, and ensure your reasoning is solid.

Step 6: Provide your final answer based on your verified solution.

Begin your solution:""",
        
        "math": f"""Solve the following problem using advanced reasoning techniques:

Problem: {query}

Step 1: Break down the problem into simpler sub-problems.

Step 2: For each sub-problem, explore 2-3 different approaches.

Step 3: Solve each sub-problem step by step using the best approach.

Step 4: Combine the solutions to create a complete solution.

Step 5: Review your solution. Check for errors, verify calculations, and ensure solid reasoning.

Step 6: Provide your final answer based on the verified solution.

Begin your solution:""",

        "vietnamese": f"""Giải bài toán sau đây bằng các kỹ thuật suy luận nâng cao:

Bài toán: {query}

Bước 1: Chia nhỏ bài toán thành các bài toán con đơn giản hơn.

Bước 2: Với mỗi bài toán con, khám phá 2-3 cách tiếp cận khác nhau.

Bước 3: Giải từng bài toán con theo từng bước sử dụng cách tiếp cận tốt nhất.

Bước 4: Kết hợp các lời giải của các bài toán con để tạo thành lời giải hoàn chỉnh.

Bước 5: Xem xét lại lời giải của bạn. Kiểm tra lỗi, xác minh tính toán và đảm bảo suy luận vững chắc.

Bước 6: Đưa ra câu trả lời cuối cùng dựa trên lời giải đã được xác minh.

Bắt đầu lời giải của bạn:"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to math if task_type not found (since most classical problems are math)
    return prompts.get(task_type.lower(), prompts["math"])


def zero_shot_cot_prompt(query, task_type="math"):
    """
    Implement the Zero-Shot Chain-of-Thought prompt that simply instructs
    the model to think step by step.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    
    Returns:
    - str: Formatted Zero-Shot CoT prompt
    """
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        base_prompt = f"""
{query}

Hãy suy nghĩ từng bước một.
"""
    else:
        base_prompt = f"""
{query}

Let's think step by step.
"""
    
    return base_prompt


def tree_of_thought_prompt(query, task_type="math"):
    """
    Implement a Tree-of-Thought prompt that encourages the model to:
    1. Explore multiple reasoning paths
    2. Evaluate each path
    3. Select and extend the most promising paths
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    
    Returns:
    - str: Formatted Tree-of-Thought prompt
    """
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        prompt = f"""Hãy giải bài toán sau đây bằng phương pháp Cây Suy Luận (Tree of Thought):

Bài toán: {query}

Các bước thực hiện:
1. Khởi tạo: Đề xuất 3 cách tiếp cận khác nhau cho bài toán.
2. Đánh giá: Cho mỗi cách tiếp cận, đánh giá khả năng thành công và lý do tại sao.
3. Phát triển: Chọn cách tiếp cận triển vọng nhất và phát triển thêm một bước.
4. Lặp lại: Sau mỗi bước, đánh giá lại các hướng đi khả thi và lựa chọn hướng tốt nhất.
5. Kết luận: Khi đã tìm được lời giải, tổng kết bằng câu trả lời cuối cùng.

Bắt đầu giải quyết:
"""
    else:
        prompt = f"""Solve the following problem using Tree of Thought approach:

Problem: {query}

Steps to follow:
1. Initiation: Propose 3 different approaches to tackle this problem.
2. Evaluation: For each approach, assess its likelihood of success and why.
3. Development: Select the most promising approach and develop it one step further.
4. Iteration: After each step, reevaluate viable paths and choose the best direction.
5. Conclusion: Once a solution is found, summarize with the final answer.

Begin solving:
"""
    
    return prompt 