"""
Prompt templates for querying language models with different reasoning strategies.
"""

def standard_prompt(query, task_type="math"):
    """
    Create a standardized prompt for different types of tasks.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', 'coding', 'writing', etc.)
    
    Returns:
    - str: Formatted prompt ready to be sent to an LLM
    """
    prompts = {
        "general": f"""Answer the following question thoroughly and accurately:

Question: {query}

Answer: """,
        
        "math": f"""Solve the following mathematics problem step by step:

Problem: {query}

Solution: """,
        
        "classical_problem": f"""Hãy giải bài toán cổ điển sau một cách chi tiết và có hệ thống:

Bài toán: {query}

Lời giải: """
    }
    
    # Default to general if task_type not found
    return prompts.get(task_type.lower(), prompts["general"])


def chain_of_thought_prompt(query, task_type="math"):
    """
    Create a Chain-of-Thought prompt that explicitly asks the model
    to reason step by step before giving the final answer.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', 'coding', 'writing', etc.)
    
    Returns:
    - str: Formatted CoT prompt ready to be sent to an LLM
    """
    prompts = {
        "general": f"""Answer the following question step by step. First, break down the problem, analyze each component, and then provide the final answer.

Question: {query}

Thought process:
1) """,
        
        "math": f"""Solve the following mathematics problem step by step. First analyze what is being asked, identify the key information, plan your approach, then execute each step until you reach the final answer.

Problem: {query}

Thought process:
1) Let me understand what the problem is asking.
2) """,
        
        "classical_problem": f"""Hãy giải bài toán cổ điển sau theo phương pháp suy luận từng bước. Trước tiên, hãy phân tích đề bài, xác định thông tin quan trọng, lập kế hoạch giải quyết, sau đó thực hiện từng bước tính toán cho đến khi có kết quả cuối cùng.

Bài toán: {query}

Quá trình suy luận:
1) Tôi sẽ phân tích kỹ đề bài để hiểu yêu cầu.
2) """
    }
    
    # Default to general if task_type not found
    return prompts.get(task_type.lower(), prompts["general"])


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
        
        "math": f"""Giải bài toán sau đây bằng kỹ thuật suy luận nâng cao:

Bài toán: {query}

Bước 1: Phân tích bài toán thành các bài toán con đơn giản hơn.

Bước 2: Với mỗi bài toán con, hãy đưa ra 2-3 cách tiếp cận khác nhau.

Bước 3: Giải quyết từng bài toán con theo cách tiếp cận tốt nhất, lập luận từng bước một.

Bước 4: Kết hợp các lời giải của bài toán con để tạo thành lời giải hoàn chỉnh.

Bước 5: Tự đánh giá lời giải của bạn. Kiểm tra lỗi, xác minh tính toán và đảm bảo lập luận vững chắc.

Bước 6: Đưa ra câu trả lời cuối cùng dựa trên lời giải đã xác minh.

Bắt đầu lời giải:"""
    }
    
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
    
    base_prompt = f"""
{query}

Hãy suy nghĩ từng bước một.
"""
    return base_prompt


# def tree_of_thought_prompt(query, task_type="math"):
#     """
#     Implement a Tree of Thought prompt that encourages the model to explore
#     multiple reasoning branches and evaluate them.
    
#     Parameters:
#     - query (str): The user's question or problem statement
#     - task_type (str): Type of task ('general', 'math', etc.)
    
#     Returns:
#     - str: Formatted Tree of Thought prompt
#     """
    
#     prompts = {
#         "general": f"""Solve the following problem using a tree of thought approach:

# Problem: {query}

# First, identify 3 different approaches to solve this problem.
# For each approach:
# 1. Describe the approach
# 2. Develop the solution step by step
# 3. Evaluate the strengths and weaknesses of this approach

# Finally, select the best approach and provide the final answer based on it.

# Begin your solution:""",
        
#         "math": f"""Giải bài toán sau đây bằng phương pháp cây suy luận (tree of thought):

# Bài toán: {query}

# Đầu tiên, xác định 3 cách tiếp cận khác nhau để giải bài toán này.
# Với mỗi cách tiếp cận:
# 1. Mô tả phương pháp
# 2. Phát triển lời giải từng bước một
# 3. Đánh giá điểm mạnh và điểm yếu của phương pháp này

# Cuối cùng, chọn phương pháp tốt nhất và đưa ra câu trả lời cuối cùng dựa trên phương pháp đó.

# Bắt đầu lời giải:"""
#     }
    
#     # Default to math if task_type not found
#     return prompts.get(task_type.lower(), prompts["math"]) 