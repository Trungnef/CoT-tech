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
        
        "classical_problem": f"""Solve the following classical problem in detail and systematically:

Problem: {query}

Solution: """,

        "vietnamese": f"""Hãy giải bài toán sau một cách chi tiết và có hệ thống:

Bài toán: {query}

Lời giải: """
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
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
        "general": f"""Answer the following question by working through it step by step.

Question: {query}

Step-by-step thinking:
1. First, I'll identify what the question is asking for.
2. Next, I'll determine what information is provided.
3. I'll think about how to approach this systematically.
4. I'll work through each logical step in sequence.
5. Finally, I'll verify my answer for correctness.

Let me solve this:
""",
        
        "math": f"""Solve the following mathematics problem by carefully working through each step.

Problem: {query}

Step-by-step solution:
1. I'll identify the mathematical concepts and formulas needed.
2. I'll organize the given information and determine what I need to find.
3. I'll apply the appropriate mathematical techniques.
4. I'll solve the equations or perform calculations systematically.
5. I'll verify my solution by checking units and reasonableness.

Working through the solution:
""",
        
        "classical_problem": f"""Solve the following classical problem with clear step-by-step reasoning.

Problem: {query}

Step-by-step reasoning:
1. First, I'll identify the core problem and relevant formulas or principles.
2. Next, I'll organize the given information and variables.
3. I'll develop a strategy to solve the problem.
4. I'll execute each calculation step carefully.
5. I'll verify my solution by checking its validity.

Detailed solution:
""",

        "vietnamese": f"""Giải bài toán sau đây bằng cách suy luận từng bước một cách rõ ràng.

Bài toán: {query}

Lời giải từng bước:
1. Đầu tiên, tôi sẽ xác định các khái niệm và công thức toán học cần thiết.
2. Tiếp theo, tôi sẽ tổ chức thông tin đã cho và xác định những gì cần tìm.
3. Tôi sẽ áp dụng các kỹ thuật toán học phù hợp.
4. Tôi sẽ giải các phương trình hoặc thực hiện tính toán một cách có hệ thống.
5. Cuối cùng, tôi sẽ kiểm tra lại lời giải để đảm bảo tính chính xác.

Tiến hành giải chi tiết:
"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
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

Think step by step.
"""
    return base_prompt


def tree_of_thought_prompt(query, task_type="math"):
    """
    Implement a Tree of Thought prompt that encourages the model to explore
    multiple reasoning branches and evaluate them.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    
    Returns:
    - str: Formatted Tree of Thought prompt
    """
    
    prompts = {
        "general": f"""Solve the following problem using a tree of thought approach:

Problem: {query}

First, identify 3 different approaches to solve this problem.
For each approach:
1. Describe the approach
2. Develop the solution step by step
3. Evaluate the strengths and weaknesses of this approach

Finally, select the best approach and provide the final answer based on it.

Begin your solution:""",
        
        "math": f"""Solve the following problem using a tree of thought approach:

Problem: {query}

First, identify 3 different approaches to solve this problem.
For each approach:
1. Describe the method
2. Develop the solution step by step
3. Evaluate the strengths and weaknesses of this approach

Finally, select the best approach and provide the final answer based on it.

Begin your solution:""",

        "vietnamese": f"""Giải bài toán sau đây bằng phương pháp cây suy nghĩ:

Bài toán: {query}

Đầu tiên, xác định 3 cách tiếp cận khác nhau để giải quyết bài toán này.
Đối với mỗi cách tiếp cận:
1. Mô tả phương pháp
2. Phát triển lời giải từng bước một
3. Đánh giá điểm mạnh và điểm yếu của cách tiếp cận này

Cuối cùng, chọn cách tiếp cận tốt nhất và đưa ra câu trả lời cuối cùng dựa trên cách tiếp cận đó.

Bắt đầu lời giải của bạn:"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to math if task_type not found
    return prompts.get(task_type.lower(), prompts["math"])

# Add new prompt types below

def zero_shot_prompt(query, task_type="math"):
    """
    Create a zero-shot prompt that directly asks the model to answer the question
    without any examples or specific reasoning instructions.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    
    Returns:
    - str: Formatted zero-shot prompt
    """
    prompts = {
        "general": f"""Answer the following question directly and accurately:

Question: {query}

Answer:""",
        
        "math": f"""Solve the following mathematics problem accurately:

Problem: {query}

Solution:""",
        
        "classical_problem": f"""Solve the following classical problem precisely:

Problem: {query}

Solution:""",

        "vietnamese": f"""Giải bài toán sau đây một cách chính xác:

Bài toán: {query}

Lời giải:"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to general if task_type not found
    return prompts.get(task_type.lower(), prompts["general"])

def few_shot_prompt(query, task_type="math", num_shots=3, example_selection="default", custom_examples=None, examples_file=None):
    """
    Create a few-shot prompt with a specified number of examples.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    - num_shots (int): Number of examples to include (3, 5, or 7)
    - example_selection (str): Method to select examples:
        - "default": Use hardcoded examples
        - "random": Randomly select from available examples
        - "custom": Use provided custom_examples
        - "file": Load examples from specified examples_file
    - custom_examples (list): List of custom examples to use (when example_selection="custom")
    - examples_file (str): Path to JSON file containing examples (when example_selection="file")
    
    Returns:
    - str: Formatted few-shot prompt with examples
    """
    # Determine the appropriate examples based on selection method and task type
    examples = get_flexible_examples(task_type, num_shots, example_selection, custom_examples, examples_file)
    
    # Create the prompt with the examples and the query
    if any('\u00C0' <= c <= '\u1EF9' for c in query):  # Vietnamese
        prompt = f"""Dưới đây là một số ví dụ về cách giải quyết các bài toán tương tự. Hãy học từ các mẫu này để giải bài toán mới:

{examples}

Bây giờ, hãy áp dụng cách tiếp cận tương tự để giải bài toán sau:

Bài toán: {query}

Lời giải:"""
    else:  # English
        prompt = f"""Below are examples demonstrating how to solve similar problems. Learn from these patterns to solve the new problem:

{examples}

Now, apply a similar approach to solve the following problem:

Problem: {query}

Solution:"""
    
    return prompt

def get_flexible_examples(task_type="math", num_shots=3, example_selection="default", custom_examples=None, examples_file=None):
    """
    Get examples for few-shot learning based on the specified selection method.
    
    Parameters:
    - task_type (str): Type of task ('general', 'math', etc.)
    - num_shots (int): Number of examples to include
    - example_selection (str): Method to select examples (default, random, custom, file)
    - custom_examples (list): List of custom examples to use
    - examples_file (str): Path to JSON file containing examples
    
    Returns:
    - str: Formatted examples
    """
    if example_selection == "custom" and custom_examples:
        # Use provided custom examples
        formatted_examples = format_custom_examples(custom_examples[:min(num_shots, len(custom_examples))])
        return formatted_examples
    
    elif example_selection == "file" and examples_file:
        # Load examples from file
        try:
            import json
            import os
            if os.path.exists(examples_file):
                with open(examples_file, 'r', encoding='utf-8') as f:
                    file_examples = json.load(f)
                
                # Check if the file has the expected structure
                if task_type in file_examples:
                    examples_data = file_examples[task_type]
                    # Preprocess examples to ensure they have problem, solution and reasoning
                    processed_examples = preprocess_examples_from_file(examples_data)
                    
                    # Select examples
                    if example_selection == "random":
                        import random
                        # Randomly select num_shots examples if there are enough examples
                        if len(processed_examples) > num_shots:
                            processed_examples = random.sample(processed_examples, num_shots)
                        else:
                            processed_examples = processed_examples[:num_shots]
                    else:
                        # Take the first num_shots examples
                        processed_examples = processed_examples[:min(num_shots, len(processed_examples))]
                    
                    # Format the selected examples
                    formatted_examples = format_examples_from_file(processed_examples)
                    return formatted_examples
        except Exception as e:
            print(f"Error loading examples from file: {e}")
            # Fall back to default examples
            pass
    
    elif example_selection == "random":
        # Get default examples and randomly select from them
        default_examples = get_examples_for_task(task_type, max(7, num_shots))  # Get more examples to randomly select from
        examples_list = default_examples.split("\n\n")
        
        # Filter out empty examples
        examples_list = [example for example in examples_list if example.strip()]
        
        # Randomly select num_shots examples if there are enough examples
        import random
        if len(examples_list) > num_shots:
            selected_examples = random.sample(examples_list, num_shots)
            return "\n\n".join(selected_examples) + "\n\n"
    
    # Default case: use hardcoded examples
    return get_examples_for_task(task_type, num_shots)

def format_custom_examples(examples_list):
    """
    Format a list of custom examples for use in few-shot prompts.
    
    Parameters:
    - examples_list (list): List of dictionaries containing 'problem' and 'solution' keys
    
    Returns:
    - str: Formatted examples
    """
    formatted_examples = ""
    
    for i, example in enumerate(examples_list):
        if isinstance(example, dict) and 'problem' in example and 'solution' in example:
            formatted_examples += f"""Example {i+1}:
Problem: {example['problem']}
Solution: {example['solution']}

"""
    
    return formatted_examples

def preprocess_examples_from_file(examples_data):
    """
    Preprocess examples from a file to ensure they have the required fields.
    
    Parameters:
    - examples_data (list): List of example dictionaries from the file
    
    Returns:
    - list: List of processed example dictionaries
    """
    processed_examples = []
    
    for example in examples_data:
        # Check if example has the required fields
        if 'question' in example and 'answer' in example:
            # Create a dictionary with the required fields
            processed_example = {
                'problem': example['question'],
                'solution': example['answer']
            }
            
            # If there's a 'reasoning' field, include it in the solution
            if 'reasoning' in example and example['reasoning']:
                processed_example['solution'] = f"{example['reasoning']}\n\nTherefore, {example['answer']}"
            
            processed_examples.append(processed_example)
    
    return processed_examples

def format_examples_from_file(examples_list):
    """
    Format examples loaded from a file for use in few-shot prompts.
    
    Parameters:
    - examples_list (list): List of dictionaries containing 'problem' and 'solution' keys
    
    Returns:
    - str: Formatted examples
    """
    return format_custom_examples(examples_list)

def few_shot_3_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for 3-shot prompting with flexible example selection"""
    return few_shot_prompt(query, task_type, num_shots=3, example_selection=example_selection, 
                          custom_examples=custom_examples, examples_file=examples_file)

def few_shot_5_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for 5-shot prompting with flexible example selection"""
    return few_shot_prompt(query, task_type, num_shots=5, example_selection=example_selection, 
                          custom_examples=custom_examples, examples_file=examples_file)

def few_shot_7_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for 7-shot prompting with flexible example selection"""
    return few_shot_prompt(query, task_type, num_shots=7, example_selection=example_selection, 
                          custom_examples=custom_examples, examples_file=examples_file)

def cot_self_consistency_prompt(query, task_type="math", num_iterations=3):
    """
    Create a prompt for Chain-of-Thought with Self-Consistency approach.
    The model will be instructed to generate multiple different solutions
    and then select the most consistent answer.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    - num_iterations (int): Number of solution attempts to generate (3, 5, or 7)
    
    Returns:
    - str: Formatted CoT Self-Consistency prompt
    """
    prompts = {
        "general": f"""Answer the following question by generating {num_iterations} distinct reasoning paths, then identify the most reliable answer.

Question: {query}

Guidelines:
1. Generate {num_iterations} completely different solution approaches
2. For each approach, reason step-by-step to reach an answer
3. After completing all approaches, evaluate which answer appears most consistent
4. Provide your final answer with an explanation of why you believe it's correct

Solution Approach #1:
""",
        
        "math": f"""Solve the following mathematics problem using {num_iterations} different approaches to verify the answer's consistency.

Problem: {query}

Instructions:
1. Develop {num_iterations} distinct solution methods using different mathematical techniques
2. Work through each approach step-by-step to reach a conclusion
3. Compare the answers from all approaches to identify the most consistent result
4. Provide your final answer with justification based on the consistency of results

First Approach:
""",
        
        "classical_problem": f"""Solve the following classical problem using {num_iterations} alternative solution methods to verify consistency.

Problem: {query}

Method:
1. Develop {num_iterations} fundamentally different approaches to solve this problem
2. Carefully work through each approach step-by-step
3. After completing all approaches, identify which answer appears most frequently
4. Present your final answer with confidence based on the consistency across approaches

Approach #1:
""",

        "vietnamese": f"""Giải bài toán sau đây bằng cách sử dụng {num_iterations} phương pháp khác nhau, sau đó xác định câu trả lời đáng tin cậy nhất.

Bài toán: {query}

Hướng dẫn:
1. Phát triển {num_iterations} phương pháp giải khác nhau sử dụng các kỹ thuật toán học khác nhau
2. Giải từng phương pháp theo từng bước để đi đến kết luận
3. So sánh các câu trả lời từ tất cả các cách tiếp cận để xác định kết quả nhất quán nhất
4. Đưa ra câu trả lời cuối cùng với lý giải dựa trên sự nhất quán của các kết quả

Phương pháp #1:
"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to math if task_type not found
    return prompts.get(task_type.lower(), prompts["math"])

def react_prompt(query, task_type="math"):
    """
    Create a ReAct (Reasoning + Acting) prompt that guides the model to alternate
    between reasoning about the problem and specifying actions to gather information.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', etc.)
    
    Returns:
    - str: Formatted ReAct prompt
    """
    prompts = {
        "general": f"""Solve the following problem using the ReAct framework (Reasoning + Acting).

Problem: {query}

FORMAT INSTRUCTIONS:
For each step in your solution:
1. THINK: Explain your reasoning for the current step
2. ACTION: Decide on a specific action to take (analyze, calculate, apply formula, etc.)
3. OBSERVATION: Record what you observe after taking the action

Your solution should follow this structured process for each step until you reach the final answer.
Begin by carefully analyzing the problem statement.

Step 1:
THINK: I need to understand what the problem is asking and identify the key elements.
ACTION: Analyze the problem statement to identify the question, given information, and required approach.
OBSERVATION: 

""",
        
        "math": f"""Solve the following mathematics problem using the ReAct framework (Reasoning + Acting).

Problem: {query}

FORMAT INSTRUCTIONS:
For each step in your mathematical solution:
1. THINK: Reason about what mathematical concepts apply and what approach to take
2. ACTION: Apply a specific mathematical operation or technique
3. OBSERVATION: Record the result of that mathematical operation

Continue this structured process until you reach the final solution.
Begin by analyzing what the problem is asking.

Step 1:
THINK: I need to understand what mathematical concepts and formulas are relevant to this problem.
ACTION: Identify the key variables, relationships, and applicable mathematical principles.
OBSERVATION: 

""",
        
        "classical_problem": f"""Solve the following classical problem using the ReAct framework (Reasoning + Acting).

Problem: {query}

FORMAT INSTRUCTIONS:
For each step in your solution process:
1. THINK: Reason about your approach and the concepts needed for this step
2. ACTION: Apply a specific technique, formula, or calculation
3. OBSERVATION: Record what you discovered from that action

This structured process should be followed for each step until you reach the final answer.
Begin with a careful analysis of what the problem is asking.

Step 1:
THINK: I need to understand the problem type and identify the relevant concepts and formulas.
ACTION: Analyze the problem statement to determine the variables, constants, and relationships.
OBSERVATION: 

""",

        "vietnamese": f"""Giải bài toán sau đây sử dụng khung ReAct (Lập luận + Hành động).

Bài toán: {query}

HƯỚNG DẪN ĐỊNH DẠNG:
Với mỗi bước trong quá trình giải:
1. SUY NGHĨ: Giải thích lý do cho bước hiện tại
2. HÀNH ĐỘNG: Quyết định một hành động cụ thể (phân tích, tính toán, áp dụng công thức, v.v.)
3. QUAN SÁT: Ghi lại những gì bạn quan sát được sau khi thực hiện hành động

Lời giải của bạn nên tuân theo quy trình có cấu trúc này cho đến khi đạt được câu trả lời cuối cùng.
Bắt đầu bằng cách phân tích kỹ đề bài.

Bước 1:
SUY NGHĨ: Tôi cần hiểu bài toán đang yêu cầu gì và xác định các khái niệm toán học liên quan.
HÀNH ĐỘNG: Phân tích đề bài để xác định các biến, hằng số và mối quan hệ.
QUAN SÁT: 

"""
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to math if task_type not found
    return prompts.get(task_type.lower(), prompts["math"])

def get_examples_for_task(task_type="math", num_shots=3):
    """
    Get appropriate examples for a specific task type and number of shots.
    
    Parameters:
    - task_type (str): Type of task ('general', 'math', etc.)
    - num_shots (int): Number of examples to include (3, 5, or 7)
    
    Returns:
    - str: Formatted examples
    """
    # Dictionary of examples for different task types
    # We'll include up to 7 examples for each task type
    examples = {
        "math": [
            """Example 1:
Problem: Calculate the area of a circle with radius 5 cm.
Solution: The area of a circle is given by the formula A = πr². 
With radius r = 5 cm, the area is A = π × 5² = π × 25 = 78.54 cm².

""",
            """Example 2:
Problem: Solve for x in the equation 2x + 7 = 15.
Solution: 
2x + 7 = 15
2x = 15 - 7
2x = 8
x = 4

""",
            """Example 3:
Problem: Find the derivative of f(x) = 3x² + 2x - 5.
Solution: Using the power rule and linearity of differentiation:
f'(x) = 3 × 2x + 2 = 6x + 2

""",
            """Example 4:
Problem: If a car travels at 60 km/h, how far will it go in 2.5 hours?
Solution: Distance = Speed × Time
Distance = 60 km/h × 2.5 h = 150 km

""",
            """Example 5:
Problem: Find the value of x if log₁₀(x) = 2.
Solution: 
log₁₀(x) = 2
10² = x
x = 100

""",
            """Example 6:
Problem: A box contains 3 red balls, 4 blue balls, and 5 green balls. If a ball is selected at random, what is the probability of selecting a red ball?
Solution: 
Total number of balls = 3 + 4 + 5 = 12
Probability of selecting a red ball = Number of red balls / Total number of balls = 3/12 = 1/4 = 0.25

""",
            """Example 7:
Problem: Find the sum of the first 10 terms of the arithmetic sequence 5, 8, 11, 14, ...
Solution: 
This is an arithmetic sequence with first term a = 5 and common difference d = 3.
Sum of the first n terms of an arithmetic sequence is S_n = (n/2)(2a + (n-1)d)
S_10 = (10/2)(2×5 + (10-1)×3)
S_10 = 5(10 + 27)
S_10 = 5 × 37
S_10 = 185

"""
        ],
        
        "classical_problem": [
            """Example 1:
Problem: If a train travels at 60 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 420 km apart if they start at the same point?
Solution: 
The relative speed of the trains is 60 + 80 = 140 km/h.
Time = Distance / Speed
Time = 420 km / 140 km/h = 3 hours.

""",
            """Example 2:
Problem: A rectangle has a perimeter of 30 cm and an area of 56 cm². Find its dimensions.
Solution:
Let the length be l and width be w.
From the perimeter: 2l + 2w = 30, so l + w = 15.
From the area: l × w = 56.
Using the first equation, w = 15 - l.
Substituting into the second equation: l × (15 - l) = 56
l × 15 - l² = 56
-l² + 15l - 56 = 0
l² - 15l + 56 = 0
Using the quadratic formula: l = 7 or l = 8.
If l = 7, then w = 15 - 7 = 8.
If l = 8, then w = 15 - 8 = 7.
The dimensions are 7 cm × 8 cm.

""",
            """Example 3:
Problem: Find two consecutive integers whose product is 182.
Solution:
Let the integers be n and n+1.
n × (n+1) = 182
n² + n = 182
n² + n - 182 = 0
Using the quadratic formula: n = (-1 ± √(1 + 4×182))/2 = (-1 ± √729)/2 = (-1 ± 27)/2
n = 13 or n = -14
Since we need consecutive integers, the answer is either (13, 14) or (-14, -13).
Checking: 13 × 14 = 182 and -14 × -13 = 182.
Both pairs work, but typically we'd use the positive pair (13, 14).

""",
            """Example 4:
Problem: A mixture of 40 liters contains 15% alcohol. How much water should be added to make the alcohol content 10%?
Solution:
Initial amount of alcohol = 40L × 15% = 6L.
After adding water, the alcohol amount remains 6L.
If the new percentage is 10%, and the amount of alcohol is 6L, then:
10% of new volume = 6L
New volume = 6L / 10% = 6L / 0.1 = 60L
Amount of water to add = 60L - 40L = 20L.

""",
            """Example 5:
Problem: A car travels from city A to city B at an average speed of 60 km/h and returns at 40 km/h. Find the average speed for the entire journey.
Solution:
Let the distance between cities be d.
Time from A to B = d/60 hours.
Time from B to A = d/40 hours.
Total time = d/60 + d/40 = (d×40 + d×60)/(60×40) = 100d/2400 = d/24 hours.
Total distance = 2d.
Average speed = Total distance / Total time = 2d / (d/24) = 48 km/h.

""",
            """Example 6:
Problem: Solve the system of equations: 3x + 2y = 7 and 5x - y = 8.
Solution:
From the second equation: y = 5x - 8.
Substituting into the first equation:
3x + 2(5x - 8) = 7
3x + 10x - 16 = 7
13x = 23
x = 23/13
Using y = 5x - 8:
y = 5(23/13) - 8 = 115/13 - 104/13 = 11/13
The solution is x = 23/13, y = 11/13.

""",
            """Example 7:
Problem: A ladder 10 meters long is leaning against a vertical wall. The bottom of the ladder is 6 meters from the wall. How high up the wall does the ladder reach?
Solution:
Using the Pythagorean theorem in the right triangle:
Length of ladder² = Distance from wall² + Height up the wall²
10² = 6² + h²
100 = 36 + h²
h² = 64
h = 8 meters.

"""
        ],
        
        "general": [
            """Example 1:
Question: What is the capital of France?
Answer: The capital of France is Paris.

""",
            """Example 2:
Question: Who wrote the novel "Pride and Prejudice"?
Answer: "Pride and Prejudice" was written by Jane Austen.

""",
            """Example 3:
Question: What is the chemical formula for water?
Answer: The chemical formula for water is H₂O, which consists of two hydrogen atoms and one oxygen atom.

""",
            """Example 4:
Question: What is photosynthesis?
Answer: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.

""",
            """Example 5:
Question: Who was the first person to walk on the moon?
Answer: Neil Armstrong was the first person to walk on the moon on July 20, 1969, during the Apollo 11 mission.

""",
            """Example 6:
Question: What's the difference between a virus and a bacterium?
Answer: Viruses and bacteria are different types of microorganisms. Viruses are much smaller, aren't cell-based, and cannot reproduce on their own—they need to infect a host cell. Bacteria are single-celled organisms that can reproduce independently. Bacteria can be treated with antibiotics, while antibiotics are ineffective against viruses.

""",
            """Example 7:
Question: What causes the northern lights?
Answer: The northern lights, or aurora borealis, are caused by interactions between solar particles and the Earth's magnetic field. When charged particles from the sun collide with atoms in Earth's atmosphere, they produce light of various colors, creating the spectacular light displays seen near the polar regions.

"""
        ],
        
        # Vietnamese examples
        "vietnamese": [
            """Ví dụ 1:
Bài toán: Tính diện tích của một hình tròn có bán kính 5 cm.
Lời giải: Diện tích của hình tròn được tính bằng công thức A = πr². 
Với bán kính r = 5 cm, diện tích là A = π × 5² = π × 25 = 78.54 cm².

""",
            """Ví dụ 2:
Bài toán: Giải phương trình 2x + 7 = 15.
Lời giải: 
2x + 7 = 15
2x = 15 - 7
2x = 8
x = 4

""",
            """Ví dụ 3:
Bài toán: Tìm đạo hàm của hàm số f(x) = 3x² + 2x - 5.
Lời giải: Sử dụng quy tắc lũy thừa và tính chất tuyến tính của đạo hàm:
f'(x) = 3 × 2x + 2 = 6x + 2

""",
            """Ví dụ 4:
Bài toán: Nếu một chiếc xe di chuyển với vận tốc 60 km/h, nó sẽ đi được bao xa trong 2.5 giờ?
Lời giải: Quãng đường = Vận tốc × Thời gian
Quãng đường = 60 km/h × 2.5 h = 150 km

""",
            """Ví dụ 5:
Bài toán: Tìm giá trị của x nếu log₁₀(x) = 2.
Lời giải: 
log₁₀(x) = 2
10² = x
x = 100

""",
            """Ví dụ 6:
Bài toán: Một hộp chứa 3 viên bi đỏ, 4 viên bi xanh, và 5 viên bi xanh lá. Nếu một viên bi được chọn ngẫu nhiên, xác suất chọn được viên bi đỏ là bao nhiêu?
Lời giải: 
Tổng số viên bi = 3 + 4 + 5 = 12
Xác suất chọn được viên bi đỏ = Số viên bi đỏ / Tổng số viên bi = 3/12 = 1/4 = 0.25

""",
            """Ví dụ 7:
Bài toán: Tìm tổng 10 số hạng đầu tiên của cấp số cộng 5, 8, 11, 14, ...
Lời giải: 
Đây là cấp số cộng với số hạng đầu tiên a = 5 và công sai d = 3.
Tổng n số hạng đầu tiên của cấp số cộng là S_n = (n/2)(2a + (n-1)d)
S_10 = (10/2)(2×5 + (10-1)×3)
S_10 = 5(10 + 27)
S_10 = 5 × 37
S_10 = 185

"""
        ]
    }
    
    # Default to math if task_type not found
    task_examples = examples.get(task_type.lower(), examples["math"])
    
    # Limit the number of examples based on num_shots
    limited_examples = task_examples[:min(num_shots, len(task_examples))]
    
    # Join the examples into a single string
    return "".join(limited_examples)

def cot_self_consistency_3_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for CoT Self-Consistency with 3 iterations and flexible example selection"""
    return cot_self_consistency_prompt(query, task_type, num_iterations=3)

def cot_self_consistency_5_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for CoT Self-Consistency with 5 iterations and flexible example selection"""
    return cot_self_consistency_prompt(query, task_type, num_iterations=5)

def cot_self_consistency_7_prompt(query, task_type="math", example_selection="default", custom_examples=None, examples_file=None):
    """Helper function for CoT Self-Consistency with 7 iterations and flexible example selection"""
    return cot_self_consistency_prompt(query, task_type, num_iterations=7) 