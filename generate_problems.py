from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import random
from typing import List, Dict, Tuple
import math
import json
import os

class ProblemGenerator:
    def __init__(self):
        self.pdf = canvas.Canvas('db/Nhung_bai_toan_co_dien.pdf', pagesize=A4)
        pdfmetrics.registerFont(TTFont('times', r'C:\Windows\Fonts\times.ttf'))
        pdfmetrics.registerFont(TTFont('timesbd', r'C:\Windows\Fonts\timesbd.ttf'))
        pdfmetrics.registerFont(TTFont('timesi', r'C:\Windows\Fonts\timesi.ttf'))
        self.width, self.height = A4
        self.current_section = 1
        self.current_problem = 1
        self.y = self.height - 50
        self.problem_types = {
            "Câu đố logic": {
                "templates": [
                    {
                        "template": "Một người nông dân có {total_animals} con vật gồm gà và thỏ. Đếm tổng số chân thì thấy có {total_legs} cái chân. Hỏi người nông dân có bao nhiêu con gà và bao nhiêu con thỏ?",
                        "solution": "Gọi số gà là x con. Khi đó số thỏ là {total_animals} - x con. Số chân gà: 2x. Số chân thỏ: 4({total_animals} - x). Theo đề bài: 2x + 4({total_animals} - x) = {total_legs}. Giải ra: x = {chickens} con gà. Vậy có {rabbits} con thỏ."
                    },
                    {
                        "template": "Một người bán cam mua {total_oranges} quả cam với giá {buy_price}đ một quả. Người đó bán lại với giá {sell_price}đ một quả nhưng có {bad_oranges} quả bị hỏng không bán được. Hỏi người đó lãi hay lỗ bao nhiêu?",
                        "solution": "Tiền mua: {total_oranges} × {buy_price} = {total_cost}đ. Số cam bán được: {total_oranges} - {bad_oranges} = {sold_oranges} quả. Tiền bán: {sold_oranges} × {sell_price} = {revenue}đ. Vậy {profit_loss_text}: {result}đ."
                    },
                    {
                        "template": "Có {total_people} người đứng thành vòng tròn, đếm từ 1 đến {count_to} rồi loại người thứ {count_to} ra khỏi vòng. Cứ tiếp tục như vậy cho đến khi còn lại 1 người. Hỏi người đứng ở vị trí thứ mấy ban đầu sẽ là người còn lại cuối cùng?",
                        "solution": "Đây là bài toán Josephus. Với n = {total_people} người và k = {count_to}, công thức là: f(n,k) = (f(n-1,k) + k-1) mod n + 1, với f(1,k) = 1. Thực hiện từng bước: {steps}. Vậy người đứng ở vị trí {result} sẽ là người còn lại cuối cùng."
                    }
                ],
                "weight": 0.15  # 300 bài
            },
            "Câu hỏi kiểu bài luận": {
                "templates": [
                    {
                        "template": "Giải thích tại sao khi chia một số cho {divisor}, nếu tổng các chữ số của số đó chia hết cho {divisor} thì số đó cũng chia hết cho {divisor}?",
                        "solution": "Để chứng minh điều này, ta xét một số N có n chữ số: N = a₁×10^(n-1) + a₂×10^(n-2) + ... + aₙ. Khi chia 10^k cho {divisor}, ta được dư {remainder}. Do đó, N ≡ a₁×{remainder}^(n-1) + a₂×{remainder}^(n-2) + ... + aₙ (mod {divisor}). Vì {explanation}, nên tổng các chữ số chia hết cho {divisor} thì N cũng chia hết cho {divisor}."
                    },
                    {
                        "template": "Chứng minh rằng trong tam giác vuông, nếu một góc là {angle}°, thì tỷ số giữa cạnh đối với góc đó và cạnh huyền luôn bằng {ratio}.",
                        "solution": "Trong tam giác vuông, sin của một góc bằng tỷ số giữa cạnh đối và cạnh huyền. Ta có sin({angle}°) = {ratio}. Điều này đúng với mọi tam giác vuông có góc {angle}° vì các tam giác vuông có góc bằng nhau thì đồng dạng với nhau. Do đó, tỷ số giữa các cạnh tương ứng luôn không đổi."
                    },
                    {
                        "template": "Giải thích tại sao khi cộng một số với số đảo ngược của nó, kết quả thường có nhiều chữ số giống nhau? Ví dụ: {number} + {reversed_number} = {sum}",
                        "solution": "Khi cộng một số với số đảo ngược của nó, ta thực hiện phép cộng các chữ số ở vị trí đối xứng: {explanation}. Do tính chất đối xứng này, các chữ số ở vị trí đối xứng trong kết quả thường bằng nhau. Trong ví dụ: {number} + {reversed_number} = {sum}, ta thấy {pattern}."
                    }
                ],
                "weight": 0.15  # 300 bài
            },
            "Thơ toán học": {
                "templates": [
                    {
                        "template": "Có {total} quả cam ngon,\nChia đều cho {people} người em.\nMỗi người được mấy quả?\nCòn dư mấy quả cam?",
                        "solution": "Số cam chia được: {total} ÷ {people} = {quotient} quả mỗi người.\nSố cam còn dư: {total} - ({people} × {quotient}) = {remainder} quả."
                    },
                    {
                        "template": "Hình vuông có cạnh {side} phân,\nTính xem chu vi với diện tích là bao nhiêu?\nBốn cạnh bằng nhau thật đều,\nDiện tích bằng mấy, hãy điền vào ngay!",
                        "solution": "Chu vi hình vuông: 4 × {side} = {perimeter} phân.\nDiện tích hình vuông: {side} × {side} = {area} phân vuông."
                    },
                    {
                        "template": "Ao nhà nuôi vịt với gà,\n{total_animals} con đếm đủ, chân là {total_legs}.\nHỏi xem vịt có bao nhiêu?\nGà bao nhiêu để tính điều cho mau?",
                        "solution": "Gọi số vịt là x.\nSố gà là: {total_animals} - x\nPhương trình: 2x + 2({total_animals} - x) = {total_legs}\nVậy có {ducks} con vịt và {chickens} con gà."
                    }
                ],
                "weight": 0.1  # 200 bài
            },
            "Bài toán từ vựng toán học": {
                "templates": [
                    {
                        "template": "Một cửa hàng bán {item} với giá {price}đ một {unit}. Nếu mua {quantity} {unit}, sau đó được giảm {discount}%, hỏi phải trả bao nhiêu tiền?",
                        "solution": "Giá gốc: {price} × {quantity} = {original_cost}đ\nSố tiền được giảm: {original_cost} × {discount}/100 = {discount_amount}đ\nSố tiền phải trả: {original_cost} - {discount_amount} = {final_cost}đ"
                    },
                    {
                        "template": "Một bể chứa nước hình hộp chữ nhật có chiều dài {length}m, chiều rộng {width}m và chiều cao {height}m. Hỏi bể chứa được bao nhiêu lít nước?",
                        "solution": "Thể tích bể = dài × rộng × cao = {length} × {width} × {height} = {volume_m3}m³\nĐổi sang lít: {volume_m3}m³ = {volume_liters} lít (1m³ = 1000 lít)"
                    },
                    {
                        "template": "Một học sinh làm đúng {correct} câu trong bài kiểm tra có {total} câu. Tính tỷ lệ phần trăm số câu đúng của học sinh đó.",
                        "solution": "Tỷ lệ phần trăm = (Số câu đúng ÷ Tổng số câu) × 100\n= ({correct} ÷ {total}) × 100 = {percentage}%"
                    }
                ],
                "weight": 0.2  # 400 bài
            },
            "Câu hỏi trắc nghiệm": {
                "templates": [
                    {
                        "template": "Kết quả của phép tính {num1} + {num2} × {num3} là bao nhiêu?\nA. {wrong1}\nB. {correct}\nC. {wrong2}\nD. {wrong3}",
                        "solution": "Theo thứ tự ưu tiên, ta thực hiện phép nhân trước:\n{num2} × {num3} = {mult_result}\nSau đó cộng với {num1}:\n{num1} + {mult_result} = {correct}\nĐáp án đúng là B. {correct}"
                    },
                    {
                        "template": "Một hình vuông có chu vi {perimeter}cm. Diện tích của hình vuông này là bao nhiêu?\nA. {wrong1}cm²\nB. {wrong2}cm²\nC. {correct}cm²\nD. {wrong3}cm²",
                        "solution": "Chu vi hình vuông = 4 × cạnh\n{perimeter} = 4 × cạnh\nCạnh = {perimeter}/4 = {side}cm\nDiện tích = cạnh × cạnh = {side} × {side} = {correct}cm²\nĐáp án đúng là C. {correct}cm²"
                    },
                    {
                        "template": "{fraction1} + {fraction2} = ?\nA. {wrong1}\nB. {wrong2}\nC. {wrong3}\nD. {correct}",
                        "solution": "Quy đồng mẫu số: {process}\nKết quả: {correct}\nĐáp án đúng là D. {correct}"
                    }
                ],
                "weight": 0.15  # 300 bài
            }
        }

    def add_header(self):
        self.pdf.setFont('timesbd', 24)
        self.pdf.drawCentredString(self.width/2, self.height - 50, 'NHỮNG BÀI TOÁN CỔ ĐIỂN')
        self.pdf.setFont('times', 14)
        self.pdf.drawCentredString(self.width/2, self.height - 80, 'Tuyển tập 1500 bài toán đa dạng')
        self.y = self.height - 120

    def add_section(self, title: str):
        if self.y < 100:
            self.pdf.showPage()
            self.y = self.height - 50
        self.pdf.setFont('timesbd', 16)
        section_title = f'Phần {self.current_section}: {title}'
        self.pdf.drawString(50, self.y, section_title)
        self.y -= 30
        self.current_section += 1

    def add_problem(self, problem: str, solution: str = None):
        if self.y < 100:
            self.pdf.showPage()
            self.y = self.height - 50
        
        self.pdf.setFont('timesbd', 12)
        self.pdf.drawString(50, self.y, f'Bài {self.current_problem}:')
        self.y -= 20
        
        self.pdf.setFont('times', 12)
        lines = self._wrap_text(problem, 80)
        for line in lines:
            self.pdf.drawString(50, self.y, line)
            self.y -= 20
        
        if solution:
            self.pdf.setFont('timesi', 11)
            self.pdf.drawString(50, self.y, 'Hướng dẫn giải:')
            self.y -= 20
            lines = self._wrap_text(solution, 80)
            for line in lines:
                self.pdf.drawString(50, self.y, line)
                self.y -= 20
        
        self.y -= 10
        self.current_problem += 1

    def _wrap_text(self, text: str, width: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def generate_problem(self, problem_type: str) -> Dict:
        template_data = random.choice(self.problem_types[problem_type]["templates"])
        template = template_data["template"]
        solution_template = template_data["solution"]

        try:
            if problem_type == "Câu đố logic":
                if "nông dân" in template:
                    total_animals = random.randint(10, 30)
                    chickens = random.randint(total_animals//3, 2*total_animals//3)
                    rabbits = total_animals - chickens
                    total_legs = 2*chickens + 4*rabbits
                    return {
                        "question": template.format(total_animals=total_animals, total_legs=total_legs),
                        "solution": solution_template.format(total_animals=total_animals, total_legs=total_legs,
                                                          chickens=chickens, rabbits=rabbits)
                    }
                elif "cam" in template:
                    total_oranges = random.randint(100, 300)
                    buy_price = random.randint(5000, 10000)
                    sell_price = buy_price + random.randint(2000, 5000)
                    bad_oranges = random.randint(5, 20)
                    total_cost = total_oranges * buy_price
                    sold_oranges = total_oranges - bad_oranges
                    revenue = sold_oranges * sell_price
                    result = revenue - total_cost
                    profit_loss_text = "lãi" if result > 0 else "lỗ"
                    return {
                        "question": template.format(total_oranges=total_oranges, buy_price=buy_price,
                                                 sell_price=sell_price, bad_oranges=bad_oranges),
                        "solution": solution_template.format(total_oranges=total_oranges, buy_price=buy_price,
                                                          total_cost=total_cost, bad_oranges=bad_oranges,
                                                          sold_oranges=sold_oranges, sell_price=sell_price,
                                                          revenue=revenue, profit_loss_text=profit_loss_text,
                                                          result=abs(result))
                    }
                else:  # Bài toán Josephus
                    total_people = random.randint(5, 15)
                    count_to = random.randint(2, 5)
                    steps = self._solve_josephus(total_people, count_to)
                    result = steps[-1]
                    return {
                        "question": template.format(total_people=total_people, count_to=count_to),
                        "solution": solution_template.format(total_people=total_people, count_to=count_to,
                                                          steps=", ".join(map(str, steps)), result=result)
                    }
            elif problem_type == "Câu hỏi kiểu bài luận":
                if "chia" in template:
                    divisor = random.choice([3, 9])
                    remainder = 1
                    explanation = "3 và 9 là ước của 999...999" if divisor == 3 else "9 là ước của 999...999"
                    return {
                        "question": template.format(divisor=divisor),
                        "solution": solution_template.format(divisor=divisor, remainder=remainder,
                                                          explanation=explanation)
                    }
                elif "tam giác" in template:
                    angle = random.choice([30, 45, 60])
                    ratio = "1/2" if angle == 30 else "√2/2" if angle == 45 else "√3/2"
                    return {
                        "question": template.format(angle=angle, ratio=ratio),
                        "solution": solution_template.format(angle=angle, ratio=ratio)
                    }
                else:  # Bài toán số đảo ngược
                    number = random.randint(100, 999)
                    reversed_number = int(str(number)[::-1])
                    sum_result = number + reversed_number
                    explanation = f"Hàng đơn vị: {number%10} + {reversed_number%10} = {sum_result%10}"
                    pattern = f"các chữ số {', '.join(str(sum_result))}"
                    return {
                        "question": template.format(number=number, reversed_number=reversed_number,
                                                 sum=sum_result),
                        "solution": solution_template.format(number=number, reversed_number=reversed_number,
                                                          sum=sum_result, explanation=explanation,
                                                          pattern=pattern)
                    }
            elif problem_type == "Thơ toán học":
                if "cam" in template.lower():
                    total = random.randint(20, 50)
                    people = random.randint(3, 8)
                    quotient = total // people
                    remainder = total % people
                    return {
                        "question": template.format(total=total, people=people),
                        "solution": solution_template.format(total=total, people=people,
                                                          quotient=quotient, remainder=remainder)
                    }
                elif "hình vuông" in template.lower():
                    side = random.randint(5, 20)
                    perimeter = 4 * side
                    area = side * side
                    return {
                        "question": template.format(side=side),
                        "solution": solution_template.format(side=side, perimeter=perimeter, area=area)
                    }
                else:  # Bài toán vịt và gà
                    total_animals = random.randint(10, 30)
                    ducks = random.randint(total_animals//3, 2*total_animals//3)
                    chickens = total_animals - ducks
                    total_legs = 2*ducks + 2*chickens
                    return {
                        "question": template.format(total_animals=total_animals, total_legs=total_legs),
                        "solution": solution_template.format(total_animals=total_animals, total_legs=total_legs,
                                                          ducks=ducks, chickens=chickens)
                    }
            elif problem_type == "Bài toán từ vựng toán học":
                if "cửa hàng" in template:
                    item = random.choice(["cam", "quần", "áo", "giày", "túi"])
                    unit = "cái" if item in ["quần", "áo", "giày", "túi"] else "quả"
                    price = random.randint(10000, 50000)
                    quantity = random.randint(10, 50)
                    discount = random.randint(10, 50)
                    original_cost = price * quantity
                    discount_amount = int(original_cost * discount / 100)
                    final_cost = original_cost - discount_amount
                    return {
                        "question": template.format(item=item, price=price, unit=unit, quantity=quantity, discount=discount),
                        "solution": solution_template.format(price=price, quantity=quantity, original_cost=original_cost,
                                                          discount=discount, discount_amount=discount_amount,
                                                          final_cost=final_cost)
                    }
                elif "bể chứa" in template:
                    length = random.randint(1, 10)
                    width = random.randint(1, 10)
                    height = random.randint(1, 10)
                    volume_m3 = length * width * height
                    volume_liters = volume_m3 * 1000
                    return {
                        "question": template.format(length=length, width=width, height=height),
                        "solution": solution_template.format(length=length, width=width, height=height,
                                                          volume_m3=volume_m3, volume_liters=volume_liters)
                    }
                else:  # Bài toán tỷ lệ phần trăm
                    total = random.randint(20, 50)
                    correct = random.randint(1, total)
                    percentage = round((correct / total) * 100, 2)
                    return {
                        "question": template.format(correct=correct, total=total),
                        "solution": solution_template.format(correct=correct, total=total, percentage=percentage)
                    }
            elif problem_type == "Câu hỏi trắc nghiệm":
                if "phép tính" in template:
                    num1 = random.randint(1, 20)
                    num2 = random.randint(1, 10)
                    num3 = random.randint(1, 10)
                    mult_result = num2 * num3
                    correct = num1 + mult_result
                    wrong1 = correct + random.randint(1, 5)
                    wrong2 = correct - random.randint(1, 5)
                    wrong3 = correct + random.randint(6, 10)
                    return {
                        "question": template.format(num1=num1, num2=num2, num3=num3,
                                                 wrong1=wrong1, correct=correct,
                                                 wrong2=wrong2, wrong3=wrong3),
                        "solution": solution_template.format(num1=num1, num2=num2, num3=num3,
                                                          mult_result=mult_result, correct=correct)
                    }
                elif "hình vuông" in template:
                    perimeter = random.randint(20, 100)
                    side = perimeter // 4
                    correct = side * side
                    wrong1 = correct - random.randint(1, 10)
                    wrong2 = correct + random.randint(1, 10)
                    wrong3 = correct + random.randint(11, 20)
                    return {
                        "question": template.format(perimeter=perimeter, wrong1=wrong1,
                                                 wrong2=wrong2, correct=correct, wrong3=wrong3),
                        "solution": solution_template.format(perimeter=perimeter, side=side,
                                                          correct=correct)
                    }
                else:  # Bài toán phân số
                    num1, den1 = random.randint(1, 10), random.randint(2, 10)
                    num2, den2 = random.randint(1, 10), random.randint(2, 10)
                    fraction1 = f"{num1}/{den1}"
                    fraction2 = f"{num2}/{den2}"
                    # Tính tổng phân số
                    from math import gcd
                    lcm = (den1 * den2) // gcd(den1, den2)
                    num_result = (num1 * lcm // den1) + (num2 * lcm // den2)
                    gcd_result = gcd(num_result, lcm)
                    correct = f"{num_result//gcd_result}/{lcm//gcd_result}"
                    process = f"Quy đồng mẫu số {lcm}, ta có:\n{num1}×{lcm//den1}/{den1}×{lcm//den1} + {num2}×{lcm//den2}/{den2}×{lcm//den2}"
                    wrong1 = f"{num_result//gcd_result + 1}/{lcm//gcd_result}"
                    wrong2 = f"{num_result//gcd_result - 1}/{lcm//gcd_result}"
                    wrong3 = f"{num_result//gcd_result}/{lcm//gcd_result + 1}"
                    return {
                        "question": template.format(fraction1=fraction1, fraction2=fraction2,
                                                 wrong1=wrong1, wrong2=wrong2,
                                                 wrong3=wrong3, correct=correct),
                        "solution": solution_template.format(process=process, correct=correct)
                    }
        except Exception as e:
            print(f"Lỗi khi tạo bài toán {problem_type}: {str(e)}")
            return self.generate_problem(problem_type)  # Thử lại với template khác
        return None

    def _solve_josephus(self, n: int, k: int) -> List[int]:
        """Giải bài toán Josephus và trả về các bước giải."""
        steps = []
        pos = 1
        for i in range(1, n + 1):
            pos = ((pos + k - 1) % i) + 1
            steps.append(pos)
        return steps

    def generate_problems(self, total_count: int = 2000) -> List[Dict]:
        problems = []
        problem_types = list(self.problem_types.keys())
        
        # Tính số lượng bài toán cho mỗi loại
        type_counts = {}
        remaining_count = total_count
        
        for ptype in problem_types[:-1]:
            count = round(total_count * self.problem_types[ptype]["weight"])
            type_counts[ptype] = count
            remaining_count -= count
        
        type_counts[problem_types[-1]] = remaining_count
        
        # Tạo bài toán theo số lượng đã tính
        for ptype, count in type_counts.items():
            for i in range(count):
                problem = self.generate_problem(ptype)
                if problem:
                    problems.append({
                        "id": len(problems) + 1,
                        "type": ptype,
                        "question": problem["question"],
                        "solution": problem["solution"],
                        "difficulty": self._estimate_difficulty(problem["question"], problem["solution"]),
                        "tags": self._extract_tags(problem["question"])
                    })
        
        random.shuffle(problems)
        return problems

    def _estimate_difficulty(self, question: str, solution: str) -> str:
        # Ước lượng độ khó dựa trên độ dài và độ phức tạp
        complexity = len(solution.split()) / len(question.split()) if question else 1
        if complexity > 2:
            return "Khó"
        elif complexity > 1.5:
            return "Trung bình"
        return "Dễ"

    def _extract_tags(self, question: str) -> List[str]:
        keywords = {
            'số học': ['số', 'chữ số', 'ước số', 'bội số', 'tổng', 'hiệu', 'tích', 'thương', 'chia'],
            'hình học': ['tam giác', 'hình vuông', 'hình chữ nhật', 'diện tích', 'chu vi', 'cạnh'],
            'đại số': ['phương trình', 'biểu thức', 'số x', 'nghiệm', 'tỷ số'],
            'logic': ['nếu', 'thì', 'hoặc', 'và', 'suy ra', 'chứng minh', 'giải thích'],
            'thực tế': ['tiền', 'tuổi', 'giờ', 'ngày', 'tháng', 'năm', 'cam', 'gà', 'vịt', 'thỏ']
        }
        
        tags = []
        text = question.lower()
        for category, words in keywords.items():
            if any(word in text for word in words):
                tags.append(category)
        return tags

    def save_problems(self, problems: List[Dict]):
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs('db/questions', exist_ok=True)
            print(f"✓ Đã tạo thư mục db/questions")
            
            filepath = 'db/questions/problems.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "questions": problems
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ Đã tạo và lưu {len(problems)} bài toán vào {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu file: {str(e)}")

    def generate_all_problems(self):
        self.add_header()
        
        # Phân bổ số lượng bài toán cho mỗi loại
        self.generate_problems(300)
        
        # Lưu file PDF
        self.pdf.save()

if __name__ == '__main__':
    try:
        generator = ProblemGenerator()
        print("✓ Đã khởi tạo generator")
        problems = generator.generate_problems(1500)  # Tạo 800 bài toán
        print(f"✓ Đã tạo {len(problems)} bài toán")
        generator.save_problems(problems)
    except Exception as e:
        print(f"Lỗi: {str(e)}") 