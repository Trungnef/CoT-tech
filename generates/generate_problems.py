# --- START OF FILE generate_problems_improved.py ---

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import random
from typing import List, Dict, Tuple
import math
import json
import os

## <--- CẢI TIẾN: Di chuyển import lên đầu file cho đúng chuẩn
from math import gcd

class ProblemGenerator:
    """
    Lớp này dùng để tạo ra một tuyển tập các bài toán đa dạng,
    lưu kết quả dưới dạng file PDF và file JSON.
    Các câu hỏi và lời giải đã được tối ưu hóa cho đối tượng học sinh tiểu học.
    """
    def __init__(self):
        # Lưu ý: Đường dẫn font cứng có thể không hoạt động trên các máy tính khác.
        # Cân nhắc sử dụng các font chuẩn của PDF hoặc đóng gói font cùng với mã nguồn.
        font_path = r'C:\Windows\Fonts\times.ttf'
        if not os.path.exists(font_path):
            print(f"Cảnh báo: Không tìm thấy font tại '{font_path}'. Sẽ sử dụng font mặc định.")
            # Đăng ký font thay thế nếu không tìm thấy
            pdfmetrics.registerFont(TTFont('times', 'Helvetica'))
            pdfmetrics.registerFont(TTFont('timesbd', 'Helvetica-Bold'))
            pdfmetrics.registerFont(TTFont('timesi', 'Helvetica-Oblique'))
        else:
            pdfmetrics.registerFont(TTFont('times', font_path))
            pdfmetrics.registerFont(TTFont('timesbd', r'C:\Windows\Fonts\timesbd.ttf'))
            pdfmetrics.registerFont(TTFont('timesi', r'C:\Windows\Fonts\timesi.ttf'))

        output_dir = './db'
        os.makedirs(output_dir, exist_ok=True)
        self.pdf = canvas.Canvas(os.path.join(output_dir, 'Nhung_bai_toan_co_dien.pdf'), pagesize=A4)

        self.width, self.height = A4
        self.current_section = 1
        self.current_problem = 1
        self.y = self.height - 50
        
        ## <--- CẢI TIẾN: Cập nhật và thêm mới các dạng bài toán
        self.problem_types = {
            "Câu đố logic": {
                "templates": [
                    {
                        "template": "Vừa gà vừa thỏ, bó lại cho tròn, ba mươi sáu con, một trăm chân chẵn. Hỏi có bao nhiêu con gà, bao nhiêu con thỏ?",
                        "solution": ("Đây là dạng toán 'giả thiết tạm'.\n"
                                     "Giả sử tất cả 36 con đều là thỏ. Khi đó, tổng số chân là: 36 × 4 = 144 (chân).\n"
                                     "Số chân dôi ra so với thực tế là: 144 - 100 = 44 (chân).\n"
                                     "Số chân dôi ra này là do ta đã thay mỗi con gà (2 chân) bằng một con thỏ (4 chân). Mỗi lần thay như vậy, số chân tăng thêm là: 4 - 2 = 2 (chân).\n"
                                     "Vậy số con gà là: 44 ÷ 2 = 22 (con).\n"
                                     "Số con thỏ là: 36 - 22 = 14 (con).\n"
                                     "Đáp số: 22 con gà, 14 con thỏ.")
                    },
                    {
                        "template": "Một người bán cam mua {total_oranges} quả cam với giá {buy_price} đồng một quả. Người đó bán lại với giá {sell_price} đồng một quả nhưng có {bad_oranges} quả bị hỏng không bán được. Hỏi sau khi bán hết số cam, người đó lãi hay lỗ bao nhiêu tiền?",
                        "solution": ("Tổng số vốn bỏ ra để mua cam là: {total_oranges} × {buy_price} = {total_cost} đồng.\n"
                                     "Số cam còn lại có thể bán được là: {total_oranges} - {bad_oranges} = {sold_oranges} quả.\n"
                                     "Số tiền thu được sau khi bán hết cam là: {sold_oranges} × {sell_price} = {revenue} đồng.\n"
                                     "Ta so sánh tiền thu được và tiền vốn: {revenue} đồng so với {total_cost} đồng.\n"
                                     "Vậy, người đó {profit_loss_text} số tiền là: |{revenue} - {total_cost}| = {result} đồng.")
                    }
                ],
                "weight": 0.15 # 225 bài
            },
            "Bài toán đố điển hình (tiểu học)": { ## <--- CẢI TIẾN: Thêm nhóm bài toán mới
                "templates": [
                    {
                        "template": "Tìm hai số khi biết tổng của chúng là {sum} và hiệu của chúng là {diff}. Hỏi số lớn và số bé là bao nhiêu?",
                        "solution": ("Đây là dạng toán tìm hai số khi biết tổng và hiệu.\n"
                                     "Áp dụng công thức, ta có:\n"
                                     "Số lớn = (Tổng + Hiệu) ÷ 2 = ({sum} + {diff}) ÷ 2 = {large_num}.\n"
                                     "Số bé = (Tổng - Hiệu) ÷ 2 = ({sum} - {diff}) ÷ 2 = {small_num}.\n"
                                     "Thử lại: {large_num} + {small_num} = {sum} (đúng) và {large_num} - {small_num} = {diff} (đúng).\n"
                                     "Đáp số: Số lớn là {large_num}, số bé là {small_num}.")
                    },
                    {
                        "template": "Tổng của hai số là {sum}. Tỉ số của hai số đó là {ratio_num}/{ratio_den}. Tìm hai số đó.",
                        "solution": ("Đây là dạng toán tìm hai số khi biết tổng và tỉ số.\n"
                                     "Ta có sơ đồ:\n"
                                     "Số bé: |---|...| ({ratio_num} phần)\n"
                                     "Số lớn: |---|...|---| ({ratio_den} phần)\n"
                                     "Tổng số phần bằng nhau là: {ratio_num} + {ratio_den} = {total_parts} (phần).\n"
                                     "Giá trị của một phần là: {sum} ÷ {total_parts} = {part_value}.\n"
                                     "Số bé là: {part_value} × {ratio_num} = {small_num}.\n"
                                     "Số lớn là: {part_value} × {ratio_den} = {large_num}.\n"
                                     "Đáp số: Hai số cần tìm là {small_num} và {large_num}.")
                    },
                    {
                        "template": "Hiện nay mẹ hơn con {age_diff} tuổi. {years_ago} năm trước, tổng số tuổi của hai mẹ con là {past_total_age}. Tính tuổi của mỗi người hiện nay.",
                        "solution": ("Hiệu số tuổi của hai mẹ con không thay đổi theo thời gian, vậy hiện nay mẹ vẫn hơn con {age_diff} tuổi.\n"
                                     "Tổng số tuổi của hai mẹ con hiện nay là: {past_total_age} + {years_ago} + {years_ago} = {current_total_age}.\n"
                                     "Bài toán trở về dạng tìm hai số khi biết tổng ({current_total_age}) và hiệu ({age_diff}).\n"
                                     "Tuổi của con hiện nay là: ({current_total_age} - {age_diff}) ÷ 2 = {child_age} (tuổi).\n"
                                     "Tuổi của mẹ hiện nay là: {child_age} + {age_diff} = {mom_age} (tuổi).\n"
                                     "Đáp số: Hiện nay con {child_age} tuổi, mẹ {mom_age} tuổi.")
                    }
                ],
                "weight": 0.25 # 375 bài
            },
            "Câu hỏi giải thích, suy luận": { ## <--- CẢI TIẾN: Đổi tên và nội dung cho phù hợp
                "templates": [
                    {
                        "template": "Hãy giải thích tại sao một số có tổng các chữ số chia hết cho 9 thì số đó cũng chia hết cho 9?",
                        "solution": ("Để chứng minh điều này, ta phân tích cấu tạo của số. Ví dụ, xét số 126.\n"
                                     "Ta có: 126 = 100 + 20 + 6 = 1×100 + 2×10 + 6.\n"
                                     "Mà 100 = 99 + 1, và 10 = 9 + 1.\n"
                                     "Nên 126 = 1×(99+1) + 2×(9+1) + 6 = (1×99 + 1) + (2×9 + 2) + 6.\n"
                                     "Nhóm lại: 126 = (1×99 + 2×9) + (1 + 2 + 6).\n"
                                     "Ta thấy (1×99 + 2×9) là một số chắc chắn chia hết cho 9.\n"
                                     "Phần còn lại là (1 + 2 + 6) chính là tổng các chữ số của 126.\n"
                                     "Vì vậy, để 126 chia hết cho 9 thì tổng các chữ số (1+2+6=9) cũng phải chia hết cho 9.\n"
                                     "Lập luận này đúng với mọi số tự nhiên.")
                    },
                    {
                        "template": "Tại sao khi nhân một số tự nhiên với 10, ta chỉ cần viết thêm một chữ số 0 vào bên phải số đó? Ví dụ: 25 x 10 = 250.",
                        "solution": ("Điều này dựa trên hệ đếm thập phân của chúng ta.\n"
                                     "Mỗi chữ số trong một số có một giá trị tùy theo vị trí của nó (hàng đơn vị, hàng chục, hàng trăm,...).\n"
                                     "Ví dụ, số 25 có nghĩa là 2 chục và 5 đơn vị (2×10 + 5).\n"
                                     "Khi ta nhân số 25 với 10:\n"
                                     "25 × 10 = (2×10 + 5) × 10 = 2×10×10 + 5×10 = 2×100 + 5×10.\n"
                                     "Kết quả này tương ứng với một số có 2 trăm, 5 chục và 0 đơn vị. Đó chính là số 250.\n"
                                     "Việc thêm chữ số 0 vào cuối đã dịch chuyển tất cả các chữ số khác sang trái một hàng, làm cho giá trị của chúng tăng lên 10 lần.")
                    }
                ],
                "weight": 0.10 # 150 bài
            },
            "Thơ toán học": {
                "templates": [
                    {
                        "template": "Có {total} quả cam ngon,\nChia đều cho {people} bạn nhỏ.\nMỗi bạn được mấy quả?\nCòn dư mấy quả cam?",
                        "solution": ("Đây là bài toán chia có dư.\n"
                                     "Ta thực hiện phép chia: {total} ÷ {people}.\n"
                                     "Thương của phép chia là số quả cam mỗi bạn nhận được: {total} // {people} = {quotient} (quả).\n"
                                     "Số dư của phép chia là số cam còn lại: {total} % {people} = {remainder} (quả).\n"
                                     "Đáp số: Mỗi bạn được {quotient} quả, còn dư {remainder} quả.")
                    },
                    {
                        "template": "Một hình chữ nhật xinh xinh,\nChiều dài {length} mét, rộng {width} mét mình cùng đo.\nHỏi chu vi với diện tích,\nLà bao nhiêu mét, bé tính cho nhanh?",
                        "solution": ("Chu vi của hình chữ nhật được tính bằng công thức: (dài + rộng) × 2.\n"
                                     "Chu vi là: ({length} + {width}) × 2 = {perimeter} (mét).\n"
                                     "Diện tích của hình chữ nhật được tính bằng công thức: dài × rộng.\n"
                                     "Diện tích là: {length} × {width} = {area} (mét vuông).\n"
                                     "Đáp số: Chu vi {perimeter}m, Diện tích {area}m².")
                    },
                    {
                        ## <--- CẢI TIẾN: Sửa bài toán Gà/Vịt (vốn sai logic) thành Gà/Chó
                        "template": "Gà, Chó đứng chung một sân,\nTổng cộng {total_animals} con, chân đếm được {total_legs}.\nĐố bạn tính được cho rành,\nMấy con gà, mấy con chó trong sân?",
                        "solution": ("Đây là bài toán 'giả thiết tạm'.\n"
                                     "Giả sử tất cả {total_animals} con đều là chó. Tổng số chân sẽ là: {total_animals} × 4 = {assumed_legs_all_dogs} (chân).\n"
                                     "Số chân dôi ra so với đề bài là: {assumed_legs_all_dogs} - {total_legs} = {extra_legs} (chân).\n"
                                     "Sở dĩ số chân dôi ra vì ta đã thay mỗi con gà (2 chân) bằng một con chó (4 chân). Mỗi lần thay như vậy, số chân tăng thêm: 4 - 2 = 2 (chân).\n"
                                     "Vậy, số con gà là: {extra_legs} ÷ 2 = {chickens} (con).\n"
                                     "Số con chó là: {total_animals} - {chickens} = {dogs} (con).\n"
                                     "Đáp số: {chickens} con gà, {dogs} con chó.")
                    }
                ],
                "weight": 0.15 # 225 bài
            },
            "Bài toán từ vựng toán học": {
                "templates": [
                    {
                        "template": "Một cửa hàng bán {item} với giá {price} đồng một {unit}. Nếu mua {quantity} {unit} thì sẽ được giảm giá {discount}%. Hỏi người mua phải trả bao nhiêu tiền?",
                        "solution": ("Giá tiền ban đầu khi chưa giảm giá là: {price} × {quantity} = {original_cost} đồng.\n"
                                     "Số tiền được giảm giá là: {original_cost} × {discount} / 100 = {discount_amount} đồng.\n"
                                     "Số tiền cuối cùng người mua phải trả là: {original_cost} - {discount_amount} = {final_cost} đồng.\n"
                                     "Đáp số: {final_cost} đồng.")
                    },
                    {
                        "template": "Một bể chứa nước hình hộp chữ nhật có chiều dài {length}m, chiều rộng {width}m và chiều cao {height}m. Hỏi bể này có thể chứa được tối đa bao nhiêu lít nước? (Biết 1m³ = 1000 lít)",
                        "solution": ("Thể tích của bể nước được tính bằng công thức: dài × rộng × cao.\n"
                                     "Thể tích bể là: {length} × {width} × {height} = {volume_m3} (mét khối).\n"
                                     "Để đổi từ mét khối sang lít, ta nhân với 1000: {volume_m3} × 1000 = {volume_liters} (lít).\n"
                                     "Đáp số: Bể chứa được {volume_liters} lít nước.")
                    },
                    {
                        "template": "Trong một bài kiểm tra có {total} câu hỏi, bạn An làm đúng được {correct} câu. Hỏi tỉ lệ phần trăm số câu An làm đúng là bao nhiêu?",
                        "solution": ("Để tìm tỉ lệ phần trăm, ta lấy số câu đúng chia cho tổng số câu rồi nhân với 100.\n"
                                     "Tỉ lệ phần trăm = (Số câu đúng / Tổng số câu) × 100\n"
                                     "= ({correct} / {total}) × 100 = {percentage}%.\n"
                                     "Đáp số: An làm đúng {percentage}% số câu.")
                    }
                ],
                "weight": 0.20 # 300 bài
            },
            "Câu hỏi trắc nghiệm": {
                "templates": [
                    {
                        # Sửa template để dùng các lựa chọn chung
                        "template": "Kết quả của phép tính {num1} + {num2} × {num3} là bao nhiêu?\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}",
                        "solution": ("Theo quy tắc 'nhân chia trước, cộng trừ sau', ta thực hiện phép nhân trước:\n"
                                     "{num2} × {num3} = {mult_result}.\n"
                                     "Sau đó thực hiện phép cộng:\n"
                                     "{num1} + {mult_result} = {correct}.\n"
                                     "Vậy, đáp án đúng là {correct_choice}: {correct}.")
                    },
                    {
                        # Sửa template để dùng các lựa chọn chung
                        "template": "Một hình vuông có chu vi {perimeter}cm. Diện tích của hình vuông đó là:\nA. {choice_a}cm²\nB. {choice_b}cm²\nC. {choice_c}cm²\nD. {choice_d}cm²",
                        "solution": ("Chu vi hình vuông bằng độ dài một cạnh nhân với 4.\n"
                                     "Độ dài cạnh của hình vuông là: {perimeter} ÷ 4 = {side} (cm).\n"
                                     "Diện tích hình vuông bằng độ dài một cạnh nhân với chính nó:\n"
                                     "Diện tích = {side} × {side} = {correct} (cm²).\n"
                                     "Vậy, đáp án đúng là {correct_choice}: {correct}cm².")
                    },
                    {
                        # Sửa template để dùng các lựa chọn chung
                        "template": "Kết quả của phép cộng hai phân số {fraction1} + {fraction2} là:\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}",
                        "solution": ("Để cộng hai phân số, trước hết ta phải quy đồng mẫu số. Mẫu số chung nhỏ nhất của {den1} và {den2} là {lcm}.\n"
                                     "Ta có: {fraction1} = {num1_new}/{lcm} và {fraction2} = {num2_new}/{lcm}.\n"
                                     "Vậy: {fraction1} + {fraction2} = {num1_new}/{lcm} + {num2_new}/{lcm} = {sum_num}/{lcm}.\n"
                                     "Rút gọn phân số (nếu có thể), ta được kết quả cuối cùng là {correct}.\n"
                                     "Vậy, đáp án đúng là {correct_choice}: {correct}.")
                    }
                ],
                "weight": 0.15 # 225 bài
            }
        }
    # ... (Các hàm add_header, add_section, add_problem, _wrap_text không thay đổi) ...

    def add_header(self):
        self.pdf.setFont('timesbd', 24)
        self.pdf.drawCentredString(self.width/2, self.height - 50, 'NHỮNG BÀI TOÁN CỔ ĐIỂN')
        self.pdf.setFont('times', 14)
        self.pdf.drawCentredString(self.width/2, self.height - 80, 'Tuyển tập 1000 bài toán đa dạng tiểu học')
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
        if self.y < 150: # Tăng khoảng trống để tránh cắt nội dung
            self.pdf.showPage()
            self.y = self.height - 50
        
        self.pdf.setFont('timesbd', 12)
        problem_title = f'Bài {self.current_problem}:'
        lines = self._wrap_text(problem, 90)
        
        self.pdf.drawString(50, self.y, problem_title)
        self.y -= 20
        
        self.pdf.setFont('times', 12)
        for line in lines:
            self.pdf.drawString(70, self.y, line)
            self.y -= 15
        
        self.y -= 5
        
        if solution:
            self.pdf.setFont('timesi', 11)
            self.pdf.drawString(50, self.y, 'Hướng dẫn giải:')
            self.y -= 20
            solution_lines = self._wrap_text(solution, 85)
            for line in solution_lines:
                self.pdf.drawString(70, self.y, line)
                self.y -= 15
        
        self.y -= 20
        self.current_problem += 1

    def _wrap_text(self, text: str, width: int) -> List[str]:
        output_lines = []
        # Tách theo ký tự xuống dòng trước
        paragraphs = text.split('\n')
        for para in paragraphs:
            words = para.split()
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= width:
                    current_line.append(word)
                else:
                    output_lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                output_lines.append(' '.join(current_line))
        return output_lines

    ## <--- CẢI TIẾN: Cập nhật toàn bộ hàm generate_problem
    def generate_problem(self, problem_type: str) -> Dict:
        template_data = random.choice(self.problem_types[problem_type]["templates"])
        template = template_data["template"]
        solution_template = template_data["solution"]

        try:
            if problem_type == "Câu đố logic":
                if "gà" in template: # Bài toán gà thỏ
                    # Đã có sẵn template, không cần sinh số ngẫu nhiên
                    return { "question": template, "solution": solution_template }
                elif "cam" in template:
                    total_oranges = random.randint(50, 200)
                    buy_price = random.randint(2, 5) * 1000
                    sell_price = buy_price + random.randint(1, 3) * 1000
                    bad_oranges = random.randint(5, total_oranges // 10)
                    total_cost = total_oranges * buy_price
                    sold_oranges = total_oranges - bad_oranges
                    revenue = sold_oranges * sell_price
                    result = abs(revenue - total_cost)
                    profit_loss_text = "lãi" if revenue > total_cost else "lỗ"
                    return {
                        "question": template.format(total_oranges=total_oranges, buy_price=buy_price, sell_price=sell_price, bad_oranges=bad_oranges),
                        "solution": solution_template.format(total_oranges=total_oranges, buy_price=buy_price, total_cost=total_cost, bad_oranges=bad_oranges, sold_oranges=sold_oranges, sell_price=sell_price, revenue=revenue, profit_loss_text=profit_loss_text, result=result)
                    }

            elif problem_type == "Bài toán đố điển hình (tiểu học)":
                if "tổng" in template and "hiệu" in template:
                    small_num = random.randint(10, 50)
                    diff = random.randint(2, 20) * 2 # Đảm bảo hiệu là số chẵn để tổng và hiệu cùng tính chẵn lẻ
                    large_num = small_num + diff
                    sum_val = large_num + small_num
                    return {
                        "question": template.format(sum=sum_val, diff=diff),
                        "solution": solution_template.format(sum=sum_val, diff=diff, large_num=large_num, small_num=small_num)
                    }
                elif "tỉ số" in template:
                    ratio_num = random.randint(1, 4)
                    ratio_den = random.randint(ratio_num + 1, 7)
                    part_value = random.randint(5, 20)
                    small_num = ratio_num * part_value
                    large_num = ratio_den * part_value
                    sum_val = small_num + large_num
                    total_parts = ratio_num + ratio_den
                    return {
                        "question": template.format(sum=sum_val, ratio_num=ratio_num, ratio_den=ratio_den),
                        "solution": solution_template.format(sum=sum_val, ratio_num=ratio_num, ratio_den=ratio_den, total_parts=total_parts, part_value=part_value, small_num=small_num, large_num=large_num)
                    }
                elif "tuổi" in template:
                    child_age = random.randint(5, 12)
                    age_diff = random.randint(20, 30)
                    mom_age = child_age + age_diff
                    years_ago = random.randint(1, child_age - 1)
                    past_total_age = (mom_age - years_ago) + (child_age - years_ago)
                    current_total_age = mom_age + child_age
                    return {
                        "question": template.format(age_diff=age_diff, years_ago=years_ago, past_total_age=past_total_age),
                        "solution": solution_template.format(age_diff=age_diff, years_ago=years_ago, past_total_age=past_total_age, current_total_age=current_total_age, child_age=child_age, mom_age=mom_age)
                    }

            elif problem_type == "Câu hỏi giải thích, suy luận":
                # Các template này không cần sinh số ngẫu nhiên
                return { "question": template, "solution": solution_template }

            elif problem_type == "Thơ toán học":
                if "cam" in template:
                    total = random.randint(20, 50)
                    people = random.randint(3, 8)
                    quotient = total // people
                    remainder = total % people
                    return {
                        "question": template.format(total=total, people=people),
                        "solution": solution_template.format(total=total, people=people, quotient=quotient, remainder=remainder)
                    }
                elif "chữ nhật" in template:
                    length = random.randint(10, 30)
                    width = random.randint(5, length - 2)
                    perimeter = 2 * (length + width)
                    area = length * width
                    return {
                        "question": template.format(length=length, width=width),
                        "solution": solution_template.format(length=length, width=width, perimeter=perimeter, area=area)
                    }
                elif "Chó" in template:
                    chickens = random.randint(10, 30)
                    dogs = random.randint(5, 20)
                    total_animals = chickens + dogs
                    total_legs = chickens * 2 + dogs * 4
                    assumed_legs_all_dogs = total_animals * 4
                    extra_legs = assumed_legs_all_dogs - total_legs
                    return {
                        "question": template.format(total_animals=total_animals, total_legs=total_legs),
                        "solution": solution_template.format(total_animals=total_animals, total_legs=total_legs, assumed_legs_all_dogs=assumed_legs_all_dogs, extra_legs=extra_legs, chickens=chickens, dogs=dogs)
                    }

            elif problem_type == "Bài toán từ vựng toán học":
                if "cửa hàng" in template:
                    item = random.choice(["quyển vở", "cái bút", "hộp màu", "cái cặp"])
                    unit = "cái"
                    if "vở" in item: unit = "quyển"
                    elif "màu" in item: unit = "hộp"
                    price = random.randint(5, 50) * 1000
                    quantity = random.randint(2, 10)
                    discount = random.choice([10, 15, 20, 25])
                    original_cost = price * quantity
                    discount_amount = int(original_cost * discount / 100)
                    final_cost = original_cost - discount_amount
                    return {
                        "question": template.format(item=item, price=price, unit=unit, quantity=quantity, discount=discount),
                        "solution": solution_template.format(price=price, quantity=quantity, original_cost=original_cost, discount=discount, discount_amount=discount_amount, final_cost=final_cost)
                    }
                elif "bể chứa" in template:
                    length = random.randint(1, 5)
                    width = random.randint(1, 4)
                    height = random.randint(1, 3)
                    volume_m3 = length * width * height
                    volume_liters = volume_m3 * 1000
                    return {
                        "question": template.format(length=length, width=width, height=height),
                        "solution": solution_template.format(length=length, width=width, height=height, volume_m3=volume_m3, volume_liters=volume_liters)
                    }
                elif "tỉ lệ" in template:
                    total = random.choice([10, 20, 25, 40, 50])
                    correct = random.randint(total // 2, total-1)
                    percentage = round((correct / total) * 100, 2)
                    return {
                        "question": template.format(correct=correct, total=total),
                        "solution": solution_template.format(correct=correct, total=total, percentage=percentage)
                    }

            elif problem_type == "Câu hỏi trắc nghiệm":
                if "phép tính" in template:
                    num1 = random.randint(1, 50)
                    num2 = random.randint(2, 10)
                    num3 = random.randint(2, 10)
                    mult_result = num2 * num3
                    correct = num1 + mult_result

                    # Tạo các lựa chọn sai và gộp vào một danh sách
                    choices = {correct}
                    while len(choices) < 4:
                        offset = random.randint(-10, 10)
                        if offset == 0: continue
                        # Đảm bảo lựa chọn không bị âm
                        wrong_choice = correct + offset
                        if wrong_choice >= 0:
                            choices.add(wrong_choice)
                    
                    shuffled_choices = list(choices)
                    random.shuffle(shuffled_choices)
                    correct_choice = "ABCD"[shuffled_choices.index(correct)]

                    return {
                        "question": template.format(num1=num1, num2=num2, num3=num3,
                                                 choice_a=shuffled_choices[0],
                                                 choice_b=shuffled_choices[1],
                                                 choice_c=shuffled_choices[2],
                                                 choice_d=shuffled_choices[3]),
                        "solution": solution_template.format(num1=num1, num2=num2, num3=num3,
                                                          mult_result=mult_result, correct=correct,
                                                          correct_choice=correct_choice)
                    }
                elif "hình vuông" in template:
                    side = random.randint(5, 25)
                    perimeter = side * 4
                    correct = side * side

                    choices = {correct}
                    while len(choices) < 4:
                        offset = random.randint(-20, 20)
                        if offset == 0: continue
                        wrong_choice = correct + offset
                        if wrong_choice > 0:
                            choices.add(wrong_choice)

                    shuffled_choices = list(choices)
                    random.shuffle(shuffled_choices)
                    correct_choice = "ABCD"[shuffled_choices.index(correct)]

                    return {
                        "question": template.format(perimeter=perimeter,
                                                 choice_a=shuffled_choices[0],
                                                 choice_b=shuffled_choices[1],
                                                 choice_c=shuffled_choices[2],
                                                 choice_d=shuffled_choices[3]),
                        "solution": solution_template.format(perimeter=perimeter, side=side,
                                                          correct=correct, correct_choice=correct_choice)
                    }
                else:  # Bài toán phân số
                    den1, den2 = random.randint(2, 10), random.randint(2, 10)
                    num1, num2 = random.randint(1, den1), random.randint(1, den2)
                    fraction1 = f"{num1}/{den1}"
                    fraction2 = f"{num2}/{den2}"
                    lcm = (den1 * den2) // gcd(den1, den2)
                    num1_new = num1 * (lcm // den1)
                    num2_new = num2 * (lcm // den2)
                    sum_num = num1_new + num2_new
                    common_divisor = gcd(sum_num, lcm)
                    correct_num = sum_num // common_divisor
                    correct_den = lcm // common_divisor
                    
                    # Xử lý trường hợp mẫu số là 1
                    if correct_den == 1:
                        correct = f"{correct_num}"
                    else:
                        correct = f"{correct_num}/{correct_den}"

                    choices = {correct}
                    while len(choices) < 4:
                        # Tạo các đáp án sai hợp lý
                        if random.random() < 0.5: # Cộng tử với tử, mẫu với mẫu (lỗi sai phổ biến)
                             common = gcd(num1+num2, den1+den2)
                             choices.add(f"{(num1+num2)//common}/{(den1+den2)//common}")
                        else: # Sai khi quy đồng hoặc rút gọn
                             wrong_num = sum_num + random.randint(-2, 2)
                             wrong_den = lcm + random.randint(-1, 1)
                             if wrong_den == 0: continue
                             common = gcd(wrong_num, wrong_den)
                             choices.add(f"{wrong_num//common}/{wrong_den//common}")
                    
                    shuffled_choices = list(choices)
                    random.shuffle(shuffled_choices)
                    correct_choice = "ABCD"[shuffled_choices.index(correct)]

                    return {
                        "question": template.format(fraction1=fraction1, fraction2=fraction2,
                                                 choice_a=shuffled_choices[0],
                                                 choice_b=shuffled_choices[1],
                                                 choice_c=shuffled_choices[2],
                                                 choice_d=shuffled_choices[3]),
                        "solution": solution_template.format(fraction1=fraction1, fraction2=fraction2,
                                                           den1=den1, den2=den2, lcm=lcm,
                                                           num1_new=num1_new, num2_new=num2_new,
                                                           sum_num=sum_num, correct=correct,
                                                           correct_choice=correct_choice)
                    }
        except Exception as e:
            print(f"Lỗi khi tạo bài toán {problem_type}: {str(e)}")
            # Thử lại với template/số liệu khác để tránh vòng lặp vô hạn nếu có lỗi logic
            return self.generate_problem(problem_type)  
        return None
    
    # ... (Các hàm _estimate_difficulty, _extract_tags không thay đổi) ...

    def generate_problems(self, total_count: int = 1000) -> List[Dict]:
        problems = []
        problem_types_list = list(self.problem_types.keys())
        weights = [self.problem_types[ptype]["weight"] for ptype in problem_types_list]
        
        # Tạo bài toán dựa trên phân phối trọng số
        for i in range(total_count):
            ptype = random.choices(problem_types_list, weights=weights, k=1)[0]
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
        
        # Không xáo trộn để các bài cùng loại có thể gần nhau, dễ theo dõi hơn
        random.shuffle(problems) 
        return problems

    def _estimate_difficulty(self, question: str, solution: str) -> str:
        # Ước lượng độ khó dựa trên độ dài và độ phức tạp
        if not question or not solution: return "Dễ"
        complexity = len(solution.split()) / len(question.split())
        length_factor = len(solution.split())

        if complexity > 3.0 or length_factor > 80:
            return "Khó"
        elif complexity > 1.8 or length_factor > 40:
            return "Trung bình"
        return "Dễ"

    def _extract_tags(self, question: str) -> List[str]:
        keywords = {
            'số học': ['số', 'chữ số', 'tổng', 'hiệu', 'tích', 'thương', 'chia', 'phân số'],
            'hình học': ['tam giác', 'hình vuông', 'hình chữ nhật', 'diện tích', 'chu vi', 'cạnh', 'mét'],
            'đại số': ['tìm hai số', 'tỉ số', 'phần trăm'],
            'logic': ['giả sử', 'suy ra', 'chứng minh', 'giải thích', 'tại sao'],
            'thực tế': ['tiền', 'tuổi', 'giờ', 'cam', 'gà', 'chó', 'bể nước', 'cửa hàng']
        }
        
        tags = []
        text = question.lower()
        for category, words in keywords.items():
            if any(word in text for word in words):
                tags.append(category)
        if not tags: tags.append("tổng hợp")
        return list(set(tags))


    def save_problems(self, problems: List[Dict]):
        try:
            # Tạo thư mục nếu chưa tồn tại
            output_dir = './db/questions'
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Đã tạo/kiểm tra thư mục {output_dir}")
            
            filepath = os.path.join(output_dir, 'problems.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "questions": problems
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ Đã tạo và lưu {len(problems)} bài toán vào {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu file JSON: {str(e)}")

    def generate_all_problems_pdf(self, problems: List[Dict]):
        self.add_header()
        
        sections = {}
        for p in problems:
            if p['type'] not in sections:
                sections[p['type']] = []
            sections[p['type']].append(p)
            
        for section_title, problem_list in sections.items():
            self.add_section(section_title)
            for p in problem_list:
                self.add_problem(p['question'], p['solution'])

        try:
            self.pdf.save()
            print(f"✓ Đã lưu thành công file PDF.")
        except Exception as e:
            print(f"Lỗi khi lưu file PDF: {str(e)}")

if __name__ == '__main__':
    try:
        generator = ProblemGenerator()
        print("✓ Đã khởi tạo generator")
        
        problems = generator.generate_problems(1099)
        print(f"✓ Đã tạo {len(problems)} bài toán trong bộ nhớ")
        
        generator.save_problems(problems)
        
        # Tạo lại generator để bắt đầu file PDF mới
        pdf_generator = ProblemGenerator()
        pdf_generator.generate_all_problems_pdf(problems)
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình thực thi: {str(e)}")