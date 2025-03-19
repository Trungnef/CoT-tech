from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import random
from typing import List, Dict, Tuple, Optional
import math
import json
import os

class UnifiedProblemGenerator:
    def __init__(self):
        self.pdf = canvas.Canvas('./db/Nhung_bai_toan_co_dien.pdf', pagesize=A4)
        try:
            pdfmetrics.registerFont(TTFont('times', r'C:\Windows\Fonts\times.ttf'))
            pdfmetrics.registerFont(TTFont('timesbd', r'C:\Windows\Fonts\timesbd.ttf'))
            pdfmetrics.registerFont(TTFont('timesi', r'C:\Windows\Fonts\timesi.ttf'))
        except:
            # Fallback to default fonts if Times New Roman is not available
            print("Không tìm thấy font Times New Roman, sử dụng font mặc định")
            pdfmetrics.registerFont(TTFont('times', 'Helvetica'))
            pdfmetrics.registerFont(TTFont('timesbd', 'Helvetica-Bold'))
            pdfmetrics.registerFont(TTFont('timesi', 'Helvetica-Oblique'))
        
        self.width, self.height = A4
        self.current_section = 1
        self.current_problem = 1
        self.y = self.height - 50
        
        # Kết hợp tất cả các loại bài toán với tỷ lệ mới để tổng là 2345 bài
        self.problem_types = {
            # Từ generate_problems.py (1111 bài ~ 50%)
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
                "weight": 0.125  # 293 bài
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
                "weight": 0.125  # 293 bài
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
                "weight": 0.085  # 199 bài
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
                "weight": 0.17  # 398 bài
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
                "weight": 0.125  # 293 bài
            },
            
            # Từ generate_classical_problems.py (1111 bài ~ 50%)
            "Bài toán chuyển động": {
                "templates": [
                    {
                        "template": "Hai xe xuất phát cùng lúc từ A và B cách nhau {distance} km. Xe thứ nhất đi với vận tốc {speed1} km/h, xe thứ hai đi với vận tốc {speed2} km/h. Hỏi sau bao lâu hai xe gặp nhau?",
                        "solution": "Gọi thời gian gặp nhau là t giờ. Khoảng cách hai xe đi được bằng khoảng cách AB: {speed1}t + {speed2}t = {distance}. Giải ra: t = {distance}/({speed1}+{speed2}) = {result} giờ."
                    },
                    {
                        "template": "Một người đi xe đạp từ A đến B với vận tốc {speed1} km/h. Khi về, do gió ngược nên vận tốc chỉ còn {speed2} km/h. Biết quãng đường AB dài {distance} km. Tính thời gian người đó đi hết cả quãng đường đi và về.",
                        "solution": "Thời gian đi: t1 = {distance}/{speed1} = {time1} giờ. Thời gian về: t2 = {distance}/{speed2} = {time2} giờ. Tổng thời gian: {result} giờ."
                    },
                    {
                        "template": "Một tàu hỏa dài {train_length}m chạy với vận tốc {speed} km/h. Tàu đi qua một cây cầu dài {bridge_length}m hết {time} giây. Tính vận tốc thực của tàu theo đơn vị m/s.",
                        "solution": "Quãng đường tàu đi = chiều dài cầu + chiều dài tàu = {total_length}m. Vận tốc = quãng đường/thời gian = {total_length}/{time} = {result} m/s."
                    }
                ],
                "weight": 0.17  # 398 bài
            },
            "Bài toán về tuổi": {
                "templates": [
                    {
                        "template": "Hiện nay tuổi cha là {age_father} tuổi, tuổi con là {age_child} tuổi. Hỏi sau bao nhiêu năm nữa thì tuổi cha gấp {times} lần tuổi con?",
                        "solution": "Gọi số năm cần tìm là x. Ta có: ({age_father}+x)/({age_child}+x)={times}. Giải ra: x = {result} năm."
                    },
                    {
                        "template": "Hiện nay tuổi mẹ hơn tuổi con là {age_diff} tuổi. Cách đây {years_ago} năm, tuổi mẹ gấp {times} lần tuổi con. Tính tuổi hiện nay của hai mẹ con.",
                        "solution": "Gọi tuổi con hiện nay là x. Tuổi mẹ hiện nay là x + {age_diff}. Theo đề bài: (x + {age_diff} - {years_ago})/(x - {years_ago}) = {times}. Giải ra: x = {child_age}, tuổi mẹ = {mother_age}."
                    }
                ],
                "weight": 0.085  # 199 bài
            },
            "Bài toán chia kẹo": {
                "templates": [
                    {
                        "template": "Có {total} viên kẹo chia cho {people} người. Người thứ hai được gấp {ratio} lần người thứ nhất, người thứ ba được gấp {ratio} lần người thứ hai. Hỏi mỗi người được bao nhiêu viên kẹo?",
                        "solution": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + {ratio}x + {ratio}^2x = {total}. Giải ra: x = {result} viên."
                    }
                ],
                "weight": 0.04  # 94 bài
            },
            "Bài toán hồ bơi": {
                "templates": [
                    {
                        "template": "Một hồ bơi có {volume} m³ nước. Có {pipes} ống nước chảy vào với lưu lượng {flow_in} m³/giờ và {drain} ống thoát với lưu lượng {flow_out} m³/giờ. Hỏi sau bao lâu hồ sẽ đầy?",
                        "solution": "Lưu lượng nước vào mỗi giờ: {pipes}×{flow_in}={total_in} m³/giờ. Lưu lượng nước ra mỗi giờ: {drain}×{flow_out}={total_out} m³/giờ. Lưu lượng thực tế: {total_in}-{total_out}={net_flow} m³/giờ. Thời gian cần: {volume}/{net_flow}={result} giờ."
                    }
                ],
                "weight": 0.085  # 199 bài
            },
            "Bài toán phân số": {
                "templates": [
                    {
                        "template": "Cho hai phân số {frac1} và {frac2}. Tìm tổng và tích của hai phân số này.",
                        "solution": "Quy đồng mẫu số: {process}. Tổng = {sum}, Tích = {product}."
                    }
                ],
                "weight": 0.085  # 199 bài
            },
            "Bài toán công việc": {
                "templates": [
                    {
                        "template": "Thợ A làm một công việc hết {time_a} giờ, thợ B làm hết {time_b} giờ. Hỏi nếu hai thợ cùng làm thì sau bao lâu sẽ hoàn thành công việc?",
                        "solution": "Trong 1 giờ, thợ A làm được 1/{time_a} công việc, thợ B làm được 1/{time_b} công việc. Hai thợ cùng làm trong 1 giờ được: 1/{time_a} + 1/{time_b} công việc. Thời gian cần: {result} giờ."
                    },
                    {
                        "template": "Ba thợ cùng làm một công việc trong {total_time} giờ thì xong. Nếu chỉ có thợ A và B làm thì cần thêm {extra_time} giờ nữa mới xong. Biết năng suất thợ B gấp đôi thợ A. Hỏi nếu chỉ có thợ C làm thì bao lâu sẽ xong công việc?",
                        "solution": "Gọi năng suất thợ A là x đơn vị/giờ. Theo đề bài, năng suất thợ B là 2x đơn vị/giờ. Gọi năng suất thợ C là y đơn vị/giờ. Ta có hệ phương trình: x + 2x + y = 1/{total_time}, x + 2x = 1/({total_time}+{extra_time}). Giải ra được y = {result_y} đơn vị/giờ. Vậy thợ C làm một mình hết {result} giờ."
                    }
                ],
                "weight": 0.085  # 199 bài
            },
            "Bài toán hỗn hợp": {
                "templates": [
                    {
                        "template": "Hòa tan {mass1}g muối vào {volume1}ml nước được dung dịch {percent1}%. Hỏi phải thêm bao nhiêu gam muối nữa để được dung dịch {percent2}%?",
                        "solution": "Khối lượng muối ban đầu: {mass1}g. Khối lượng dung dịch: {mass1} + {volume1} = {total_mass}g. Gọi số gam muối cần thêm là x. Ta có: ({mass1} + x)/({total_mass} + x) = {percent2}/100. Giải ra: x = {result}g."
                    }
                ],
                "weight": 0.085  # 199 bài
            },
            "Bài toán số học": {
                "templates": [
                    {
                        "template": "Tìm số tự nhiên có {digits} chữ số, biết rằng tổng các chữ số là {sum_digits} và tích các chữ số là {product_digits}.",
                        "solution": "Ta thử các số có {digits} chữ số thỏa mãn điều kiện. Số cần tìm là: {result}."
                    }
                ],
                "weight": 0.04  # 94 bài
            },
            "Bài toán hình học": {
                "templates": [
                    {
                        "template": "Cho hình chữ nhật có chiều dài {length}cm, chiều rộng {width}cm. Tính diện tích và chu vi của hình chữ nhật.",
                        "solution": "Diện tích = dài × rộng = {length} × {width} = {area}cm². Chu vi = 2 × (dài + rộng) = 2 × ({length} + {width}) = {perimeter}cm."
                    },
                    {
                        "template": "Cho tam giác có ba cạnh lần lượt là {a}cm, {b}cm và {c}cm. Tính diện tích tam giác.",
                        "solution": "Nửa chu vi p = ({a} + {b} + {c})/2 = {p}cm. Theo công thức Heron, diện tích S = √(p(p-a)(p-b)(p-c)) = {result}cm²."
                    },
                    {
                        "template": "Cho hình tròn có bán kính {radius}cm. Tính chu vi và diện tích của hình tròn. Lấy π = 3.14.",
                        "solution": "Chu vi = 2 × π × r = 2 × 3.14 × {radius} = {circumference}cm. Diện tích = π × r² = 3.14 × {radius}² = {area}cm²."
                    },
                    {
                        "template": "Cho hình vuông có cạnh {side}cm. Tính chu vi, diện tích và độ dài đường chéo của hình vuông.",
                        "solution": "Chu vi = 4 × {side} = {perimeter}cm. Diện tích = {side}² = {area}cm². Đường chéo = {side} × √2 = {diagonal}cm."
                    }
                ],
                "weight": 0.085  # 199 bài thay vì 56 bài
            },
            "Bài toán tỷ lệ": {
                "templates": [
                    {
                        "template": "Một lớp học có {total_students} học sinh. Số học sinh nam chiếm {male_percent}% tổng số học sinh. Hỏi lớp có bao nhiêu học sinh nữ?",
                        "solution": "Số học sinh nam = {total_students} × {male_percent}/100 = {male_count} học sinh. Số học sinh nữ = {total_students} - {male_count} = {result} học sinh."
                    }
                ],
                "weight": 0.04  # 94 bài
            }
        }

    def add_header(self):
        self.pdf.setFont('timesbd', 24)
        self.pdf.drawCentredString(self.width/2, self.height - 50, 'NHỮNG BÀI TOÁN CỔ ĐIỂN')
        self.pdf.setFont('times', 14)
        self.pdf.drawCentredString(self.width/2, self.height - 80, 'Tuyển tập 2345 bài toán đa dạng')
        self.y = self.height - 120

    def generate_problem(self, problem_type: str) -> Optional[Dict]:
        try:
            templates = self.problem_types[problem_type]["templates"]
            template_data = random.choice(templates)
            
            if problem_type == "Câu đố logic":
                total_animals = random.randint(10, 30)
                chickens = random.randint(total_animals // 3, 2 * total_animals // 3)
                rabbits = total_animals - chickens
                total_legs = 2 * chickens + 4 * rabbits
                
                return {
                    "question": template_data["template"].format(
                        total_animals=total_animals,
                        total_legs=total_legs
                    ),
                    "solution": template_data["solution"].format(
                        total_animals=total_animals,
                        chickens=chickens,
                        rabbits=rabbits,
                        total_legs=total_legs
                    )
                }
                
            elif problem_type == "Câu hỏi kiểu bài luận":
                divisor = random.choice([3, 9])
                remainder = random.randint(1, divisor-1)
                explanation = "vì các số dư khi chia 10 cho {} tạo thành một chu kỳ".format(divisor)
                
                return {
                    "question": template_data["template"].format(
                        divisor=divisor
                    ),
                    "solution": template_data["solution"].format(
                        divisor=divisor,
                        remainder=remainder,
                        explanation=explanation
                    )
                }
                
            elif problem_type == "Thơ toán học":
                total = random.randint(20, 100)
                people = random.randint(2, 10)
                quotient = total // people
                remainder = total % people
                
                return {
                    "question": template_data["template"].format(
                        total=total,
                        people=people
                    ),
                    "solution": template_data["solution"].format(
                        total=total,
                        people=people,
                        quotient=quotient,
                        remainder=remainder
                    )
                }
                
            elif problem_type == "Bài toán từ vựng toán học":
                items = ["bút", "vở", "sách", "thước"]
                units = ["cái", "quyển", "cuốn"]
                item = random.choice(items)
                unit = random.choice(units)
                price = random.randint(5000, 10000)
                quantity = random.randint(10, 50)
                discount = random.randint(5, 30)
                
                original_cost = price * quantity
                discount_amount = int(original_cost * discount / 100)
                final_cost = original_cost - discount_amount
                
                return {
                    "question": template_data["template"].format(
                        item=item,
                        price=price,
                        unit=unit,
                        quantity=quantity,
                        discount=discount
                    ),
                    "solution": template_data["solution"].format(
                        price=price,
                        quantity=quantity,
                        original_cost=original_cost,
                        discount=discount,
                        discount_amount=discount_amount,
                        final_cost=final_cost
                    )
                }
                
            elif problem_type == "Câu hỏi trắc nghiệm":
                num1 = random.randint(10, 50)
                num2 = random.randint(2, 10)
                num3 = random.randint(2, 10)
                mult_result = num2 * num3
                correct = num1 + mult_result
                wrong1 = correct + random.randint(1, 5)
                wrong2 = correct - random.randint(1, 5)
                wrong3 = num1 * num2 * num3
                
                return {
                    "question": template_data["template"].format(
                        num1=num1,
                        num2=num2,
                        num3=num3,
                        correct=correct,
                        wrong1=wrong1,
                        wrong2=wrong2,
                        wrong3=wrong3
                    ),
                    "solution": template_data["solution"].format(
                        num1=num1,
                        num2=num2,
                        num3=num3,
                        mult_result=mult_result,
                        correct=correct
                    )
                }
                
            elif problem_type == "Bài toán chuyển động":
                distance = random.randint(100, 500)
                speed1 = random.randint(40, 80)
                speed2 = random.randint(40, 80)
                time = round(distance / (speed1 + speed2), 2)
                
                return {
                    "question": template_data["template"].format(
                        distance=distance,
                        speed1=speed1,
                        speed2=speed2
                    ),
                    "solution": template_data["solution"].format(
                        distance=distance,
                        speed1=speed1,
                        speed2=speed2,
                        result=time
                    )
                }
                
            elif problem_type == "Bài toán về tuổi":
                age_father = random.randint(30, 50)
                age_child = random.randint(5, 15)
                times = random.randint(2, 4)
                years = (age_father - times * age_child) // (times - 1)
                
                return {
                    "question": template_data["template"].format(
                        age_father=age_father,
                        age_child=age_child,
                        times=times
                    ),
                    "solution": template_data["solution"].format(
                        age_father=age_father,
                        age_child=age_child,
                        times=times,
                        result=years
                    )
                }
                
            elif problem_type == "Bài toán chia kẹo":
                total = random.randint(100, 300)
                people = random.randint(3, 5)
                ratio = random.randint(2, 3)
                first_person = total // (1 + ratio + ratio**2)
                
                return {
                    "question": template_data["template"].format(
                        total=total,
                        people=people,
                        ratio=ratio
                    ),
                    "solution": template_data["solution"].format(
                        total=total,
                        ratio=ratio,
                        result=first_person
                    )
                }
                
            elif problem_type == "Bài toán hồ bơi":
                volume = random.randint(1000, 5000)
                pipes = random.randint(2, 5)
                flow_in = random.randint(50, 100)
                drain = random.randint(1, 2)
                flow_out = random.randint(20, 40)
                
                total_in = pipes * flow_in
                total_out = drain * flow_out
                net_flow = total_in - total_out
                time = round(volume / net_flow, 2)
                
                return {
                    "question": template_data["template"].format(
                        volume=volume,
                        pipes=pipes,
                        flow_in=flow_in,
                        drain=drain,
                        flow_out=flow_out
                    ),
                    "solution": template_data["solution"].format(
                        pipes=pipes,
                        flow_in=flow_in,
                        total_in=total_in,
                        drain=drain,
                        flow_out=flow_out,
                        total_out=total_out,
                        net_flow=net_flow,
                        volume=volume,
                        result=time
                    )
                }
                
            elif problem_type == "Bài toán phân số":
                def generate_fraction():
                    num = random.randint(1, 10)
                    den = random.randint(num + 1, 15)
                    return f"{num}/{den}"
                
                frac1 = generate_fraction()
                frac2 = generate_fraction()
                
                return {
                    "question": template_data["template"].format(
                        frac1=frac1,
                        frac2=frac2
                    ),
                    "solution": template_data["solution"].format(
                        process=f"Quy đồng mẫu số cho {frac1} và {frac2}",
                        sum="tổng hai phân số sau khi quy đồng",
                        product="tích hai phân số"
                    )
                }
                
            elif problem_type == "Bài toán công việc":
                time_a = random.randint(4, 12)
                time_b = random.randint(4, 12)
                total_time = round((time_a * time_b) / (time_a + time_b), 2)
                
                return {
                    "question": template_data["template"].format(
                        time_a=time_a,
                        time_b=time_b
                    ),
                    "solution": template_data["solution"].format(
                        time_a=time_a,
                        time_b=time_b,
                        result=total_time
                    )
                }
                
            elif problem_type == "Bài toán hỗn hợp":
                mass1 = random.randint(100, 500)
                volume1 = random.randint(500, 2000)
                percent1 = round(mass1 / (mass1 + volume1) * 100, 2)
                percent2 = round(percent1 + random.randint(5, 15), 2)
                
                return {
                    "question": template_data["template"].format(
                        mass1=mass1,
                        volume1=volume1,
                        percent1=percent1,
                        percent2=percent2
                    ),
                    "solution": template_data["solution"].format(
                        mass1=mass1,
                        volume1=volume1,
                        total_mass=mass1 + volume1,
                        percent2=percent2,
                        result=round((percent2 * (mass1 + volume1) / 100 - mass1) / (1 - percent2 / 100), 2)
                    )
                }
                
            elif problem_type == "Bài toán số học":
                digits = random.randint(2, 3)
                sum_digits = random.randint(5, 20)
                product_digits = random.randint(10, 100)
                result = self._find_number_with_sum_product(digits, sum_digits, product_digits)
                
                return {
                    "question": template_data["template"].format(
                        digits=digits,
                        sum_digits=sum_digits,
                        product_digits=product_digits
                    ),
                    "solution": template_data["solution"].format(
                        digits=digits,
                        result=result if result else "Không có số thỏa mãn"
                    )
                }
                
            elif problem_type == "Bài toán hình học":
                template_text = template_data["template"]
                
                if "hình chữ nhật" in template_text:
                    length = random.randint(5, 20)
                    width = random.randint(3, length)
                    area = length * width
                    perimeter = 2 * (length + width)
                    
                    return {
                        "question": template_data["template"].format(
                            length=length,
                            width=width
                        ),
                        "solution": template_data["solution"].format(
                            length=length,
                            width=width,
                            area=area,
                            perimeter=perimeter
                        )
                    }
                elif "tam giác" in template_text:
                    # Đảm bảo tam giác hợp lệ (tổng hai cạnh bất kỳ lớn hơn cạnh còn lại)
                    a = random.randint(5, 15)
                    b = random.randint(5, 15)
                    c = random.randint(max(a, b) - min(a, b) + 1, a + b - 1)
                    
                    p = (a + b + c) / 2
                    area = round(math.sqrt(p * (p - a) * (p - b) * (p - c)), 2)
                    
                    return {
                        "question": template_data["template"].format(
                            a=a, b=b, c=c
                        ),
                        "solution": template_data["solution"].format(
                            a=a, b=b, c=c, p=p, result=area
                        )
                    }
                elif "hình tròn" in template_text:
                    radius = random.randint(3, 15)
                    circumference = round(2 * 3.14 * radius, 2)
                    area = round(3.14 * radius * radius, 2)
                    
                    return {
                        "question": template_data["template"].format(
                            radius=radius
                        ),
                        "solution": template_data["solution"].format(
                            radius=radius,
                            circumference=circumference,
                            area=area
                        )
                    }
                elif "hình vuông" in template_text:
                    side = random.randint(5, 20)
                    perimeter = 4 * side
                    area = side * side
                    diagonal = round(side * math.sqrt(2), 2)
                    
                    return {
                        "question": template_data["template"].format(
                            side=side
                        ),
                        "solution": template_data["solution"].format(
                            side=side,
                            perimeter=perimeter,
                            area=area,
                            diagonal=diagonal
                        )
                    }
                
            elif problem_type == "Bài toán tỷ lệ":
                total_students = random.randint(30, 50)
                male_percent = random.randint(40, 60)
                male_count = round(total_students * male_percent / 100)
                female_count = total_students - male_count
                
                return {
                    "question": template_data["template"].format(
                        total_students=total_students,
                        male_percent=male_percent
                    ),
                    "solution": template_data["solution"].format(
                        total_students=total_students,
                        male_percent=male_percent,
                        male_count=male_count,
                        result=female_count
                    )
                }
                
            return None
            
        except Exception as e:
            print(f"Lỗi khi tạo bài toán {problem_type}: {str(e)}")
            return None

    def _solve_josephus(self, n: int, k: int) -> List[int]:
        """Giải bài toán Josephus và trả về các bước giải."""
        steps = []
        pos = 1
        for i in range(1, n + 1):
            pos = ((pos + k - 1) % i) + 1
            steps.append(pos)
        return steps

    def generate_problems(self, total_count: int = 2345) -> List[Dict]:
        problems = []
        problem_types = list(self.problem_types.keys())
        
        # Tính số lượng bài toán cho mỗi loại
        type_counts = {}
        remaining_count = total_count
        
        print("\n=== Phân bổ số lượng bài toán ===")
        for ptype in problem_types[:-1]:
            count = round(total_count * self.problem_types[ptype]["weight"])
            type_counts[ptype] = count
            remaining_count -= count
            print(f"- {ptype}: {count} bài ({self.problem_types[ptype]['weight']*100:.1f}%)")
        
        type_counts[problem_types[-1]] = remaining_count
        print(f"- {problem_types[-1]}: {remaining_count} bài ({remaining_count/total_count*100:.1f}%)")
        
        # Theo dõi số lượng templates đã sử dụng cho mỗi loại
        template_usage = {ptype: {} for ptype in problem_types}
        
        print("\n=== Tiến trình tạo bài toán ===")
        # Tạo bài toán theo số lượng đã tính
        for ptype, count in type_counts.items():
            print(f"\nĐang tạo {count} bài toán loại '{ptype}'...")
            success_count = 0
            
            for i in range(count):
                problem = self.generate_problem(ptype)
                if problem:
                    # Theo dõi template được sử dụng
                    template = problem["question"][:50] + "..."  # Lấy 50 ký tự đầu
                    template_usage[ptype][template] = template_usage[ptype].get(template, 0) + 1
                    
                    problems.append({
                        "id": len(problems) + 1,
                        "type": ptype,
                        "question": problem["question"],
                        "solution": problem["solution"],
                        "difficulty": self._estimate_difficulty(problem["question"], problem["solution"]),
                        "tags": self._extract_tags(problem["question"])
                    })
                    success_count += 1
                
                # In tiến trình mỗi 10%
                if (i + 1) % max(1, count // 10) == 0:
                    print(f"  ✓ Đã tạo {i + 1}/{count} bài ({(i + 1)/count*100:.1f}%)")
            
            print(f"  ✓ Hoàn thành: {success_count}/{count} bài")
        
        print("\n=== Thống kê sử dụng templates ===")
        for ptype in problem_types:
            print(f"\n{ptype}:")
            for template, count in template_usage[ptype].items():
                print(f"- Template '{template}': {count} lần")
        
        # Thống kê độ khó
        difficulty_stats = {"Dễ": 0, "Trung bình": 0, "Khó": 0}
        for problem in problems:
            difficulty_stats[problem["difficulty"]] += 1
        
        print("\n=== Thống kê độ khó ===")
        for diff, count in difficulty_stats.items():
            print(f"- {diff}: {count} bài ({count/len(problems)*100:.1f}%)")
        
        # Thống kê tags
        tag_stats = {}
        for problem in problems:
            for tag in problem["tags"]:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
        
        print("\n=== Thống kê tags ===")
        for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"- {tag}: {count} bài ({count/len(problems)*100:.1f}%)")
        
        random.shuffle(problems)
        return problems

    def save_problems(self, problems: List[Dict]):
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs('./db/questions', exist_ok=True)
            print(f"✓ Đã tạo thư mục ./db/questions")
            
            filepath = './db/questions/problems.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_problems": len(problems),
                    "questions": problems
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ Đã tạo và lưu {len(problems)} bài toán vào {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu file: {str(e)}")

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

    def _find_number_with_sum_product(self, digits: int, sum_digits: int, product_digits: int) -> int:
        def get_digits(n: int) -> list:
            return [int(d) for d in str(n)]
        
        start = 10 ** (digits - 1)
        end = 10 ** digits - 1
        
        for num in range(start, end + 1):
            num_digits = get_digits(num)
            if sum(num_digits) == sum_digits and \
               eval('*'.join(map(str, num_digits))) == product_digits:
                return num
        return None

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

    def generate_all_problems(self):
        self.add_header()
        
        # Phân bổ số lượng bài toán cho mỗi loại
        problems = self.generate_problems(2345)
        
        # Thêm các bài toán vào PDF
        problem_types = list(self.problem_types.keys())
        for problem_type in problem_types:
            self.add_section(problem_type)
            type_problems = [p for p in problems if p["type"] == problem_type]
            for problem in type_problems[:50]:  # Giới hạn 50 bài mỗi loại để PDF không quá lớn
                self.add_problem(problem["question"], problem["solution"])
        
        # Lưu file PDF
        self.pdf.save()
        print(f"✓ Đã tạo file PDF: ./db/Nhung_bai_toan_co_dien.pdf")
        
        # Lưu file JSON
        self.save_problems(problems)

if __name__ == '__main__':
    try:
        generator = UnifiedProblemGenerator()
        print("✓ Đã khởi tạo generator")
        generator.generate_all_problems()
    except Exception as e:
        print(f"Lỗi: {str(e)}") 