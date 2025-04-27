import random
import math
import json
import os
from typing import List, Dict, Tuple, Optional

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class UnifiedProblemGenerator:
    def __init__(self):
        # Tạo thư mục db nếu chưa có
        if not os.path.exists('./db'):
            os.makedirs('./db')
        if not os.path.exists('./db/questions'):
            os.makedirs('./db/questions')

        self.pdf = canvas.Canvas('./db/Nhung_bai_toan_co_dien.pdf', pagesize=A4)
        try:
            # Ưu tiên sử dụng font Times New Roman nếu có
            times_font_path = r'C:\Windows\Fonts\times.ttf'
            timesbd_font_path = r'C:\Windows\Fonts\timesbd.ttf'
            timesi_font_path = r'C:\Windows\Fonts\timesi.ttf'
            if os.path.exists(times_font_path) and os.path.exists(timesbd_font_path) and os.path.exists(timesi_font_path):
                 pdfmetrics.registerFont(TTFont('times', times_font_path))
                 pdfmetrics.registerFont(TTFont('timesbd', timesbd_font_path))
                 pdfmetrics.registerFont(TTFont('timesi', timesi_font_path))
                 print("Đã đăng ký font Times New Roman.")
            else:
                 raise FileNotFoundError # Nếu thiếu 1 trong 3 file, fallback
        except (FileNotFoundError, Exception):
             # Fallback to default fonts if Times New Roman is not available or error occurs
             print("Không tìm thấy đầy đủ bộ font Times New Roman hoặc có lỗi, sử dụng font Helvetica mặc định.")
             pdfmetrics.registerFont(TTFont('times', 'Helvetica'))
             pdfmetrics.registerFont(TTFont('timesbd', 'Helvetica-Bold'))
             pdfmetrics.registerFont(TTFont('timesi', 'Helvetica-Oblique'))

        self.width, self.height = A4
        self.current_section = 1
        self.current_problem = 1
        self.y = self.height - 50

        # --- Cập nhật Templates và Solutions ---
        self.problem_types = {
            # Từ generate_problems.py (1111 bài ~ 50%)
            "Câu đố logic": {
                "templates": [
                    {
                        "template": "Một người nông dân có {total_animals} con vật gồm gà và thỏ. Đếm tổng số chân thì thấy có {total_legs} cái chân. Hỏi người nông dân có bao nhiêu con gà và bao nhiêu con thỏ?",
                        "solution": "Gọi số gà là x con. Khi đó số thỏ là {total_animals} - x con.\nSố chân gà: 2x. Số chân thỏ: 4({total_animals} - x).\nTheo đề bài, ta có phương trình: 2x + 4({total_animals} - x) = {total_legs}\n<=> 2x + {4_times_total} - 4x = {total_legs}\n<=> {4_times_total} - {total_legs} = 2x\n<=> {diff} = 2x\n<=> x = {chickens}.\nVậy số gà là {chickens} con. Số thỏ là {total_animals} - {chickens} = {rabbits} con."
                    },
                    # Thêm các template logic khác nếu cần
                ],
                "weight": 0.125
            },
            "Câu hỏi kiểu bài luận": {
                "templates": [
                    {   # Sửa lỗi logic giải thích chia hết cho 3, 9
                        "template": "Giải thích tại sao một số tự nhiên lại chia hết cho {divisor} khi và chỉ khi tổng các chữ số của nó chia hết cho {divisor}?",
                        "solution": "Để chứng minh điều này, ta xét một số tự nhiên N có n chữ số: N = a₁a₂...aₙ.\nTa có thể viết N dưới dạng tổng: N = a₁×10ⁿ⁻¹ + a₂×10ⁿ⁻² + ... + aₙ₋₁×10¹ + aₙ×10⁰.\nTa biết rằng 10 chia cho {divisor} dư 1 (vì 10 = {quotient} × {divisor} + 1).\nDo đó, 10 ≡ 1 (mod {divisor}).\nSuy ra, 10ᵏ ≡ 1ᵏ ≡ 1 (mod {divisor}) với mọi số mũ k ≥ 0.\nBây giờ, xét N theo modulo {divisor}:\nN ≡ a₁×(10ⁿ⁻¹ mod {divisor}) + ... + aₙ×(10⁰ mod {divisor}) (mod {divisor})\nN ≡ a₁×(1) + a₂×(1) + ... + aₙ×(1) (mod {divisor})\nN ≡ a₁ + a₂ + ... + aₙ (mod {divisor})\nĐiều này chứng tỏ rằng số N và tổng các chữ số của nó (S = a₁ + a₂ + ... + aₙ) có cùng số dư khi chia cho {divisor}.\nVì vậy, N chia hết cho {divisor} (tức là N ≡ 0 mod {divisor}) khi và chỉ khi tổng các chữ số của nó chia hết cho {divisor} (tức là S ≡ 0 mod {divisor})."
                    },
                    {
                        "template": "Chứng minh rằng trong tam giác vuông, sin của một góc nhọn {angle}° (ví dụ 30°, 45°, 60°) luôn bằng một hằng số {ratio}.",
                        "solution": "Xét tam giác ABC vuông tại A, có góc B = {angle}°. Sin của góc B được định nghĩa là tỷ số giữa cạnh đối (AC) và cạnh huyền (BC): sin(B) = AC/BC.\nTheo định nghĩa lượng giác trong tam giác vuông, sin({angle}°) = {ratio}.\nĐể chứng minh điều này luôn đúng, xét một tam giác A'B'C' vuông tại A' cũng có góc B' = {angle}°. Khi đó, tam giác ABC đồng dạng với tam giác A'B'C' (theo trường hợp góc-góc, vì có góc vuông và góc {angle}° bằng nhau).\nDo đồng dạng, tỷ số các cạnh tương ứng bằng nhau: AC/A'C' = BC/B'C' = AB/A'B'.\nTừ đó suy ra AC/BC = A'C'/B'C'.\nVậy, sin(B) = sin(B'), tức là sin({angle}°) luôn có giá trị không đổi là {ratio} đối với mọi tam giác vuông có một góc nhọn bằng {angle}°."
                    },
                    # Thêm template bài luận khác nếu cần
                ],
                "weight": 0.125
            },
             "Thơ toán học": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Có {total} quả cam ngon,\nChia đều cho {people} người em.\nMỗi người được mấy quả?\nCòn dư mấy quả cam?",
                        "solution": "Số cam chia được: {total} ÷ {people} = {quotient} quả mỗi người.\nSố cam còn dư: {total} mod {people} = {remainder} quả."
                    },
                    {
                        "template": "Hình vuông có cạnh {side} phân,\nTính xem chu vi với diện tích là bao nhiêu?\nBốn cạnh bằng nhau thật đều,\nDiện tích bằng mấy, hãy điền vào ngay!",
                        "solution": "Chu vi hình vuông: 4 × {side} = {perimeter} phân.\nDiện tích hình vuông: {side} × {side} = {area} phân vuông."
                    }
                ],
                "weight": 0.085
            },
            "Bài toán từ vựng toán học": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Một cửa hàng bán {item} với giá {price}đ một {unit}. Nếu mua {quantity} {unit}, sau đó được giảm {discount}%, hỏi phải trả bao nhiêu tiền?",
                        "solution": "Giá gốc khi chưa giảm: {quantity} × {price} = {original_cost}đ.\nSố tiền được giảm: {original_cost} × {discount}/100 = {discount_amount}đ.\nSố tiền phải trả sau khi giảm giá: {original_cost} - {discount_amount} = {final_cost}đ."
                    },
                     {
                        "template": "Một bể chứa nước hình hộp chữ nhật có chiều dài {length}m, chiều rộng {width}m và chiều cao {height}m. Hỏi bể chứa được bao nhiêu lít nước (biết 1m³ = 1000 lít)?",
                        "solution": "Thể tích của bể hình hộp chữ nhật được tính bằng công thức: V = dài × rộng × cao.\nV = {length}m × {width}m × {height}m = {volume_m3} m³.\nĐổi sang lít: {volume_m3} m³ × 1000 lít/m³ = {volume_liters} lít.\nVậy bể chứa được {volume_liters} lít nước."
                    },
                ],
                "weight": 0.17
            },
             "Câu hỏi trắc nghiệm": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Kết quả của phép tính {num1} + {num2} × {num3} là bao nhiêu?\nA. {wrong1}\nB. {correct}\nC. {wrong2}\nD. {wrong3}",
                        "solution": "Theo quy tắc ưu tiên phép tính (nhân chia trước, cộng trừ sau), ta thực hiện phép nhân trước:\n{num2} × {num3} = {mult_result}.\nSau đó thực hiện phép cộng:\n{num1} + {mult_result} = {correct}.\nVậy đáp án đúng là B. {correct}"
                    },
                    {
                        "template": "Một hình vuông có chu vi {perimeter}cm. Diện tích của hình vuông này là bao nhiêu?\nA. {wrong1}cm²\nB. {wrong2}cm²\nC. {correct}cm²\nD. {wrong3}cm²",
                        "solution": "Chu vi hình vuông được tính bằng 4 lần độ dài cạnh (P = 4a).\nTa có: 4 × cạnh = {perimeter} cm.\nSuy ra độ dài cạnh là: cạnh = {perimeter} / 4 = {side} cm.\nDiện tích hình vuông được tính bằng cạnh nhân cạnh (S = a²).\nDiện tích = {side} cm × {side} cm = {correct} cm².\nVậy đáp án đúng là C. {correct}cm²"
                    },
                ],
                "weight": 0.125
            },
            "Bài toán chuyển động": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Hai xe xuất phát cùng lúc từ hai địa điểm A và B cách nhau {distance} km và đi ngược chiều nhau. Xe thứ nhất đi từ A với vận tốc {speed1} km/h, xe thứ hai đi từ B với vận tốc {speed2} km/h. Hỏi sau bao lâu hai xe gặp nhau?",
                        "solution": "Gọi thời gian hai xe đi đến lúc gặp nhau là t (giờ).\nQuãng đường xe thứ nhất đi được là: S₁ = {speed1} × t (km).\nQuãng đường xe thứ hai đi được là: S₂ = {speed2} × t (km).\nKhi hai xe gặp nhau, tổng quãng đường hai xe đi được bằng khoảng cách AB:\nS₁ + S₂ = {distance}\n{speed1}t + {speed2}t = {distance}\n({speed1} + {speed2})t = {distance}\nt = {distance} / ({speed1} + {speed2})\nt = {result} giờ.\nVậy sau {result} giờ hai xe gặp nhau."
                    },
                     {
                        "template": "Một ca nô đi xuôi dòng từ A đến B hết {time_down} giờ và đi ngược dòng từ B về A hết {time_up} giờ. Biết vận tốc dòng nước là {water_speed} km/h. Tính vận tốc thực của ca nô và quãng đường AB.",
                        "solution": "Gọi vận tốc thực của ca nô là v (km/h, v > {water_speed}).\nVận tốc xuôi dòng: v_xuoi = v + {water_speed} (km/h).\nVận tốc ngược dòng: v_nguoc = v - {water_speed} (km/h).\nQuãng đường AB không đổi:\nAB = v_xuoi × t_xuoi = (v + {water_speed}) × {time_down}\nAB = v_nguoc × t_nguoc = (v - {water_speed}) × {time_up}\nSuy ra: (v + {water_speed}) × {time_down} = (v - {water_speed}) × {time_up}\n{time_down}v + {const1} = {time_up}v - {const2}\n{diff_coeff}v = {sum_const}\nv = {canoe_speed} km/h.\nQuãng đường AB = ({canoe_speed} + {water_speed}) × {time_down} = {distance} km.\nVậy vận tốc thực của ca nô là {canoe_speed} km/h và quãng đường AB dài {distance} km."
                     }
                ],
                "weight": 0.17
            },
            "Bài toán về tuổi": {
                 # Sửa logic tạo bài toán tuổi (Cả 2 template)
                "templates": [
                    { # Template 1: Tuổi tương lai
                        "template": "Hiện nay tuổi cha là {age_father} tuổi, tuổi con là {age_child} tuổi. Hỏi sau bao nhiêu năm nữa thì tuổi cha gấp {times} lần tuổi con?",
                        "solution": "Gọi số năm cần tìm là x (năm, x > 0).\nSau x năm nữa, tuổi cha là: {age_father} + x (tuổi).\nSau x năm nữa, tuổi con là: {age_child} + x (tuổi).\nTheo đề bài, lúc đó tuổi cha gấp {times} lần tuổi con, ta có phương trình:\n{age_father} + x = {times} × ({age_child} + x)\n{age_father} + x = {times} × {age_child} + {times}x\n{age_father} - {times_age_child} = {times}x - x\n{diff_ages} = ({times} - 1)x\n{diff_ages} = {times_minus_1}x\nx = {diff_ages} / {times_minus_1}\nx = {result}\nVậy sau {result} năm nữa thì tuổi cha gấp {times} lần tuổi con."
                        # LƯU Ý: Template solution này vẫn còn vấn đề tiềm ẩn như đã thảo luận
                        # và có thể gây KeyError nếu bạn vô tình dùng {age_father + result} ở đâu đó.
                        # Bản sửa lỗi thứ hai giải quyết vấn đề này.
                    },
                    { # Template 2: Tuổi quá khứ
                        "template": "Hiện nay tuổi mẹ hơn tuổi con là {age_diff} tuổi. Cách đây {years_ago} năm, tuổi mẹ gấp {times} lần tuổi con. Tính tuổi hiện nay của hai mẹ con.",
                        "solution": "Gọi tuổi con hiện nay là x (tuổi, x > {years_ago}).\nTuổi mẹ hiện nay là: x + {age_diff} (tuổi).\nCách đây {years_ago} năm:\nTuổi con là: x - {years_ago} (tuổi).\nTuổi mẹ là: (x + {age_diff}) - {years_ago} (tuổi).\nTheo đề bài, cách đây {years_ago} năm tuổi mẹ gấp {times} lần tuổi con, ta có phương trình:\n(x + {age_diff} - {years_ago}) = {times} × (x - {years_ago})\nx + {age_diff_minus_years} = {times}x - {times_years_ago}\n{age_diff_minus_years} + {times_years_ago} = {times}x - x\n{sum_const} = ({times} - 1)x\n{sum_const} = {times_minus_1}x\nx = {sum_const} / {times_minus_1}\nx = {child_age}\nVậy tuổi con hiện nay là {child_age} tuổi.\nTuổi mẹ hiện nay là: {child_age} + {age_diff} = {mother_age} tuổi."
                    }
                ],
                "weight": 0.085
            },
            "Bài toán chia kẹo": {
                 # Sửa logic tạo bài toán chia kẹo (chỉ 3 người)
                "templates": [
                    {
                        "template": "Có tổng cộng {total} viên kẹo chia cho 3 bạn An, Bình, Cường. Biết rằng số kẹo của Bình gấp {ratio} lần số kẹo của An, số kẹo của Cường gấp {ratio} lần số kẹo của Bình. Hỏi mỗi bạn được bao nhiêu viên kẹo?",
                        "solution": "Gọi số kẹo của An là x (viên, x > 0).\nTheo đề bài, số kẹo của Bình là: {ratio}x (viên).\nSố kẹo của Cường là: {ratio} × (số kẹo của Bình) = {ratio} × ({ratio}x) = {ratio_sq}x (viên).\nTổng số kẹo của ba bạn là {total} viên, nên ta có phương trình:\nx + {ratio}x + {ratio_sq}x = {total}\n(1 + {ratio} + {ratio_sq})x = {total}\n{sum_coeffs}x = {total}\nx = {total} / {sum_coeffs}\nx = {person1_candies}\nVậy số kẹo của An là {person1_candies} viên.\nSố kẹo của Bình là: {ratio} × {person1_candies} = {person2_candies} viên.\nSố kẹo của Cường là: {ratio_sq} × {person1_candies} = {person3_candies} viên.\nĐáp số: An: {person1_candies} viên, Bình: {person2_candies} viên, Cường: {person3_candies} viên."
                    }
                ],
                "weight": 0.04
            },
            "Bài toán hồ bơi": { # Giữ nguyên, không yêu cầu sửa
                 "templates": [
                    {
                        "template": "Một bể nước dạng hình hộp chữ nhật có thể chứa tối đa {volume} m³ nước. Người ta mở {pipes_in} vòi nước chảy vào bể, mỗi vòi chảy được {flow_in} m³/giờ, đồng thời mở {pipes_out} vòi tháo nước ra, mỗi vòi chảy ra {flow_out} m³/giờ. Hỏi nếu bể đang cạn, sau bao lâu thì bể sẽ đầy?",
                        "solution": "Tổng lượng nước chảy vào bể mỗi giờ là:\nL_vào = {pipes_in} vòi × {flow_in} m³/giờ/vòi = {total_in} m³/giờ.\nTổng lượng nước chảy ra khỏi bể mỗi giờ là:\nL_ra = {pipes_out} vòi × {flow_out} m³/giờ/vòi = {total_out} m³/giờ.\nLượng nước thực tế tăng thêm trong bể mỗi giờ là:\nL_thực = L_vào - L_ra = {total_in} - {total_out} = {net_flow} m³/giờ.\n(Giả sử {net_flow} > 0 để bể có thể đầy).\nThời gian để bể đầy nước là:\nT = Dung tích bể / Lượng nước thực tế mỗi giờ\nT = {volume} m³ / {net_flow} m³/giờ = {result} giờ.\nVậy sau {result} giờ thì bể sẽ đầy."
                    }
                ],
                "weight": 0.085
            },
            "Bài toán phân số": { # Đã sửa logic gọi hàm simplify_fraction
                 "templates": [
                    {
                        "template": "Cho hai phân số {frac1} và {frac2}. Hãy tính tổng và tích của hai phân số này (kết quả để dạng phân số tối giản).",
                        "solution": "Để tính tổng, ta quy đồng mẫu số:\n{frac1} + {frac2} = {sum_step1} = {sum_step2}\nTối giản (nếu cần): {sum_simplified}\nĐể tính tích, ta nhân tử với tử, mẫu với mẫu:\n{frac1} × {frac2} = {product_step1} = {product_step2}\nTối giản (nếu cần): {product_simplified}\nVậy tổng là {sum_simplified} và tích là {product_simplified}."
                    }
                ],
                "weight": 0.085
            },
             "Bài toán công việc": { # Đã sửa logic gọi hàm simplify_fraction
                "templates": [
                    {
                        "template": "Người thợ thứ nhất làm một mình có thể hoàn thành một công việc trong {time_a} giờ. Người thợ thứ hai làm một mình có thể hoàn thành công việc đó trong {time_b} giờ. Hỏi nếu cả hai người thợ cùng làm thì sau bao lâu sẽ hoàn thành công việc?",
                        "solution": "Trong 1 giờ, người thợ thứ nhất làm được: 1/{time_a} (công việc).\nTrong 1 giờ, người thợ thứ hai làm được: 1/{time_b} (công việc).\nTrong 1 giờ, cả hai người cùng làm được: 1/{time_a} + 1/{time_b} = {sum_rates_frac} = {sum_rates_decimal} (công việc).\nThời gian để cả hai người cùng hoàn thành công việc là:\nT = 1 / (Năng suất chung mỗi giờ) = 1 / ({sum_rates_decimal}) = {result} giờ.\nVậy nếu cả hai cùng làm thì sau {result} giờ sẽ xong công việc."
                    },
                    # Thêm template công việc khác nếu cần
                ],
                "weight": 0.085
            },
             "Bài toán hỗn hợp": {
                # Sửa wording template, giữ nguyên logic tính toán
                "templates": [
                    {
                        "template": "Một dung dịch muối được tạo ra bằng cách hòa tan {mass1}g muối vào {volume1}ml nước (coi 1ml nước nặng 1g). Hỏi cần thêm bao nhiêu gam muối nữa vào dung dịch này để thu được dung dịch mới có nồng độ {percent2}%?",
                        "solution": "Khối lượng nước ban đầu: {volume1}g.\nKhối lượng muối ban đầu: {mass1}g.\nKhối lượng dung dịch ban đầu: m_dd1 = {mass1} + {volume1} = {initial_total_mass}g.\nGọi khối lượng muối cần thêm vào là x (gam, x > 0).\nKhối lượng muối trong dung dịch mới: m_muoi2 = {mass1} + x (g).\nKhối lượng dung dịch mới: m_dd2 = m_dd1 + x = {initial_total_mass} + x (g).\nNồng độ phần trăm của dung dịch mới là {percent2}%, ta có công thức:\nC% = (m_muoi2 / m_dd2) × 100%\n{percent2} = (({mass1} + x) / ({initial_total_mass} + x)) × 100\n{percent2}/100 = ({mass1} + x) / ({initial_total_mass} + x)\n{percent2_decimal} = ({mass1} + x) / ({initial_total_mass} + x)\n{percent2_decimal} × ({initial_total_mass} + x) = {mass1} + x\n{const1} + {percent2_decimal}x = {mass1} + x\n{const1} - {mass1} = x - {percent2_decimal}x\n{const2} = (1 - {percent2_decimal})x\n{const2} = {one_minus_percent2_decimal}x\nx = {const2} / {one_minus_percent2_decimal}\nx = {result}\nVậy cần thêm {result}g muối nữa."
                    }
                ],
                "weight": 0.085
            },
            "Bài toán số học": {
                 # Sửa logic tạo bài toán số học (đảm bảo có nghiệm)
                "templates": [
                    {
                        "template": "Tìm số tự nhiên có {digits} chữ số, biết rằng tổng các chữ số của nó bằng {sum_digits} và tích các chữ số của nó bằng {product_digits}.",
                        "solution": "Gọi số cần tìm là N. N có {digits} chữ số.\nTheo đề bài, tổng các chữ số của N là {sum_digits}.\nTheo đề bài, tích các chữ số của N là {product_digits}.\nTa cần tìm N thỏa mãn cả hai điều kiện trên.\nBằng cách thử hoặc phân tích các chữ số có thể có (ví dụ: các ước của {product_digits}), ta tìm được số thỏa mãn là: N = {result}.\nKiểm tra lại: Số {result} có {digits} chữ số. Tổng các chữ số: {check_sum} = {sum_digits}. Tích các chữ số: {check_product} = {product_digits}. (Thỏa mãn)"
                    }
                ],
                "weight": 0.04
            },
            "Bài toán hình học": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Cho hình chữ nhật có chiều dài {length}cm và chiều rộng {width}cm. Tính diện tích và chu vi của hình chữ nhật đó.",
                        "solution": "Diện tích hình chữ nhật được tính bằng công thức: S = chiều dài × chiều rộng.\nS = {length}cm × {width}cm = {area} cm².\nChu vi hình chữ nhật được tính bằng công thức: P = 2 × (chiều dài + chiều rộng).\nP = 2 × ({length}cm + {width}cm) = 2 × {sum_len_wid}cm = {perimeter} cm.\nVậy diện tích là {area} cm² và chu vi là {perimeter} cm."
                    },
                    { # Heron's formula needs valid triangle
                        "template": "Cho tam giác có độ dài ba cạnh lần lượt là {a}cm, {b}cm và {c}cm. Tính diện tích của tam giác này.",
                        "solution": "Để tính diện tích tam giác khi biết độ dài 3 cạnh, ta dùng công thức Heron.\nTrước hết, tính nửa chu vi (p):\np = (a + b + c) / 2 = ({a} + {b} + {c}) / 2 = {p_sum} / 2 = {p} cm.\nDiện tích S được tính bằng:\nS = √[p(p-a)(p-b)(p-c)]\nS = √[{p}({p}-{a})({p}-{b})({p}-{c})]\nS = √[{p} × {p_minus_a} × {p_minus_b} × {p_minus_c}]\nS = √[{product_inside_sqrt}]\nS ≈ {result} cm².\nVậy diện tích tam giác khoảng {result} cm²."
                    },
                     {
                        "template": "Cho hình tròn có bán kính {radius}cm. Tính chu vi và diện tích của hình tròn (Lấy π ≈ 3.14).",
                        "solution": "Chu vi (hay đường tròn) của hình tròn được tính bằng công thức: C = 2 × π × r.\nC ≈ 2 × 3.14 × {radius} cm = {circumference} cm.\nDiện tích của hình tròn được tính bằng công thức: S = π × r².\nS ≈ 3.14 × ({radius} cm)² = 3.14 × {radius_sq} cm² = {area} cm².\nVậy chu vi hình tròn khoảng {circumference} cm và diện tích khoảng {area} cm²."
                     }
                ],
                "weight": 0.085
            },
            "Bài toán tỷ lệ": { # Giữ nguyên, không yêu cầu sửa
                "templates": [
                    {
                        "template": "Một lớp học có {total_students} học sinh, trong đó số học sinh nam chiếm {male_percent}% tổng số học sinh của lớp. Hỏi lớp học đó có bao nhiêu học sinh nữ?",
                        "solution": "Số học sinh nam của lớp là:\nSố nam = Tổng số học sinh × Tỷ lệ nam (%)\nSố nam = {total_students} × {male_percent}% = {total_students} × ({male_percent}/100) = {male_count_exact} ≈ {male_count} học sinh.\n(Làm tròn đến số nguyên gần nhất nếu cần).\nSố học sinh nữ của lớp là:\nSố nữ = Tổng số học sinh - Số học sinh nam\nSố nữ = {total_students} - {male_count} = {result} học sinh.\nVậy lớp học đó có {result} học sinh nữ."
                    }
                ],
                "weight": 0.04
            }
        }

    # --- Helper methods (Đã thêm _gcd và _simplify_fraction) ---
    def _gcd(self, a, b):
        """Tính ước chung lớn nhất."""
        while b:
            a, b = b, a % b
        return abs(a)

    def _simplify_fraction(self, num, den):
        """Tối giản phân số và trả về dạng chuỗi."""
        if den == 0: return "Lỗi chia 0"
        if num == 0: return "0"
        common_divisor = self._gcd(num, den)
        num //= common_divisor
        den //= common_divisor
        return f"{num}/{den}" if den != 1 else str(num)

    def add_header(self):
        self.pdf.setFont('timesbd', 24)
        self.pdf.drawCentredString(self.width/2, self.height - 50, 'NHỮNG BÀI TOÁN CỔ ĐIỂN')
        self.pdf.setFont('times', 14)
        self.pdf.drawCentredString(self.width/2, self.height - 80, 'Tuyển tập 2345 bài toán đa dạng')
        self.y = self.height - 120

    def generate_problem(self, problem_type: str) -> Optional[Dict]:
        try:
            templates = self.problem_types[problem_type]["templates"]
            if not templates: # Xử lý trường hợp không có template cho loại bài toán
                print(f"Cảnh báo: Không tìm thấy template cho loại bài toán '{problem_type}'")
                return None

            template_data = random.choice(templates)
            params = {} # Dictionary để chứa các tham số được tạo ra

            # --- Logic tạo tham số cho từng loại bài toán ---

            if problem_type == "Câu đố logic":
                # Logic Gà và Thỏ (ví dụ)
                while True: # Đảm bảo có nghiệm nguyên dương
                    total_animals = random.randint(10, 30)
                    # Chọn số gà ngẫu nhiên nhưng hợp lý
                    min_chickens = max(1, (4 * total_animals - 200) // 2) # Giả sử max 200 chân để tránh vô lý
                    max_chickens = min(total_animals - 1, (4 * total_animals - 20) // 2) # Giả sử min 20 chân
                    if min_chickens >= max_chickens: continue # Thử lại nếu khoảng không hợp lệ
                    chickens = random.randint(min_chickens, max_chickens)
                    rabbits = total_animals - chickens
                    if rabbits > 0: # Đảm bảo có thỏ
                         total_legs = 2 * chickens + 4 * rabbits
                         if total_legs > 0 and total_legs % 2 == 0 : # Chân phải là số chẵn dương
                            params = {
                                "total_animals": total_animals,
                                "total_legs": total_legs,
                                "chickens": chickens,
                                "rabbits": rabbits,
                                "4_times_total": 4 * total_animals,
                                "diff": 4 * total_animals - total_legs
                            }
                            break # Thoát vòng lặp khi tìm được bộ hợp lệ

            elif problem_type == "Câu hỏi kiểu bài luận":
                 # Logic cho bài luận chia hết
                if "chia hết cho {divisor}" in template_data["template"]:
                    divisor = random.choice([3, 9])
                    quotient = 10 // divisor # = 3 nếu divisor=3, = 1 nếu divisor=9
                    params = {
                        "divisor": divisor,
                        "quotient": quotient
                        # Không cần remainder=1 vì đã fixed trong solution template
                    }
                # Logic cho bài luận sin(góc)
                elif "sin của một góc nhọn" in template_data["template"]:
                    angle = random.choice([30, 45, 60])
                    ratio_map = {30: "1/2", 45: "√2/2", 60: "√3/2"}
                    params = {
                        "angle": angle,
                        "ratio": ratio_map[angle]
                    }
                # Thêm logic cho các template bài luận khác nếu có

            elif problem_type == "Thơ toán học":
                 # Logic thơ chia kẹo
                 if "quả cam ngon" in template_data["template"]:
                    total = random.randint(10, 50)
                    people = random.randint(2, 7)
                    quotient = total // people
                    remainder = total % people
                    params = {"total": total, "people": people, "quotient": quotient, "remainder": remainder}
                 # Logic thơ hình vuông
                 elif "Hình vuông có cạnh" in template_data["template"]:
                    side = random.randint(3, 15)
                    perimeter = 4 * side
                    area = side * side
                    params = {"side": side, "perimeter": perimeter, "area": area}
                 # Thêm logic cho các bài thơ khác nếu có

            elif problem_type == "Bài toán từ vựng toán học":
                 # Logic giảm giá
                 if "giảm {discount}%" in template_data["template"]:
                    items = ["bút", "vở", "sách", "thước kẻ", "gôm tẩy"]
                    units = ["cái", "quyển", "cuốn", "chiếc", "cục"]
                    item = random.choice(items)
                    # Chọn unit phù hợp với item (đơn giản hóa)
                    unit = random.choice(units) if item not in ["vở", "sách"] else ("quyển" if item == "vở" else "cuốn")

                    price = random.randint(2, 10) * 1000 # Giá chẵn nghìn
                    quantity = random.randint(5, 20)
                    discount = random.choice([5, 10, 15, 20, 25, 30]) # % giảm giá phổ biến

                    original_cost = price * quantity
                    discount_amount = int(original_cost * discount / 100)
                    final_cost = original_cost - discount_amount
                    params = {
                        "item": item, "price": price, "unit": unit, "quantity": quantity,
                        "discount": discount, "original_cost": original_cost,
                        "discount_amount": discount_amount, "final_cost": final_cost
                    }
                 # Logic bể nước
                 elif "bể chứa nước hình hộp" in template_data["template"]:
                    length = random.randint(2, 10)
                    width = random.randint(1, length) # Rộng <= dài
                    height = random.randint(1, 5)
                    volume_m3 = length * width * height
                    volume_liters = volume_m3 * 1000
                    params = {
                        "length": length, "width": width, "height": height,
                        "volume_m3": volume_m3, "volume_liters": volume_liters
                    }
                 # Thêm logic cho các bài toán từ vựng khác

            elif problem_type == "Câu hỏi trắc nghiệm":
                # Logic phép tính + và *
                if "+ {num2} × {num3}" in template_data["template"]:
                    num1 = random.randint(1, 50)
                    num2 = random.randint(2, 10)
                    num3 = random.randint(2, 10)
                    mult_result = num2 * num3
                    correct = num1 + mult_result
                    # Tạo các đáp án sai một cách hợp lý
                    wrong_options = {
                        (num1 + num2) * num3, # Sai thứ tự ưu tiên
                        correct + random.randint(1, 5),
                        correct - random.randint(1, 5),
                        num1 * num2 + num3, # Sai thứ tự
                        num1 - mult_result, # Sai phép tính
                    }
                    wrong_options.discard(correct) # Loại bỏ nếu trùng đáp án đúng
                    # Chọn 3 đáp án sai khác nhau và khác đáp án đúng
                    wrongs = random.sample(list(wrong_options), min(3, len(wrong_options)))
                    while len(wrongs) < 3: # Bổ sung nếu không đủ 3 đáp án sai
                        new_wrong = correct + random.randint(-10, 10)
                        if new_wrong != correct and new_wrong not in wrongs:
                            wrongs.append(new_wrong)

                    options = wrongs + [correct]
                    random.shuffle(options) # Xáo trộn vị trí đáp án
                    params = {
                        "num1": num1, "num2": num2, "num3": num3,
                        "mult_result": mult_result, "correct": correct,
                        "wrong1": options[0], "wrong2": options[1],
                        "wrong3": options[2] # Giả sử correct nằm ở D ban đầu
                    }
                    # Gán lại A, B, C, D dựa trên vị trí của 'correct' sau khi shuffle
                    # Tìm index của correct
                    correct_index = options.index(correct)
                    options_dict = {
                        'A': options[0], 'B': options[1], 'C': options[2], 'D': options[3]
                    }
                    # Cập nhật template để hiển thị đúng A, B, C, D
                    template_data["template"] = template_data["template"].split('\n')[0] # Lấy dòng câu hỏi
                    template_data["template"] += f"\nA. {options_dict['A']}\nB. {options_dict['B']}\nC. {options_dict['C']}\nD. {options_dict['D']}"
                    # Cập nhật solution để chỉ đúng đáp án
                    correct_letter = chr(ord('A') + correct_index)
                    template_data["solution"] = template_data["solution"].replace("Đáp án đúng là B", f"Đáp án đúng là {correct_letter}")


                # Logic hình vuông từ chu vi -> diện tích
                elif "hình vuông có chu vi" in template_data["template"]:
                    side = random.randint(2, 15)
                    perimeter = 4 * side
                    correct = side * side # Diện tích đúng
                    # Tạo đáp án sai
                    wrong_options = {
                        perimeter, # Nhầm lẫn chu vi và diện tích
                        perimeter * side,
                        correct + random.randint(1, 10),
                        correct - random.randint(1, side) if correct > side else correct + side,
                        side * 2 # Nhầm lẫn cạnh với đường kính?
                    }
                    wrong_options.discard(correct)
                    wrongs = random.sample(list(wrong_options), min(3, len(wrong_options)))
                    while len(wrongs) < 3:
                        new_wrong = correct + random.randint(-15, 15)
                        if new_wrong != correct and new_wrong not in wrongs and new_wrong > 0:
                            wrongs.append(new_wrong)

                    options = wrongs + [correct]
                    random.shuffle(options)
                    params = {
                        "perimeter": perimeter, "side": side, "correct": correct,
                        "wrong1": options[0], "wrong2": options[1], "wrong3": options[2]
                    }
                    correct_index = options.index(correct)
                    options_dict = {
                        'A': options[0], 'B': options[1], 'C': options[2], 'D': options[3]
                    }
                    template_data["template"] = template_data["template"].split('\n')[0] # Lấy dòng câu hỏi
                    template_data["template"] += f"\nA. {options_dict['A']}cm²\nB. {options_dict['B']}cm²\nC. {options_dict['C']}cm²\nD. {options_dict['D']}cm²"
                    correct_letter = chr(ord('A') + correct_index)
                    template_data["solution"] = template_data["solution"].replace("Đáp án đúng là C", f"Đáp án đúng là {correct_letter}")
                # Thêm logic cho các câu trắc nghiệm khác

            elif problem_type == "Bài toán chuyển động":
                # Logic gặp nhau ngược chiều
                if "ngược chiều nhau" in template_data["template"]:
                     distance = random.randint(50, 300)
                     speed1 = random.randint(30, 60)
                     speed2 = random.randint(35, 70)
                     # Đảm bảo thời gian gặp nhau không quá nhỏ hoặc quá lớn
                     time_to_meet = distance / (speed1 + speed2)
                     if 0.5 < time_to_meet < 5: # Chỉ chấp nhận t từ 0.5h đến 5h
                         params = {
                             "distance": distance, "speed1": speed1, "speed2": speed2,
                             "result": round(time_to_meet, 2)
                         }
                     else: # Nếu thời gian không phù hợp, trả về None để thử loại khác
                         return None # Hoặc dùng vòng lặp để thử lại
                # Logic ca nô xuôi ngược dòng
                elif "ca nô đi xuôi dòng" in template_data["template"]:
                    # Tạo ngược để đảm bảo nghiệm đẹp
                    canoe_speed_real = random.randint(15, 30) # Vận tốc thực
                    water_speed = random.randint(2, 5)
                    if canoe_speed_real <= water_speed : return None # Vô lý
                    # Chọn thời gian xuôi/ngược hợp lý
                    time_down = random.uniform(1.5, 4.0) # Thời gian xuôi
                    # Tính quãng đường
                    speed_down = canoe_speed_real + water_speed
                    distance = speed_down * time_down
                    # Tính thời gian ngược
                    speed_up = canoe_speed_real - water_speed
                    if speed_up <= 0: return None # Vô lý
                    time_up = distance / speed_up
                    # Làm tròn thời gian để bài toán trông tự nhiên hơn
                    time_down_q = round(time_down, 1)
                    time_up_q = round(time_up, 1)
                    # Kiểm tra xem với thời gian làm tròn, kết quả có còn hợp lý không
                    # Giải lại với time_down_q, time_up_q để tìm v'
                    # (v' + ws)*td = (v' - ws)*tu => v'(td-tu) = -ws(td+tu) => v' = ws(td+tu)/(tu-td)
                    if time_up_q > time_down_q:
                         v_prime = water_speed * (time_down_q + time_up_q) / (time_up_q - time_down_q)
                         # Chỉ chấp nhận nếu v' gần với v ban đầu
                         if abs(v_prime - canoe_speed_real) < 1: # Sai số chấp nhận được
                              params = {
                                  "time_down": time_down_q,
                                  "time_up": time_up_q,
                                  "water_speed": water_speed,
                                  "const1": round(water_speed * time_down_q, 2),
                                  "const2": round(water_speed * time_up_q, 2),
                                  "diff_coeff": round(time_up_q - time_down_q, 2),
                                  "sum_const": round(water_speed * (time_down_q + time_up_q), 2),
                                  "canoe_speed": round(v_prime, 2),
                                  "distance": round((round(v_prime, 2) + water_speed) * time_down_q, 2)
                              }
                         else: return None
                    else: return None # Thời gian ngược phải lớn hơn thời gian xuôi

                else: return None # Nếu không khớp template nào

            elif problem_type == "Bài toán về tuổi":
                 # --- Logic mới cho Bài toán về tuổi ---
                if "sau bao nhiêu năm" in template_data["template"]: # Template 1: Tương lai
                    while True:
                        # Tạo ngược từ kết quả mong muốn
                        years_future = random.randint(3, 15) # Số năm trong tương lai
                        times = random.randint(2, 4)        # Số lần gấp
                        age_child_future = random.randint(years_future + 1, 25) # Tuổi con trong tương lai > số năm
                        age_father_future = times * age_child_future

                        # Tính tuổi hiện tại
                        age_child = age_child_future - years_future
                        age_father = age_father_future - years_future

                        # Kiểm tra điều kiện hợp lệ
                        if age_child >= 1 and age_father > age_child + 15 and age_father < 70: # Tuổi hợp lý
                            params = {
                                "age_father": age_father,
                                "age_child": age_child,
                                "times": times,
                                "result": years_future,
                                "times_age_child": times * age_child,
                                "diff_ages": age_father - (times * age_child),
                                "times_minus_1": times - 1
                                # Lưu ý: Thiếu key tuổi tương lai tường minh, sẽ được sửa ở bước 2
                            }
                            break # Thoát vòng lặp

                elif "Cách đây" in template_data["template"]: # Template 2: Quá khứ
                     while True:
                        # Tạo ngược từ quá khứ
                        years_ago = random.randint(2, 10)
                        times = random.randint(3, 7) # Thường gấp nhiều lần hơn trong quá khứ
                        age_child_past = random.randint(1, 10) # Tuổi con trong quá khứ
                        age_mother_past = times * age_child_past

                        # Tính tuổi hiện tại
                        age_child = age_child_past + years_ago
                        age_mother = age_mother_past + years_ago

                        # Tính hiệu số tuổi (không đổi)
                        age_diff = age_mother - age_child # = age_mother_past - age_child_past

                        # Kiểm tra điều kiện hợp lệ
                        # Đảm bảo age_diff chia hết cho (times - 1) theo công thức giải xuôi
                        if age_diff > 0 and age_mother > age_child + 18 and age_mother < 65 and age_diff % (times - 1) == 0:
                            params = {
                                "age_diff": age_diff,
                                "years_ago": years_ago,
                                "times": times,
                                "child_age": age_child,
                                "mother_age": age_mother,
                                # Các biến phụ cho solution template
                                "age_diff_minus_years": age_diff - years_ago,
                                "times_years_ago": times * years_ago,
                                "sum_const": (age_diff - years_ago) + (times * years_ago),
                                "times_minus_1": times - 1
                            }
                            break # Thoát vòng lặp

                else: return None # Không khớp template tuổi nào

            elif problem_type == "Bài toán chia kẹo":
                 # --- Logic mới cho Bài toán chia kẹo ---
                 # Chỉ có 1 template (3 người, tỷ lệ cố định)
                 ratio = random.randint(2, 4)
                 # Tạo số kẹo người 1 trước
                 person1_candies = random.randint(3, 15)
                 # Tính số kẹo người 2, 3
                 person2_candies = person1_candies * ratio
                 person3_candies = person2_candies * ratio # = person1_candies * ratio^2
                 # Tính tổng số kẹo
                 total = person1_candies + person2_candies + person3_candies
                 ratio_sq = ratio * ratio
                 sum_coeffs = 1 + ratio + ratio_sq

                 params = {
                     "total": total,
                     "ratio": ratio,
                     "person1_candies": person1_candies,
                     "person2_candies": person2_candies,
                     "person3_candies": person3_candies,
                     # Biến phụ cho solution
                     "ratio_sq": ratio_sq,
                     "sum_coeffs": sum_coeffs
                 }

            elif problem_type == "Bài toán hồ bơi":
                 # Đảm bảo net_flow > 0 và kết quả hợp lý
                 while True:
                     volume = random.randint(50, 500) # Dung tích nhỏ hơn cho thực tế
                     pipes_in = random.randint(1, 3)
                     flow_in = random.randint(10, 30)
                     pipes_out = random.randint(1, 2)
                     flow_out = random.randint(5, 15)

                     total_in = pipes_in * flow_in
                     total_out = pipes_out * flow_out
                     net_flow = total_in - total_out

                     if net_flow > 0: # Phải có nước vào nhiều hơn ra
                         time_to_fill = volume / net_flow
                         if 1 < time_to_fill < 20: # Thời gian hợp lý (1-20 giờ)
                             params = {
                                 "volume": volume, "pipes_in": pipes_in, "flow_in": flow_in,
                                 "pipes_out": pipes_out, "flow_out": flow_out,
                                 "total_in": total_in, "total_out": total_out,
                                 "net_flow": net_flow, "result": round(time_to_fill, 2)
                             }
                             break # Thoát vòng lặp

            elif problem_type == "Bài toán phân số":
                 # Đã chuyển định nghĩa gcd, simplify_fraction thành phương thức
                 num1 = random.randint(1, 9)
                 den1 = random.randint(num1 + 1, 15)
                 num2 = random.randint(1, 9)
                 den2 = random.randint(num2 + 1, 15)

                 frac1 = f"{num1}/{den1}"
                 frac2 = f"{num2}/{den2}"

                 # Tính tổng
                 sum_num = num1 * den2 + num2 * den1
                 sum_den = den1 * den2
                 sum_simplified = self._simplify_fraction(sum_num, sum_den) # Gọi self._simplify_fraction
                 sum_step1 = f"({num1}×{den2} + {num2}×{den1}) / ({den1}×{den2})"
                 sum_step2 = f"{sum_num}/{sum_den}"

                 # Tính tích
                 prod_num = num1 * num2
                 prod_den = den1 * den2
                 product_simplified = self._simplify_fraction(prod_num, prod_den) # Gọi self._simplify_fraction
                 product_step1 = f"({num1}×{num2}) / ({den1}×{den2})"
                 product_step2 = f"{prod_num}/{prod_den}"

                 params = {
                     "frac1": frac1, "frac2": frac2,
                     "sum_step1": sum_step1, "sum_step2": sum_step2, "sum_simplified": sum_simplified,
                     "product_step1": product_step1, "product_step2": product_step2, "product_simplified": product_simplified
                 }

            elif problem_type == "Bài toán công việc":
                 # Logic 2 người làm chung
                 if "hai người thợ cùng làm" in template_data["template"]:
                    time_a = random.randint(3, 12) # Thời gian làm một mình của A
                    time_b = random.randint(3, 12)
                    # Đảm bảo time_a != time_b để tránh nhàm chán
                    while time_b == time_a:
                        time_b = random.randint(3, 12)

                    # Năng suất mỗi giờ
                    rate_a = f"1/{time_a}"
                    rate_b = f"1/{time_b}"
                    # Năng suất chung
                    sum_rates_num = time_a + time_b
                    sum_rates_den = time_a * time_b
                    # Gọi phương thức của class để tối giản phân số
                    sum_rates_frac = self._simplify_fraction(sum_rates_num, sum_rates_den)
                    sum_rates_decimal = sum_rates_num / sum_rates_den
                    # Thời gian làm chung
                    time_together = 1 / sum_rates_decimal

                    params = {
                        "time_a": time_a, "time_b": time_b,
                        "sum_rates_frac": sum_rates_frac, # Đã sửa lỗi UnboundLocalError
                        "sum_rates_decimal": round(sum_rates_decimal, 4), # Làm tròn để hiển thị
                        "result": round(time_together, 2)
                    }
                 # Thêm logic cho các bài toán công việc khác nếu có

            elif problem_type == "Bài toán hỗn hợp":
                 # --- Logic mới cho Bài toán hỗn hợp ---
                 # Tạo ngược để đảm bảo nghiệm hợp lý
                 while True:
                    initial_total_mass = random.randint(200, 1000) # Khối lượng dd ban đầu
                    mass1 = random.randint(int(0.05 * initial_total_mass), int(0.3 * initial_total_mass)) # Muối ban đầu (5-30%)
                    volume1 = initial_total_mass - mass1 # Nước ban đầu

                    percent1 = (mass1 / initial_total_mass) * 100

                    # Chọn % đích cao hơn và hợp lý
                    percent2 = random.randint(int(percent1) + 5, min(90, int(percent1) + 25))

                    # Tính lượng muối cần thêm (x)
                    percent2_decimal = percent2 / 100
                    numerator = percent2_decimal * initial_total_mass - mass1
                    denominator = 1 - percent2_decimal

                    if denominator > 1e-6: # Tránh chia cho 0 hoặc số rất nhỏ
                        result = numerator / denominator
                        if 5 < result < initial_total_mass * 2 : # Lượng muối thêm vào phải dương và hợp lý
                            params = {
                                "mass1": mass1,
                                "volume1": volume1,
                                "percent2": percent2,
                                # Các biến phụ cho solution
                                "initial_total_mass": initial_total_mass,
                                "percent2_decimal": round(percent2_decimal, 4),
                                "const1": round(percent2_decimal * initial_total_mass, 2),
                                "const2": round(percent2_decimal * initial_total_mass - mass1, 2),
                                "one_minus_percent2_decimal": round(1 - percent2_decimal, 4),
                                "result": round(result, 2)
                            }
                            break # Thoát vòng lặp

            elif problem_type == "Bài toán số học":
                 # --- Logic mới cho Bài toán số học ---
                 while True:
                     digits = random.randint(2, 4) # Tăng lên 4 chữ số
                     # Tạo số gốc trước
                     start = 10**(digits - 1)
                     end = 10**digits - 1
                     num_result = random.randint(start, end)

                     # Tính tổng và tích thực tế
                     list_digits = [int(d) for d in str(num_result)]
                     actual_sum = sum(list_digits)
                     # Tính tích cẩn thận, tránh lỗi nếu có số 0 (dù start đảm bảo không có 0 đứng đầu)
                     actual_product = 1
                     has_zero = False
                     for d in list_digits:
                         if d == 0:
                             has_zero = True
                             actual_product = 0
                             break
                         actual_product *= d

                     # Chỉ chọn bài toán nếu tích > 0 (tránh trường hợp tích=0 quá dễ)
                     if actual_product > 0 and actual_sum > 0:
                         # Format các bước kiểm tra trong solution
                         check_sum_str = " + ".join(str(d) for d in list_digits)
                         check_prod_str = " × ".join(str(d) for d in list_digits)

                         params = {
                             "digits": digits,
                             "sum_digits": actual_sum,
                             "product_digits": actual_product,
                             "result": num_result,
                             "check_sum": check_sum_str,
                             "check_product": check_prod_str
                         }
                         break # Thoát vòng lặp

            elif problem_type == "Bài toán hình học":
                 # Logic hình chữ nhật
                 if "hình chữ nhật" in template_data["template"]:
                    length = random.randint(5, 25)
                    width = random.randint(3, length - 1) # Rộng < dài
                    area = length * width
                    perimeter = 2 * (length + width)
                    params = {
                        "length": length, "width": width, "area": area, "perimeter": perimeter,
                        "sum_len_wid": length + width
                    }
                 # Logic tam giác (Heron)
                 elif "tam giác có độ dài ba cạnh" in template_data["template"]:
                    # Tạo tam giác hợp lệ
                    while True:
                         a = random.randint(5, 20)
                         b = random.randint(5, 20)
                         # c phải thỏa mãn bất đẳng thức tam giác
                         min_c = max(1, abs(a - b) + 1)
                         max_c = a + b - 1
                         if min_c >= max_c: continue # Thử lại nếu khoảng không hợp lệ
                         c = random.randint(min_c, max_c)
                         # Kiểm tra lại lần nữa cho chắc
                         if a + b > c and a + c > b and b + c > a:
                              p = (a + b + c) / 2
                              # Đảm bảo p > a, p > b, p > c
                              if p > a and p > b and p > c:
                                  try:
                                      product_inside_sqrt = p * (p - a) * (p - b) * (p - c)
                                      if product_inside_sqrt > 0: # Đảm bảo trong căn > 0
                                           area = math.sqrt(product_inside_sqrt)
                                           params = {
                                               "a": a, "b": b, "c": c,
                                               "p_sum": a + b + c,
                                               "p": round(p, 2),
                                               "p_minus_a": round(p - a, 2),
                                               "p_minus_b": round(p - b, 2),
                                               "p_minus_c": round(p - c, 2),
                                               "product_inside_sqrt": round(product_inside_sqrt, 2),
                                               "result": round(area, 2)
                                           }
                                           break # Thoát vòng lặp khi tìm được tam giác hợp lệ
                                  except ValueError: # Bắt lỗi nếu căn bậc hai của số âm (dù đã kiểm tra)
                                       continue
                    # Kết thúc while True khi tìm được tam giác hợp lệ
                 # Logic hình tròn
                 elif "hình tròn có bán kính" in template_data["template"]:
                    radius = random.randint(2, 15)
                    pi = 3.14 # Theo đề bài
                    circumference = 2 * pi * radius
                    radius_sq = radius * radius
                    area = pi * radius_sq
                    params = {
                        "radius": radius,
                        "circumference": round(circumference, 2),
                        "radius_sq": radius_sq,
                        "area": round(area, 2)
                    }
                 # Thêm logic cho các hình khác nếu có

            elif problem_type == "Bài toán tỷ lệ":
                 # Logic nam/nữ
                 if "học sinh nam chiếm" in template_data["template"]:
                    total_students = random.randint(25, 55)
                    male_percent = random.randint(40, 60) # Tỷ lệ nam trong khoảng hợp lý

                    male_count_exact = total_students * male_percent / 100
                    # Làm tròn số nam hợp lý (ví dụ: làm tròn đến số nguyên gần nhất)
                    male_count = round(male_count_exact)
                    # Tính lại số nữ dựa trên số nam đã làm tròn
                    female_count = total_students - male_count

                    # Đảm bảo số nữ không âm (có thể xảy ra nếu làm tròn số nam lên quá cao)
                    if female_count < 0:
                        # Nếu làm tròn gây lỗi, thử làm tròn xuống
                        male_count = math.floor(male_count_exact)
                        female_count = total_students - male_count
                        if female_count < 0: return None # Vẫn lỗi, bỏ qua bài này

                    params = {
                        "total_students": total_students,
                        "male_percent": male_percent,
                        "male_count_exact": round(male_count_exact, 2), # Hiển thị số chính xác trong giải
                        "male_count": male_count, # Số nam đã làm tròn
                        "result": female_count # Số nữ
                    }
                 # Thêm logic cho bài toán tỷ lệ khác

            # --- Kết thúc phần tạo tham số ---

            # Kiểm tra xem params có được tạo thành công không
            if not params:
                 print(f"Cảnh báo: Không thể tạo tham số cho template của '{problem_type}'. Bỏ qua.")
                 return None # Bỏ qua nếu không tạo được tham số

            # Tạo câu hỏi và câu trả lời từ template và params
            question = template_data["template"].format(**params)
            solution = template_data["solution"].format(**params)

            return {
                "question": question,
                "solution": solution
            }

        except KeyError as e:
            print(f"Lỗi KeyError khi tạo bài toán {problem_type}: Thiếu key '{e}' trong params hoặc template.")
            return None
        except Exception as e:
            print(f"Lỗi không xác định khi tạo bài toán {problem_type}: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc() # In chi tiết lỗi để debug
            return None

    def generate_problems(self, total_count: int = 2345) -> List[Dict]:
        problems = []
        problem_types_available = list(self.problem_types.keys())
        weights = [self.problem_types[ptype]["weight"] for ptype in problem_types_available]

        # Tạo danh sách các loại bài toán dựa trên trọng số
        weighted_problem_types = random.choices(
            problem_types_available, weights=weights, k=total_count
        )

        print(f"\n=== Bắt đầu tạo {total_count} bài toán ===")
        # Đếm số lượng dự kiến cho mỗi loại để theo dõi
        expected_counts = {ptype: 0 for ptype in problem_types_available}
        for ptype in weighted_problem_types:
            expected_counts[ptype] += 1

        # In số lượng dự kiến
        print("Số lượng bài toán dự kiến cho mỗi loại:")
        for ptype, count in expected_counts.items():
            print(f"- {ptype}: {count} bài")

        generated_counts = {ptype: 0 for ptype in problem_types_available}
        skipped_counts = {ptype: 0 for ptype in problem_types_available}
        generated_success = 0 # Đếm số bài thành công
        skipped_total = 0 # Đếm tổng số bài bỏ qua

        # Tạo bài toán
        # Sử dụng vòng lặp để cố gắng đạt total_count bài thành công
        while generated_success < total_count and len(weighted_problem_types) > 0:
             # Chọn một loại bài toán từ danh sách còn lại
             # (Có thể dùng lại weighted_problem_types hoặc random.choices lại)
             ptype = random.choices(problem_types_available, weights=weights, k=1)[0]
             # Hoặc lấy từ danh sách đã tạo: ptype = weighted_problem_types.pop(0) # nếu muốn bám sát phân phối ban đầu hơn


             problem_data = self.generate_problem(ptype)

             if problem_data:
                 problems.append({
                     "id": generated_success + 1, # Đánh số id dựa trên số bài thành công
                     "type": ptype,
                     "question": problem_data["question"],
                     "solution": problem_data["solution"],
                     "difficulty": self._estimate_difficulty(problem_data["question"], problem_data["solution"]),
                     "tags": self._extract_tags(problem_data["question"])
                 })
                 generated_counts[ptype] += 1
                 generated_success += 1
             else:
                 skipped_counts[ptype] += 1
                 skipped_total += 1

             # In tiến trình (ví dụ: mỗi 100 bài thành công)
             if generated_success % 100 == 0 and generated_success > 0:
                  print(f"  Đã tạo thành công {generated_success}/{total_count} bài toán...")
                  # Có thể thêm thông tin về số lần thử nếu muốn:
                  # print(f"    (Tổng số lần thử: {generated_success + skipped_total}, Bỏ qua: {skipped_total})")

             # Thêm điều kiện dừng nếu số lần bỏ qua quá lớn để tránh vòng lặp vô hạn
             if skipped_total > total_count * 2: # Ví dụ: nếu bỏ qua gấp đôi số lượng cần tạo
                 print(f"CẢNH BÁO: Đã bỏ qua {skipped_total} lần thử. Dừng quá trình tạo sớm.")
                 break

        print(f"\n=== Hoàn thành tạo bài toán (hoặc dừng sớm) ===")
        print(f"Tổng số bài toán đã tạo thành công: {generated_success}")
        print(f"Tổng số lần thử bị bỏ qua: {skipped_total}")


        print("\n=== Thống kê số lượng bài toán đã tạo theo loại ===")
        for ptype in problem_types_available:
             # So sánh với số lượng dự kiến ban đầu có thể không còn chính xác nếu dùng random.choices liên tục
             # Nên chỉ hiển thị số lượng thực tế đã tạo và bỏ qua
             total_attempted_type = generated_counts[ptype] + skipped_counts[ptype]
             success_rate = (generated_counts[ptype] / total_attempted_type * 100) if total_attempted_type > 0 else 0
             print(f"- {ptype}: Tạo được: {generated_counts[ptype]}. Bỏ qua: {skipped_counts[ptype]}. Tỷ lệ thành công: {success_rate:.1f}%")


        # Thống kê độ khó
        difficulty_stats = {"Dễ": 0, "Trung bình": 0, "Khó": 0}
        for problem in problems:
            difficulty_stats[problem["difficulty"]] += 1

        print("\n=== Thống kê độ khó ===")
        if problems: # Tránh chia cho 0 nếu không tạo được bài nào
             for diff, count in difficulty_stats.items():
                 print(f"- {diff}: {count} bài ({count/len(problems)*100:.1f}%)")
        else:
             print("Không có bài toán nào được tạo để thống kê độ khó.")

        # Thống kê tags
        tag_stats = {}
        for problem in problems:
            for tag in problem["tags"]:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1

        print("\n=== Thống kê tags ===")
        if problems:
             # Sắp xếp tags theo số lượng giảm dần
             sorted_tags = sorted(tag_stats.items(), key=lambda item: item[1], reverse=True)
             for tag, count in sorted_tags:
                 print(f"- {tag}: {count} bài ({count/len(problems)*100:.1f}%)")
        else:
             print("Không có bài toán nào được tạo để thống kê tags.")


        # Xáo trộn thứ tự các bài toán cuối cùng
        random.shuffle(problems)
        # Cập nhật lại ID sau khi xáo trộn nếu cần thứ tự 1, 2, 3...
        for i, problem in enumerate(problems):
            problem["id"] = i + 1

        return problems

    def save_problems(self, problems: List[Dict]):
        filepath = './db/questions/problems.json'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_problems": len(problems),
                    "questions": problems
                }, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Đã lưu {len(problems)} bài toán vào {filepath}")
        except IOError as e:
            print(f"\nLỗi IOError khi lưu file JSON: {str(e)}")
        except Exception as e:
            print(f"\nLỗi không xác định khi lưu file JSON: {str(e)}")

    def _estimate_difficulty(self, question: str, solution: str) -> str:
        # Ước lượng đơn giản dựa trên độ dài solution so với question
        try:
             question_len = len(question.split())
             solution_len = len(solution.split())
             if question_len == 0: return "Dễ" # Tránh chia cho 0
             complexity_ratio = solution_len / question_len

             # Điều chỉnh ngưỡng dựa trên kinh nghiệm
             if complexity_ratio > 3.5 or solution_len > 100: # Lời giải dài hoặc rất dài
                 return "Khó"
             elif complexity_ratio > 1.8 or solution_len > 50: # Lời giải tương đối dài
                 return "Trung bình"
             else:
                 return "Dễ" # Lời giải ngắn
        except Exception:
             return "Trung bình" # Mặc định nếu có lỗi

    def _extract_tags(self, question: str) -> List[str]:
        # Phân loại tag dựa trên keywords (có thể cải thiện bằng NLP)
        keywords = {
            'số học': ['số', 'chữ số', 'chia hết', 'dư', 'ước', 'bội', 'nguyên tố', 'trung bình cộng', 'tổng', 'hiệu', 'tích', 'thương', 'phân số'],
            'hình học': ['tam giác', 'vuông', 'tròn', 'chữ nhật', 'hình vuông', 'diện tích', 'chu vi', 'cạnh', 'góc', 'bán kính', 'đường kính', 'thể tích', 'mét', 'cm'],
            'đại số': ['phương trình', 'ẩn số', 'biến số', 'tìm x', 'tỷ lệ', 'phần trăm', '%', 'gấp', 'lần', 'hệ thức'],
            'chuyển động': ['vận tốc', 'quãng đường', 'thời gian', 'km/h', 'xuất phát', 'gặp nhau', 'ca nô', 'ô tô', 'xe máy', 'xuôi dòng', 'ngược dòng'],
            'công việc': ['công việc', 'làm chung', 'làm riêng', 'vòi nước', 'bể', 'giờ', 'hoàn thành', 'năng suất'],
            'logic': ['hỏi', 'bao nhiêu', 'mấy', 'tìm', 'giải thích', 'chứng minh', 'suy luận', 'đúng', 'sai', 'so sánh'],
            'thực tế': ['tuổi', 'tiền', 'kẹo', 'cam', 'táo', 'gà', 'thỏ', 'nước', 'muối', 'dung dịch', 'học sinh', 'lớp học', 'cửa hàng', 'giảm giá']
        }
        tags = set() # Dùng set để tránh trùng lặp ban đầu
        text_lower = question.lower()
        for category, words in keywords.items():
            for word in words:
                # Tìm từ độc lập hoặc có dấu câu liền kề
                if f" {word} " in text_lower or text_lower.startswith(word + " ") or \
                   text_lower.endswith(" " + word) or f" {word}," in text_lower or \
                   f" {word}." in text_lower or f" {word}?" in text_lower:
                     tags.add(category)
                     break # Chỉ cần 1 keyword là đủ cho category đó
        # Ưu tiên một số tag đặc biệt
        if any(k in text_lower for k in ['tuổi']): tags.add('tuổi')
        if any(k in text_lower for k in ['công việc', 'làm chung', 'làm riêng', 'vòi nước']): tags.add('công việc')
        if any(k in text_lower for k in ['vận tốc', 'km/h', 'quãng đường', 'ca nô']): tags.add('chuyển động')
        if not tags: # Nếu không tìm thấy tag nào
            tags.add('tổng hợp')

        return sorted(list(tags)) # Trả về list đã sắp xếp

    def add_section(self, title: str):
        if self.y < 150: # Cần nhiều không gian hơn cho tiêu đề phần
            self.pdf.showPage()
            self.y = self.height - 50
            self.pdf.setFont('times', 10) # Reset font cho trang mới (nếu cần)
            self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}") # Thêm số trang

        self.pdf.setFont('timesbd', 16)
        section_title = f'Phần {self.current_section}: {title}'
        # Kiểm tra xem tiêu đề có bị tràn không (đơn giản hóa)
        title_width = self.pdf.stringWidth(section_title, 'timesbd', 16)
        if 50 + title_width > self.width - 50:
             # Xử lý tràn dòng (ví dụ: xuống dòng - cần phức tạp hơn)
             # Tạm thời chỉ vẽ
             self.pdf.drawString(50, self.y, section_title)
        else:
             self.pdf.drawString(50, self.y, section_title)

        self.y -= 30 # Khoảng cách sau tiêu đề phần
        self.current_problem = 1 # Reset số bài toán cho phần mới

    def add_problem(self, problem_text: str, solution_text: str = None):
        line_height = 15 # Chiều cao mỗi dòng (ước lượng)
        problem_prefix = f'Bài {self.current_problem}: '

        # Ước lượng chiều cao cần thiết cho bài toán và lời giải
        required_height = 30 # Khoảng cách đầu cuối
        required_height += len(self._wrap_text(problem_prefix + problem_text, 90)) * line_height
        if solution_text:
            required_height += 20 # Khoảng cách trước lời giải
            required_height += len(self._wrap_text('Hướng dẫn giải:', 90)) * line_height
            required_height += len(self._wrap_text(solution_text, 90)) * (line_height - 1) # Giải font nhỏ hơn

        # Nếu không đủ chỗ, sang trang mới
        if self.y < required_height + 50 : # Thêm lề dưới 50
            self.pdf.showPage()
            self.y = self.height - 50
            self.pdf.setFont('times', 10)
            self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")

        # Vẽ bài toán
        self.pdf.setFont('timesbd', 12)
        self.pdf.drawString(50, self.y, problem_prefix)
        # Vẽ phần text của câu hỏi với font thường
        self.pdf.setFont('times', 12)
        question_lines = self._wrap_text(problem_text, 90) # Wrap text câu hỏi
        x_offset = self.pdf.stringWidth(problem_prefix, 'timesbd', 12) # Lấy độ rộng prefix
        y_start = self.y

        # Vẽ từng dòng của câu hỏi
        for i, line in enumerate(question_lines):
            # Kiểm tra tràn trang khi đang vẽ câu hỏi
            if y_start - i * line_height < 50:
                self.pdf.showPage()
                self.y = self.height - 50
                self.pdf.setFont('times', 10)
                self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")
                # Reset font và vị trí cho dòng tiếp theo của câu hỏi
                self.pdf.setFont('timesbd', 12) # Reset prefix font nếu cần
                self.pdf.drawString(50, self.y, problem_prefix if i == 0 else "") # Vẽ lại prefix nếu là dòng đầu trên trang mới
                self.pdf.setFont('times', 12)
                y_start = self.y # Reset y bắt đầu vẽ trên trang mới
                x_offset = self.pdf.stringWidth(problem_prefix, 'timesbd', 12) if i == 0 else 0 # Recalculate offset


            current_x = 50 + x_offset if i == 0 else 50 # Dòng đầu thụt vào sau prefix, các dòng sau thụt vào lề
            # Kiểm tra lại y trước khi vẽ
            current_y = y_start - i * line_height
            if current_y < 50: # Double check trang mới
                 self.pdf.showPage()
                 self.y = self.height - 50
                 self.pdf.setFont('times', 10)
                 self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")
                 self.pdf.setFont('timesbd', 12)
                 self.pdf.drawString(50, self.y, problem_prefix if i == 0 else "")
                 self.pdf.setFont('times', 12)
                 y_start = self.y
                 current_y = y_start # Dòng này sẽ vẽ ở đầu trang mới

            self.pdf.drawString(current_x, current_y, line)
            self.y = current_y # Cập nhật y sau mỗi dòng được vẽ

        self.y -= line_height # Khoảng cách sau câu hỏi (đã cập nhật y ở trên)
        self.y -= 10 # Thêm khoảng cách nhỏ nữa

        # Vẽ lời giải nếu có
        if solution_text:
            # Ước lượng chiều cao lời giải
            solution_prefix_height = line_height
            solution_text_height = len(self._wrap_text(solution_text, 90)) * (line_height - 1)
            required_solution_height = solution_prefix_height + solution_text_height + 10 # + khoảng cách

            if self.y < required_solution_height + 50 : # Kiểm tra xem có đủ chỗ cho toàn bộ lời giải không
                self.pdf.showPage()
                self.y = self.height - 50
                self.pdf.setFont('times', 10)
                self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")

            # Vẽ tiền tố "Hướng dẫn giải"
            self.pdf.setFont('timesi', 11) # Font nghiêng cho lời giải
            solution_prefix = 'Hướng dẫn giải:'
            if self.y < 50 + line_height: # Kiểm tra lần cuối trước khi vẽ prefix
                 self.pdf.showPage()
                 self.y = self.height - 50
                 self.pdf.setFont('times', 10)
                 self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")
                 self.pdf.setFont('timesi', 11)

            self.pdf.drawString(50, self.y, solution_prefix)
            self.y -= line_height

            # Vẽ nội dung lời giải
            solution_lines = self._wrap_text(solution_text, 90)
            y_start_solution = self.y # Lưu vị trí y trước khi vẽ các dòng giải
            for i, line in enumerate(solution_lines):
                 current_y_solution = y_start_solution - i * (line_height - 1)
                 if current_y_solution < 50: # Kiểm tra tràn trang khi vẽ từng dòng giải
                     self.pdf.showPage()
                     self.y = self.height - 50
                     self.pdf.setFont('times', 10)
                     self.pdf.drawString(50, self.height - 30, f"Trang {self.pdf.getPageNumber()}")
                     self.pdf.setFont('timesi', 11) # Reset font nghiêng
                     y_start_solution = self.y # Reset y bắt đầu vẽ giải trên trang mới
                     current_y_solution = y_start_solution # Dòng này sẽ vẽ ở đầu trang mới

                 self.pdf.drawString(50, current_y_solution, line) # Giảm khoảng cách dòng cho giải

            self.y = y_start_solution - len(solution_lines) * (line_height - 1) # Cập nhật y sau khi vẽ hết giải

        self.y -= 20 # Khoảng cách sau mỗi bài toán
        self.current_problem += 1


    def _wrap_text(self, text: str, max_width_chars: int) -> List[str]:
        """Ngắt dòng văn bản để vừa với chiều rộng ước tính."""
        lines = []
        for paragraph in text.split('\n'): # Xử lý các đoạn đã xuống dòng sẵn
            words = paragraph.split()
            current_line = []
            current_length = 0
            # Sử dụng độ rộng thực tế của font nếu có thể (tốn kém hơn)
            # font_name = self.pdf._fontname # Lấy font hiện tại
            # font_size = self.pdf._fontsize # Lấy size hiện tại
            # max_width_points = (self.width - 100) # Chiều rộng vẽ tối đa (A4 width - 2*margin)

            for word in words:
                 # Ước lượng đơn giản bằng số ký tự
                 word_length = len(word)
                 if current_length + word_length + len(current_line) <= max_width_chars:
                     current_line.append(word)
                     current_length += word_length
                 # # Ước lượng bằng độ rộng thực tế (chính xác hơn nhưng chậm hơn)
                 # current_line_text = ' '.join(current_line + [word])
                 # if self.pdf.stringWidth(current_line_text, font_name, font_size) <= max_width_points:
                 #     current_line.append(word)
                 else:
                     if current_line: # Tránh thêm dòng trống nếu từ đầu tiên đã quá dài
                         lines.append(' '.join(current_line))
                     current_line = [word]
                     current_length = word_length # Reset length for the new line
                     # Xử lý từ dài hơn cả dòng -> ngắt từ (chưa làm)
                     # if self.pdf.stringWidth(word, font_name, font_size) > max_width_points:
                     #     # Logic ngắt từ ở đây
                     #     pass


            if current_line:
                lines.append(' '.join(current_line))
        # Xử lý trường hợp text rỗng ban đầu
        if not text:
            return []
        # Đảm bảo luôn trả về list, kể cả khi không ngắt được dòng nào
        return lines if lines else [text]


    def generate_all_problems_pdf_and_json(self, total_problems_to_generate=2345, problems_per_type_in_pdf=10):
        """Tạo cả file PDF và file JSON chứa các bài toán."""
        self.add_header()

        # 1. Tạo danh sách đầy đủ các bài toán cho JSON
        print("--- Bắt đầu tạo dữ liệu cho JSON ---")
        all_problems_data = self.generate_problems(total_problems_to_generate)

        if not all_problems_data:
            print("!!! Không tạo được bài toán nào cho JSON. Dừng quá trình.")
            return # Dừng nếu không có dữ liệu

        # 2. Lưu file JSON
        self.save_problems(all_problems_data)

        # 3. Tạo file PDF với số lượng bài toán giới hạn cho mỗi loại
        print("\n--- Bắt đầu tạo file PDF ---")
        # Lấy danh sách các loại bài toán có trong dữ liệu đã tạo
        # Sắp xếp theo thứ tự mong muốn (ví dụ: theo thứ tự trong self.problem_types)
        problem_types_in_order = [ptype for ptype in self.problem_types if any(p["type"] == ptype for p in all_problems_data)]
        # Hoặc sắp xếp theo Alphabet:
        # problem_types_in_order = sorted(list(set(p["type"] for p in all_problems_data)))


        # Thêm các bài toán vào PDF theo từng loại
        section_count = 0
        for problem_type in problem_types_in_order:
             # Lấy các bài toán thuộc loại này từ dữ liệu đã tạo
             type_problems = [p for p in all_problems_data if p["type"] == problem_type]

             if type_problems: # Chỉ thêm section nếu có bài toán loại đó
                section_count += 1
                self.current_section = section_count # Cập nhật số thứ tự section
                self.add_section(problem_type) # Thêm tiêu đề phần

                # Chọn số lượng bài toán giới hạn để thêm vào PDF
                problems_to_add_pdf = type_problems[:problems_per_type_in_pdf]
                print(f"  Thêm {len(problems_to_add_pdf)} bài toán loại '{problem_type}' vào PDF...")

                for problem in problems_to_add_pdf:
                    # Đảm bảo problem có đủ 'question' và 'solution'
                    if "question" in problem and "solution" in problem:
                         self.add_problem(problem["question"], problem["solution"])
                    else:
                         print(f"  Cảnh báo: Bài toán ID {problem.get('id', 'N/A')} loại '{problem_type}' thiếu question hoặc solution, bỏ qua khi thêm vào PDF.")

             else:
                 # Điều này không nên xảy ra nếu problem_types_in_order được tạo đúng cách
                 print(f"  Thông tin: Không tìm thấy bài toán loại '{problem_type}' trong dữ liệu (lọc lại).")


        # Lưu file PDF
        try:
             self.pdf.save()
             print(f"\n✓ Đã tạo và lưu file PDF: ./db/Nhung_bai_toan_co_dien.pdf")
        except Exception as e:
             print(f"\nLỗi khi lưu file PDF: {str(e)}")


if __name__ == '__main__':
    try:
        generator = UnifiedProblemGenerator()
        print("✓ Đã khởi tạo UnifiedProblemGenerator.")
        # Gọi hàm để tạo cả JSON và PDF
        generator.generate_all_problems_pdf_and_json(
            total_problems_to_generate=2345, # Tổng số bài tạo cho JSON
            problems_per_type_in_pdf=15      # Số bài mỗi loại hiển thị trong PDF (tăng lên 15)
        )
        print("\n✓ Hoàn thành quá trình tạo bài toán.")
    except Exception as e:
        print(f"\nLỗi nghiêm trọng trong quá trình thực thi: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()