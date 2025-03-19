import json
import random
import math
from typing import List, Dict, Tuple

class ClassicalProblemGenerator:
    def __init__(self):
        self.problem_types = {
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
                "weight": 0.25  # 500 bài
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
                "weight": 0.15  # 300 bài
            },
            "Bài toán chia kẹo": {
                "templates": [
                    {
                        "template": "Có {total} viên kẹo chia cho {people} người. Người thứ hai được gấp {ratio} lần người thứ nhất, người thứ ba được gấp {ratio} lần người thứ hai. Hỏi mỗi người được bao nhiêu viên kẹo?",
                        "solution": "Gọi số kẹo người thứ nhất là x. Theo đề bài: x + {ratio}x + {ratio}^2x = {total}. Giải ra: x = {result} viên."
                    }
                ],
                "weight": 0.05  # 100 bài
            },
            "Bài toán hồ bơi": {
                "templates": [
                    {
                        "template": "Một hồ bơi có {volume} m³ nước. Có {pipes} ống nước chảy vào với lưu lượng {flow_in} m³/giờ và {drain} ống thoát với lưu lượng {flow_out} m³/giờ. Hỏi sau bao lâu hồ sẽ đầy?",
                        "solution": "Lưu lượng nước vào mỗi giờ: {pipes}×{flow_in}={total_in} m³/giờ. Lưu lượng nước ra mỗi giờ: {drain}×{flow_out}={total_out} m³/giờ. Lưu lượng thực tế: {total_in}-{total_out}={net_flow} m³/giờ. Thời gian cần: {volume}/{net_flow}={result} giờ."
                    }
                ],
                "weight": 0.1  # 200 bài
            },
            "Bài toán phân số": {
                "templates": [
                    {
                        "template": "Cho hai phân số {frac1} và {frac2}. Tìm tổng và tích của hai phân số này.",
                        "solution": "Quy đồng mẫu số: {process}. Tổng = {sum}, Tích = {product}."
                    }
                ],
                "weight": 0.1  # 200 bài
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
                "weight": 0.1  # 200 bài
            },
            "Bài toán hỗn hợp": {
                "templates": [
                    {
                        "template": "Hòa tan {mass1}g muối vào {volume1}ml nước được dung dịch {percent1}%. Hỏi phải thêm bao nhiêu gam muối nữa để được dung dịch {percent2}%?",
                        "solution": "Khối lượng muối ban đầu: {mass1}g. Khối lượng dung dịch: {mass1} + {volume1} = {total_mass}g. Gọi số gam muối cần thêm là x. Ta có: ({mass1} + x)/({total_mass} + x) = {percent2}/100. Giải ra: x = {result}g."
                    }
                ],
                "weight": 0.1  # 200 bài
            },
            "Bài toán số học": {
                "templates": [
                    {
                        "template": "Tìm số tự nhiên có {digits} chữ số, biết rằng tổng các chữ số là {sum_digits} và tích các chữ số là {product_digits}.",
                        "solution": "Ta thử các số có {digits} chữ số thỏa mãn điều kiện. Số cần tìm là: {result}."
                    }
                ],
                "weight": 0.05  # 100 bài
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
                    }
                ],
                "weight": 0.05  # 100 bài
            },
            "Bài toán tỷ lệ": {
                "templates": [
                    {
                        "template": "Một lớp học có {total_students} học sinh. Số học sinh nam chiếm {male_percent}% tổng số học sinh. Hỏi lớp có bao nhiêu học sinh nữ?",
                        "solution": "Số học sinh nam = {total_students} × {male_percent}/100 = {male_count} học sinh. Số học sinh nữ = {total_students} - {male_count} = {result} học sinh."
                    }
                ],
                "weight": 0.05  # 100 bài
            }
        }

    def generate_problem(self, problem_type: str) -> Dict:
        template_data = random.choice(self.problem_types[problem_type]["templates"])
        template = template_data["template"]
        solution_template = template_data["solution"]

        try:
            if problem_type == "Bài toán chuyển động":
                if "tàu hỏa" in template:
                    train_length = random.randint(100, 200)
                    bridge_length = random.randint(300, 500)
                    speed = random.randint(40, 80)
                    total_length = train_length + bridge_length
                    time = round(total_length / (speed * 1000 / 3600), 1)
                    result = round(total_length / time, 2)
                    return {
                        "question": template.format(train_length=train_length, speed=speed, 
                                                 bridge_length=bridge_length, time=time),
                        "solution": solution_template.format(total_length=total_length, time=time, result=result)
                    }
                else:
                    distance = random.randint(100, 500)
                    speed1 = random.randint(30, 80)
                    speed2 = random.randint(20, 60)
                    if "gió ngược" in template:
                        time1 = round(distance/speed1, 2)
                        time2 = round(distance/speed2, 2)
                        result = round(time1 + time2, 2)
                        return {
                            "question": template.format(speed1=speed1, speed2=speed2, distance=distance),
                            "solution": solution_template.format(speed1=speed1, speed2=speed2, 
                                                              distance=distance, time1=time1, 
                                                              time2=time2, result=result)
                        }
                    else:
                        result = round(distance/(speed1+speed2), 2)
                        return {
                            "question": template.format(distance=distance, speed1=speed1, speed2=speed2),
                            "solution": solution_template.format(distance=distance, speed1=speed1, 
                                                              speed2=speed2, result=result)
                        }
            elif problem_type == "Bài toán về tuổi":
                if "mẹ hơn" in template:
                    age_diff = random.randint(20, 30)
                    years_ago = random.randint(5, 10)
                    times = random.randint(2, 4)
                    child_age = random.randint(15, 25)
                    mother_age = child_age + age_diff
                    if child_age - years_ago > 0 and mother_age - years_ago > 0:
                        return {
                            "question": template.format(age_diff=age_diff, years_ago=years_ago, times=times),
                            "solution": solution_template.format(age_diff=age_diff, years_ago=years_ago,
                                                              times=times, child_age=child_age,
                                                              mother_age=mother_age)
                        }
                    else:
                        return self.generate_problem(problem_type)
                else:
                    age_father = random.randint(30, 50)
                    age_child = random.randint(5, 15)
                    times = random.randint(2, 4)
                    result = round((age_father - times*age_child)/(times-1), 1)
                    if result > 0:
                        return {
                            "question": template.format(age_father=age_father, age_child=age_child, times=times),
                            "solution": solution_template.format(age_father=age_father, age_child=age_child,
                                                              times=times, result=result)
                        }
                    else:
                        return self.generate_problem(problem_type)
            elif problem_type == "Bài toán chia kẹo":
                total = random.randint(100, 300)
                people = 3
                ratio = random.randint(2, 4)
                result = round(total/(1 + ratio + ratio**2))
                return {
                    "question": template.format(total=total, people=people, ratio=ratio),
                    "solution": solution_template.format(total=total, ratio=ratio, result=result)
                }
            elif problem_type == "Bài toán hồ bơi":
                volume = random.randint(1000, 5000)
                pipes = random.randint(2, 4)
                flow_in = random.randint(20, 50)
                drain = random.randint(1, 2)
                flow_out = random.randint(10, 30)
                total_in = pipes * flow_in
                total_out = drain * flow_out
                net_flow = total_in - total_out
                result = round(volume/net_flow, 2)
                return {
                    "question": template.format(volume=volume, pipes=pipes, flow_in=flow_in, 
                                             drain=drain, flow_out=flow_out),
                    "solution": solution_template.format(volume=volume, pipes=pipes, flow_in=flow_in,
                                                      drain=drain, flow_out=flow_out, total_in=total_in,
                                                      total_out=total_out, net_flow=net_flow, result=result)
                }
            elif problem_type == "Bài toán phân số":
                num1, den1 = random.randint(1, 10), random.randint(2, 10)
                num2, den2 = random.randint(1, 10), random.randint(2, 10)
                frac1 = f"{num1}/{den1}"
                frac2 = f"{num2}/{den2}"
                common_den = den1 * den2
                sum_num = num1 * den2 + num2 * den1
                product_num = num1 * num2
                product_den = den1 * den2
                process = f"({num1}×{den2} + {num2}×{den1})/({den1}×{den2})"
                return {
                    "question": template.format(frac1=frac1, frac2=frac2),
                    "solution": solution_template.format(
                        process=process,
                        sum=f"{sum_num}/{common_den}",
                        product=f"{product_num}/{product_den}"
                    )
                }
            elif problem_type == "Bài toán công việc":
                if "ba thợ" in template.lower():
                    total_time = random.randint(6, 12)
                    extra_time = random.randint(2, 6)
                    x = 1/(3*(total_time + extra_time))  # năng suất thợ A
                    result_y = 1/total_time - 3*x  # năng suất thợ C
                    result = round(1/result_y, 2)  # thời gian thợ C làm một mình
                    return {
                        "question": template.format(total_time=total_time, extra_time=extra_time),
                        "solution": solution_template.format(total_time=total_time, extra_time=extra_time,
                                                          result_y=round(result_y, 4), result=result)
                    }
                else:
                    time_a = random.randint(4, 12)
                    time_b = random.randint(6, 15)
                    result = round((time_a * time_b)/(time_a + time_b), 2)
                    return {
                        "question": template.format(time_a=time_a, time_b=time_b),
                        "solution": solution_template.format(time_a=time_a, time_b=time_b, result=result)
                    }
            elif problem_type == "Bài toán hỗn hợp":
                mass1 = random.randint(50, 200)
                volume1 = random.randint(500, 1000)
                percent1 = round(mass1/(mass1 + volume1) * 100, 1)
                percent2 = round(percent1 + random.randint(5, 15), 1)
                total_mass = mass1 + volume1
                result = round((percent2 * total_mass - 100 * mass1)/(100 - percent2), 1)
                return {
                    "question": template.format(mass1=mass1, volume1=volume1, percent1=percent1, percent2=percent2),
                    "solution": solution_template.format(mass1=mass1, volume1=volume1, total_mass=total_mass,
                                                      percent2=percent2, result=result)
                }
            elif problem_type == "Bài toán số học":
                digits = random.randint(2, 3)
                if digits == 2:
                    sum_digits = random.randint(5, 15)
                    product_digits = random.randint(10, 40)
                else:
                    sum_digits = random.randint(10, 20)
                    product_digits = random.randint(30, 100)
                result = self._find_number_with_sum_product(digits, sum_digits, product_digits)
                if result is None:
                    # Nếu không tìm được số thỏa mãn, thử lại với các giá trị khác
                    return self.generate_problem(problem_type)
                return {
                    "question": template.format(digits=digits, sum_digits=sum_digits, product_digits=product_digits),
                    "solution": solution_template.format(digits=digits, result=result)
                }
            elif problem_type == "Bài toán hình học":
                if "tam giác" in template.lower():
                    # Tạo tam giác hợp lệ
                    a = random.randint(5, 15)
                    b = random.randint(5, 15)
                    c = random.randint(max(abs(a-b)+1, 5), a+b-1)
                    p = (a + b + c)/2  # nửa chu vi
                    area = round(math.sqrt(p*(p-a)*(p-b)*(p-c)), 2)
                    return {
                        "question": template.format(a=a, b=b, c=c),
                        "solution": solution_template.format(a=a, b=b, c=c, p=p, result=area)
                    }
                else:
                    length = random.randint(5, 20)
                    width = random.randint(3, 10)
                    area = length * width
                    perimeter = 2 * (length + width)
                    return {
                        "question": template.format(length=length, width=width),
                        "solution": solution_template.format(length=length, width=width,
                                                          area=area, perimeter=perimeter)
                    }
            elif problem_type == "Bài toán tỷ lệ":
                total_students = random.randint(30, 50)
                male_percent = random.randint(40, 60)
                male_count = round(total_students * male_percent / 100)
                result = total_students - male_count
                return {
                    "question": template.format(total_students=total_students, male_percent=male_percent),
                    "solution": solution_template.format(total_students=total_students, male_percent=male_percent,
                                                      male_count=male_count, result=result)
                }
        except Exception as e:
            print(f"Lỗi khi tạo bài toán {problem_type}: {str(e)}")
            return None
        return None

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

    def generate_problems(self, total_count: int = 2000) -> List[Dict]:
        problems = []
        problem_types = list(self.problem_types.keys())
        
        # Tính số lượng bài toán cho mỗi loại
        type_counts = {}
        remaining_count = total_count
        
        for ptype in problem_types[:-1]:  # Trừ loại cuối cùng
            count = round(total_count * self.problem_types[ptype]["weight"])
            type_counts[ptype] = count
            remaining_count -= count
        
        # Loại cuối cùng lấy số còn lại
        type_counts[problem_types[-1]] = remaining_count
        
        # Tạo bài toán theo số lượng đã tính
        for ptype, count in type_counts.items():
            for i in range(count):
                problem = self.generate_problem(ptype)
                if problem:  # Kiểm tra nếu tạo thành công
                    problems.append({
                        "id": len(problems) + 1,
                        "type": "Classical Problems",
                        "subtype": ptype,
                        "question": problem["question"],
                        "solution": problem["solution"],
                        "difficulty": self._estimate_difficulty(problem["question"], problem["solution"]),
                        "tags": self._extract_tags(problem["question"])
                    })
        
        # Xáo trộn thứ tự các bài toán
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

    def save_problems(self, problems: List[Dict]):
        filepath = './db/questions/classical_problems.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "section": "Classical Problems",
                "questions": problems
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã tạo và lưu {len(problems)} bài toán cổ điển vào {filepath}")

if __name__ == '__main__':
    generator = ClassicalProblemGenerator()
    problems = generator.generate_problems(500)  # Tạo 2000 bài toán
    generator.save_problems(problems) 