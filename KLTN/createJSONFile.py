import json
import re

def clean_sentence(sentence):
    """
    Loại bỏ tiền tố dạng "SNT.xxxx.x" trong câu nếu có.
    """
    return re.sub(r'^SNT\.\d+\.\d+\s+', '', sentence)

def convert_to_jsonl(vietnamese_file, lao_file, output_file):
    """
    Chuyển đổi hai file chứa câu tiếng Việt và tiếng Lào thành file JSONL.
    
    Args:
        vietnamese_file (str): Đường dẫn đến file chứa câu tiếng Việt.
        lao_file (str): Đường dẫn đến file chứa câu tiếng Lào.
        output_file (str): Đường dẫn đến file JSONL đầu ra.
    """
    with open(vietnamese_file, 'r', encoding='utf-8') as viet_f, \
         open(lao_file, 'r', encoding='utf-8') as lao_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        viet_sentences = viet_f.readlines()
        lao_sentences = lao_f.readlines()

        if len(viet_sentences) != len(lao_sentences):
            raise ValueError("Số lượng câu trong hai file không khớp.")

        for viet, lao in zip(viet_sentences, lao_sentences):
            viet = clean_sentence(viet.strip())  # Làm sạch câu tiếng Việt
            lao = clean_sentence(lao.strip())  # Làm sạch câu tiếng Lào
            if viet and lao:  # Bỏ qua các dòng trống
                json_line = {
                    "translation": {
                        "vi": viet,
                        "lo": lao
                    }
                }
                out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Đã chuyển đổi xong. File JSONL được lưu tại: {output_file}")

# Đường dẫn tới file
vietnamese_file = "./Data/data_vi.txt"  # Thay bằng đường dẫn thật
lao_file = "./Data/data_lo.txt"  # Thay bằng đường dẫn thật
output_file = "./JSONL/trainvl.jsonl"  # File JSONL đầu ra

# Thực hiện chuyển đổi
convert_to_jsonl(vietnamese_file, lao_file, output_file)
