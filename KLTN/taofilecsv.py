import pandas as pd

# Đọc file văn bản
with open("vie.txt", "r", encoding="utf-8") as f:
    vietnamese_sentences = [line.strip() for line in f.readlines()]

with open("lao.txt", "r", encoding="utf-8") as f:
    lao_sentences = [line.strip() for line in f.readlines()]

# Kiểm tra độ dài hai file có khớp không
assert len(vietnamese_sentences) == len(lao_sentences), "Hai file không có cùng số dòng!"

# Tạo DataFrame
df = pd.DataFrame({"vi": vietnamese_sentences, "lo": lao_sentences})

# Lưu thành file CSV
df.to_csv("vi_lo_data.csv", index=False, encoding="utf-8")

print("Đã tạo file vi_lo_data.csv thành công!")
