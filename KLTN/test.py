from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sacrebleu

# Đường dẫn đến model
model_path = "D:\\KLTN\\mt5-vietnamese-lao_2025"

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Load model
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Kiểm tra thiết bị (GPU nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Test model với một câu đầu vào
src = "Hôm nay tôi phải đi học"
input_ids = tokenizer(src, return_tensors="pt", max_length=384, padding="max_length", truncation=True).input_ids.to(device)

# Sinh đầu ra từ model
try:
    outputs = model.generate(input_ids, max_new_tokens=384)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Translation Output:", output_text)
except Exception as e:
    print(f"Error during model inference: {e}")
    exit()

# Đánh giá BLEU
reference_translation = "ມື້ນີ້ຂ້ອຍຕ້ອງໄປໂຮງຮຽນ"  # Thay bằng câu dịch chuẩn thực tế
bleu_score = sacrebleu.corpus_bleu([output_text], [[reference_translation]]).score
print(f"BLEU score: {bleu_score}")
