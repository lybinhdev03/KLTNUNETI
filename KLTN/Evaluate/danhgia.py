import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sacrebleu

# Đường dẫn model
model_path = "D:\\KLTN\\mt5-lao-vietnamese_2025"
jsonl_file = "D:\\KLTN\\JSONL\\test.jsonl"  # Đường dẫn tới file JSONL

# Load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load dữ liệu
sources = []
references = []
predictions = []

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        lao_text = obj["translation"]["lo"]
        vi_ref = obj["translation"]["vi"]
        
        sources.append(lao_text)
        references.append(vi_ref)

        # Tokenize và sinh kết quả từ mô hình
        inputs = tokenizer(lao_text, return_tensors="pt", max_length=384, truncation=True, padding="max_length")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=384)

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred_text)

# Tính điểm BLEU
bleu = sacrebleu.corpus_bleu(predictions, [[ref] for ref in references])
print(f"BLEU score: {bleu.score:.2f}")
