import evaluate
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import json

model_path = "D:\\KLTN\\mt5-vietnamese-lao_2025"
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model.eval()

# Load test data
with open("./JSONL/trainvl.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

sources = [item['translation']['vi'] for item in data]
references = [[item['translation']['lo']] for item in data]  # mỗi ref là 1 list con

def translate_batch(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=384)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        results.extend([tokenizer.decode(t, skip_special_tokens=True) for t in outputs])
    return results

predictions = translate_batch(sources)

# Load BLEU metric từ thư viện evaluate
bleu = evaluate.load("sacrebleu")
bleu_score = bleu.compute(predictions=predictions, references=references)

print(f"BLEU Score: {bleu_score['score']:.2f}")
