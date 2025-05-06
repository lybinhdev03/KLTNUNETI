from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc thay "*" bằng ["http://localhost:3000"] nếu chỉ muốn cho phép từ frontend của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load hai model và tokenizer
model_path_vi_lo = r"D:\KLTN\mt5-vietnamese-lao_2025"
model_path_lo_vi = r"D:\KLTN\mt5-lao-vietnamese_2025"

tokenizer_vi_lo = MT5Tokenizer.from_pretrained(model_path_vi_lo)
model_vi_lo = MT5ForConditionalGeneration.from_pretrained(model_path_vi_lo)
model_vi_lo.eval()

tokenizer_lo_vi = MT5Tokenizer.from_pretrained(model_path_lo_vi)
model_lo_vi = MT5ForConditionalGeneration.from_pretrained(model_path_lo_vi)
model_lo_vi.eval()

# Request body
class TranslateRequest(BaseModel):
    text: str
    direction: str  # "vi-lo" hoặc "lo-vi"

@app.post("/translate/")
async def translate(request: TranslateRequest):
    text = request.text.strip()
    direction = request.direction.lower().strip()

    if not text:
        return {"error": "Text input is empty"}
    if direction not in ["vi-lo", "lo-vi"]:
        return {"error": "Direction must be 'vi-lo' or 'lo-vi'"}

    # Chọn model và tokenizer phù hợp
    if direction == "vi-lo":
        tokenizer = tokenizer_vi_lo
        model = model_vi_lo
    else:
        tokenizer = tokenizer_lo_vi
        model = model_lo_vi

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"translated_text": translated_text}

# Để chạy server, sử dụng lệnh sau trong terminal:
# uvicorn apiv2:app --host 0.0.0.0 --port 8000 --reload