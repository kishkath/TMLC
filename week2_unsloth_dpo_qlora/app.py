from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.model_loader import load_finetuned_model
from configurations.config import CONFIG
import torch

app = FastAPI(title="Fitness QA Bot API", version="1.0")

# Load the model once at startup
model, tokenizer = load_finetuned_model(CONFIG)
model.eval()

# ----------------------------- Request Schema -----------------------------
class QueryRequest(BaseModel):
    user_query: str
    max_new_tokens: int = 256

# ----------------------------- Routes -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Fitness QA Bot API (Qwen3-0.6B Finetuned Model)"}

@app.post("/predict/")
def predict(req: QueryRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI fitness assistant for gym-goers and vegetarians."},
            {"role": "user", "content": req.user_query},
        ]

        # Chat template for Qwen models
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"response": response.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
