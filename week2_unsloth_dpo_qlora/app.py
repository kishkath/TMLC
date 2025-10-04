from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.model_loader import load_finetuned_model
from configurations.config import INFERENCE_CONFIG, API_CONFIG
from utils.logging_setup import logger
import torch

app = FastAPI(title="Fitness QA Bot API", version="1.0")

logger.info("Starting Fitness QA Bot API service...")

logger.info("Loading fine-tuned model...")
model, tokenizer = load_finetuned_model(
    INFERENCE_CONFIG.get("model_path"),
    INFERENCE_CONFIG.get("max_seq_length"),
    INFERENCE_CONFIG.get("load_in_4bit")
)
logger.info("✅ Model loaded successfully.")
model.eval()

class QueryRequest(BaseModel):
    user_query: str
    max_new_tokens: int = INFERENCE_CONFIG.get("max_new_tokens", 256)


@app.get("/")
def root():
    return {"message": "Welcome to Fitness QA Bot API (Qwen3-0.6B Finetuned Model)"}


@app.post(API_CONFIG.get("endpoint", "/predict/"))
def predict(req: QueryRequest):
    try:
        logger.info(f"Received query: {req.user_query}")
        from core.inference_utils import predict as infer

        response = infer(model, tokenizer, req.user_query, system_prompt=INFERENCE_CONFIG.get("system_prompt"))

        logger.info(f"Prediction completed.")
        return {"response": response.strip()}

    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = API_CONFIG.get("host", "0.0.0.0")
    port = API_CONFIG.get("port", 8000)
    logger.info(f"Starting API at {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
