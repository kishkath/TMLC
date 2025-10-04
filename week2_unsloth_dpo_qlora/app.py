from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.model_loader import load_finetuned_model
from configurations.config import INFERENCE_CONFIG, logger
import torch

app = FastAPI(title="Fitness QA Bot API", version="1.0")

logger.info("Loading fine-tuned model...")
model, tokenizer = load_finetuned_model(
    INFERENCE_CONFIG.get("model_path"),
    INFERENCE_CONFIG.get("max_seq_length"),
    INFERENCE_CONFIG.get("load_in_4bit")
)
logger.info("Model loaded successfully.")
model.eval()

class QueryRequest(BaseModel):
    user_query: str
    max_new_tokens: int = INFERENCE_CONFIG.get("max_new_tokens", 256)


@app.get("/")
def root():
    return {"message": "Welcome to Fitness QA Bot API (Qwen3-0.6B Finetuned Model)"}


@app.post("/predict/")
def predict(req: QueryRequest):
    try:
        logger.info(f"Received query: {req.user_query}")
        from core.inference_utils import predict as infer

        response = infer(model, tokenizer, req.user_query, system_prompt=INFERENCE_CONFIG.get("system_prompt"))

        logger.info(f"Prediction completed.")
        return {"response": response.strip()}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from configurations.config import API_CONFIG

    logger.info(f"Starting API at {API_CONFIG.get('host')}:{API_CONFIG.get('port')}")
    uvicorn.run(app, host=API_CONFIG.get("host", "0.0.0.0"), port=API_CONFIG.get("port", 8000))
