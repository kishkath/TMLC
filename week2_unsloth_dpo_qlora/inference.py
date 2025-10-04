from configurations.config import INFERENCE_MODEL_PATH, INFERENCE_MAX_SEQ_LENGTH, INFERENCE_LOAD_IN_4BIT
from core.model_loader import load_finetuned_model
from core.inference_utils import predict

if __name__ == "__main__":
    model, tokenizer = load_finetuned_model(INFERENCE_MODEL_PATH, INFERENCE_MAX_SEQ_LENGTH, INFERENCE_LOAD_IN_4BIT)
    print(f"✅ Fine-tuned model loaded from '{INFERENCE_MODEL_PATH}' for inference.")

    input_prompt = "Common mistakes in micronutrients (iron, calcium, vitamin D) for vegetarian lifters?"
    response = predict(model, tokenizer, input_prompt)
    print("\n💡 Prediction:\n", response)
