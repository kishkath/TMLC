import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from configurations.config import (
    MODE,
    BASE_MODEL_NAME,
    ADAPTER_PATH,
    MERGED_MODEL_PATH,
    GENERATION_PARAMS,
    QUESTIONS
)

# ------------------- Paths -------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Change working dir to repo root (optional)
os.chdir(REPO_ROOT)

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------- Functions -------------------
def save_merged_model(base_model_name, adapter_path, save_path):
    logger.info("Loading base model for merging...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")

    logger.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    logger.info("Merged model saved successfully.")


def run_inference(model_path, questions, generation_params, mode, base_model_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if mode == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    faq_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for question in questions:
        logger.info(f"Question: {question}")
        response = faq_pipeline(question, **generation_params)
        logger.info(f"Answer: {response[0]['generated_text']}\n")


# ------------------- Main -------------------
if __name__ == "__main__":
    adapter_path_abs = os.path.join(REPO_ROOT, ADAPTER_PATH)
    merged_model_path_abs = os.path.join(REPO_ROOT, MERGED_MODEL_PATH)
    base_model_name = BASE_MODEL_NAME
    generation_params = GENERATION_PARAMS
    questions = QUESTIONS
    mode = MODE

    if mode == "merged":
        save_merged_model(base_model_name, adapter_path_abs, merged_model_path_abs)
        run_inference(merged_model_path_abs, questions, generation_params, mode, base_model_name)
    else:
        run_inference(adapter_path_abs, questions, generation_params, mode, base_model_name)
