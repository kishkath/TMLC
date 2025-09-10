import logging
import sys
import torch
from pathlib import Path

# ------------------- Set working directory (optional) -------------------
ROOT_DIR = Path(__file__).resolve().parent
# Uncomment if you want to change working directory
# import os
# os.chdir(ROOT_DIR)

# ------------------- Logging -------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------- Config -------------------
from configurations.config import (
    MODEL_NAME, TRAIN_FILE, EVAL_FILE, DEVICE, MAX_LENGTH,
    LORA_R, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT, LORA_BIAS, LORA_TASK_TYPE,
    TRAINING_ARGS
)

# ------------------- Dataset Utilities -------------------
from dataset.data_utils import create_dataset, tokenize_dataset

# ------------------- Model & Trainer Utilities -------------------
from finetuning.model import load_model_and_tokenizer
from finetuning.trainer import get_lora_config, get_training_args, get_trainer

# ------------------- Device -------------------
device = DEVICE
n_gpus = torch.cuda.device_count() if device == "cuda" else 0
logger.info(f"Device: {device}, GPUs available: {n_gpus}")

# ------------------- Load Model & Tokenizer -------------------
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)
logger.info("Model and tokenizer loaded successfully.")

# ------------------- Load & Tokenize Datasets -------------------
train_dataset, eval_dataset = create_dataset(TRAIN_FILE, EVAL_FILE)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=MAX_LENGTH)
logger.info("Datasets tokenized successfully.")

# ------------------- LoRA Config & Training Args -------------------
lora_config = get_lora_config(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type=LORA_TASK_TYPE
)

training_args = get_training_args(TRAINING_ARGS)

# ------------------- Trainer -------------------
trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

# ------------------- Training -------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# ------------------- Save LoRA Adapter & Tokenizer -------------------
output_dir = TRAINING_ARGS.get("output_dir", "./qwen2.5-0.5b-finetuned")
logger.info(f"Saving LoRA adapter & tokenizer to {output_dir}...")

try:
    # No module attribute required for recent PEFT versions
    trainer.model.save_pretrained(output_dir)
except AttributeError:
    # fallback if multi-GPU model
    trainer.model.module.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)
logger.info(f"All artifacts saved successfully to {output_dir}")
