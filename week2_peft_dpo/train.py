import logging
import sys
import torch
import os

# ------------------- Paths -------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Change working dir to repo root (optional)
os.chdir(REPO_ROOT)

# ------------------- Imports -------------------
from configurations.config import (
    DEVICE,
    MODEL_NAME,
    TRAIN_FILE,
    EVAL_FILE,
    MAX_LENGTH,
    LORA_CONFIG,
    TRAINING_CONFIG,
)
from dataset.data_utils import create_dataset, tokenize_dataset
from finetuning.model import load_model_and_tokenizer
from finetuning.trainer import get_lora_config, get_training_args, get_trainer

# ------------------- Logging -------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------- Device -------------------
device = DEVICE
n_gpus = torch.cuda.device_count() if device == "cuda" else 0
logger.info(f"Device: {device}, GPUs available: {n_gpus}")

# ------------------- Model & Tokenizer -------------------
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)

# ------------------- Dataset -------------------
train_file_path = os.path.join(REPO_ROOT, TRAIN_FILE)
eval_file_path = os.path.join(REPO_ROOT, EVAL_FILE)

train_dataset, eval_dataset = create_dataset(train_file_path, eval_file_path)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=MAX_LENGTH)

# ------------------- LoRA & Training -------------------
lora_config = get_lora_config(LORA_CONFIG)
training_args = get_training_args(TRAINING_CONFIG)
trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

# ------------------- Training -------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# ------------------- Save -------------------
output_dir = os.path.join(REPO_ROOT, TRAINING_CONFIG.get("output_dir"))
logger.info("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"All artifacts saved to {output_dir}")
