import logging
import sys
import torch
from pathlib import Path
import os

# ------------------- Paths -------------------
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)

# ------------------- Imports -------------------
from configurations.config import (
    DEVICE,
    N_GPUS,
    MODEL_NAME,
    TRAIN_FILE,
    EVAL_FILE,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_DROPOUT,
    LORA_BIAS,
    LORA_TASK_TYPE,
    TRAINING_ARGS,
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

# ------------------- Model & Tokenizer -------------------
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE)

# ------------------- Dataset -------------------
train_dataset, eval_dataset = create_dataset(TRAIN_FILE, EVAL_FILE)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=MAX_LENGTH)

# ------------------- LoRA & Training -------------------
lora_config = get_lora_config(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type=LORA_TASK_TYPE
)
training_args = get_training_args(TRAINING_ARGS)

trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

# ------------------- Training -------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# ------------------- Save -------------------
logger.info("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(TRAINING_ARGS.get("output_dir"))
tokenizer.save_pretrained(TRAINING_ARGS.get("output_dir"))
logger.info(f"All artifacts saved to {TRAINING_ARGS.get('output_dir')}")
