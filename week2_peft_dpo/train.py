import logging
import sys
import torch
import os
from pathlib import Path

# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------------
# Resolve paths
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "configurations" / "config.json"

# -------------------------------
# Import project modules
# -------------------------------
from configurations.config import Config
from dataset.data_utils import create_dataset, tokenize_dataset
from finetuning.model import load_model_and_tokenizer
from finetuning.trainer import get_lora_config, get_training_args, get_trainer

# -------------------------------
# Load configuration
# -------------------------------
cfg = Config(CONFIG_PATH)

# -------------------------------
# Setup device
# -------------------------------
device = cfg.device
n_gpus = torch.cuda.device_count() if device == "cuda" else 0
logger.info(f"Device: {device}, GPUs available: {n_gpus}")

# -------------------------------
# Load model & tokenizer
# -------------------------------
model, tokenizer = load_model_and_tokenizer(cfg.model_name, device=device)

# -------------------------------
# Load datasets
# -------------------------------
train_dataset, eval_dataset = create_dataset(cfg.train_file, cfg.eval_file)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=cfg.max_length)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=cfg.max_length)

# -------------------------------
# Prepare training
# -------------------------------
lora_config = get_lora_config(cfg)
training_args = get_training_args(cfg)
trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

# -------------------------------
# Run training
# -------------------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# -------------------------------
# Save model & tokenizer
# -------------------------------
logger.info("Saving LoRA adapter & tokenizer...")
output_dir = cfg.training.get("output_dir")

# Avoid AttributeError with PEFT models
if hasattr(trainer.model, "module"):
    trainer.model.module.save_pretrained(output_dir)
else:
    trainer.model.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)
logger.info(f"All artifacts saved to {output_dir}")
