import logging
import sys
import torch
import os

# ------------------- Paths -------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_ROOT, "configurations/config.json")

# Change working dir to repo root (optional)
os.chdir(REPO_ROOT)

# ------------------- Imports -------------------
from configurations.config import Config
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

# ------------------- Config -------------------
cfg = Config(CONFIG_PATH)

# ------------------- Device -------------------
device = cfg.device
n_gpus = torch.cuda.device_count() if device == "cuda" else 0
logger.info(f"Device: {device}, GPUs available: {n_gpus}")

# ------------------- Model & Tokenizer -------------------
model, tokenizer = load_model_and_tokenizer(cfg.model_name, device=device)

# ------------------- Dataset -------------------
TRAIN_FILE = os.path.join(REPO_ROOT, cfg.train_file)
EVAL_FILE = os.path.join(REPO_ROOT, cfg.eval_file)

train_dataset, eval_dataset = create_dataset(TRAIN_FILE, EVAL_FILE)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=cfg.max_length)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=cfg.max_length)

# ------------------- LoRA & Training -------------------
lora_config = get_lora_config(cfg)
training_args = get_training_args(cfg)
trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

# ------------------- Training -------------------
logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# ------------------- Save -------------------
logger.info("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(cfg.training.get("output_dir"))
tokenizer.save_pretrained(cfg.training.get("output_dir"))
logger.info(f"All artifacts saved to {cfg.training.get('output_dir')}")
