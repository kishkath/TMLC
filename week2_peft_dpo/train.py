import logging
import sys
import torch
from pathlib import Path
import os
from rich.table import Table
from rich.console import Console
from rich.live import Live

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
from transformers import TrainerCallback, TrainerState, TrainerControl

# ------------------- Logging -------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------- Helper function -------------------
def log_print(message):
    logger.info(message)
    print(message)

# ------------------- Model & Tokenizer -------------------
log_print(f"Loading model and tokenizer: {MODEL_NAME}")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=DEVICE)
log_print("Model and tokenizer loaded.")

# ------------------- Dataset -------------------
log_print(f"Loading datasets: train={TRAIN_FILE}, eval={EVAL_FILE}")
train_dataset, eval_dataset = create_dataset(TRAIN_FILE, EVAL_FILE)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=MAX_LENGTH)
log_print("Datasets tokenized.")

# ------------------- LoRA & Training -------------------
log_print("Creating LoRA config and training arguments...")
lora_config = get_lora_config(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type=LORA_TASK_TYPE
)
training_args = get_training_args(TRAINING_ARGS)
log_print("LoRA config and training arguments ready.")

trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)
log_print("Trainer initialized.")

# ------------------- Rich Table Callback -------------------
console = Console()
table = Table()
for col in ["Step", "Loss", "LR"]:
    table.add_column(col)
rows = []

class LiveTableCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = logs or {}
        step = str(state.global_step)
        loss = f"{logs.get('loss', 0.0):.4f}"
        lr = f"{logs.get('learning_rate', 0.0):.6f}"
        rows.append([step, loss, lr])
        table.rows = []
        for r in rows[-20:]:  # show last 20 steps
            table.add_row(*r)
        live.update(table)

live = Live(table, console=console, refresh_per_second=2)
trainer.add_callback(LiveTableCallback())

# ------------------- Training -------------------
log_print("Starting training...")
with live:
    trainer.train()
log_print("Training completed.")

# ------------------- Save -------------------
log_print("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(TRAINING_ARGS.get("output_dir"))
tokenizer.save_pretrained(TRAINING_ARGS.get("output_dir"))
log_print(f"All artifacts saved to {TRAINING_ARGS.get('output_dir')}")
