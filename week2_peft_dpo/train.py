import logging
import sys
import torch
from pathlib import Path
import os
from tqdm.auto import tqdm

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
# Ensure tqdm works
training_args.remove_unused_columns = False
training_args.report_to = "none"
training_args.logging_steps = TRAINING_ARGS.get("logging_steps", 1)
training_args.progress_bar_refresh_rate = 1

log_print("LoRA config and training arguments ready.")

trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)
log_print("Trainer initialized.")

# ------------------- Training -------------------
log_print("Starting training...")

# Wrap train_dataset in tqdm for batch-level progress bar
num_train_steps = len(tokenized_train_dataset) // training_args.per_device_train_batch_size
pbar = tqdm(range(num_train_steps), desc="Training Progress", unit="step", ncols=100)

for _ in trainer.train():
    pbar.update(1)
    if trainer.state.global_step % training_args.logging_steps == 0:
        if trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            current_loss = last_log.get("loss", last_log.get("eval_loss", None))
        else:
            current_loss = None
        log_print(f"Step {trainer.state.global_step}/{num_train_steps} - Loss: {current_loss}")

pbar.close()
log_print("Training completed.")

# ------------------- Save -------------------
log_print("Saving LoRA adapter & tokenizer...")
trainer.model.save_pretrained(TRAINING_ARGS.get("output_dir"))
tokenizer.save_pretrained(TRAINING_ARGS.get("output_dir"))
log_print(f"All artifacts saved to {TRAINING_ARGS.get('output_dir')}")

