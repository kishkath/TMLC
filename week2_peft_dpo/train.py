import logging
import sys
import torch
from configurations.config import Config
from dataset.data_utils import create_dataset, tokenize_dataset
from finetuning.model import load_model_and_tokenizer
from finetuning.trainer import get_lora_config, get_training_args, get_trainer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

cfg = Config("config.json")
device = cfg.device
n_gpus = torch.cuda.device_count() if device == "cuda" else 0
logger.info(f"Device: {device}, GPUs available: {n_gpus}")

model, tokenizer = load_model_and_tokenizer(cfg.model_name, device=device)

train_dataset, eval_dataset = create_dataset(cfg.train_file, cfg.eval_file)
tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=cfg.max_length)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=cfg.max_length)

lora_config = get_lora_config(cfg)
training_args = get_training_args(cfg)
trainer = get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args)

logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# Save LoRA adapter & tokenizer
logger.info("Saving LoRA adapter & tokenizer...")
if n_gpus > 1:
    trainer.model.module.save_pretrained(cfg.training.get("output_dir"))
else:
    trainer.model.save_pretrained(cfg.training.get("output_dir"))
tokenizer.save_pretrained(cfg.training.get("output_dir"))
logger.info(f"All artifacts saved to {cfg.training.get('output_dir')}")
