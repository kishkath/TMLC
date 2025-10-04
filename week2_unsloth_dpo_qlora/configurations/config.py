import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

# Dataset configuration
TRAIN_FILE = _config.get("train_file")
VAL_FILE = _config.get("val_file")
TRAIN_LIMIT = _config.get("train_limit", 700)
TEST_LIMIT = _config.get("test_limit", 100)

# Model configuration
MODEL_NAME = _config.get("model_name")
MAX_SEQ_LENGTH = _config.get("max_seq_length", 512)
DTYPE = _config.get("dtype", None)
LOAD_IN_4BIT = _config.get("load_in_4bit", True)
LORA_CONFIG = _config.get("lora", {})

# Trainer configuration
TRAINER_CONFIG = _config.get("trainer", {})

# Inference configuration
INFERENCE_CONFIG = _config.get("inference", {})
INFERENCE_MODEL_PATH = INFERENCE_CONFIG.get("model_path", "small-qwen-exp")
INFERENCE_MAX_SEQ_LENGTH = INFERENCE_CONFIG.get("max_seq_length", 512)
INFERENCE_LOAD_IN_4BIT = INFERENCE_CONFIG.get("load_in_4bit", True)

# wandb configuration
WANDB_CONFIG = _config.get("wandb", {})
USE_WANDB = WANDB_CONFIG.get("use", False)
WANDB_PROJECT = WANDB_CONFIG.get("project", "fitness-qa-bot")
WANDB_NAME = WANDB_CONFIG.get("name", "qwen-finetuning")
