import json
import logging
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

# Dataset
TRAIN_FILE = _config.get("train_file")
VAL_FILE = _config.get("val_file")
TRAIN_LIMIT = _config.get("train_limit", 700)
TEST_LIMIT = _config.get("test_limit", 100)

# Model
MODEL_CONFIG = _config.get("model", {})
MODEL_NAME = MODEL_CONFIG.get("name")
MAX_SEQ_LENGTH = MODEL_CONFIG.get("max_seq_length", 512)
DTYPE = MODEL_CONFIG.get("dtype", None)
LOAD_IN_4BIT = MODEL_CONFIG.get("load_in_4bit", True)
LORA_CONFIG = MODEL_CONFIG.get("lora", {})

# Trainer
TRAINER_CONFIG = _config.get("trainer", {})

# Inference
INFERENCE_CONFIG = _config.get("inference", {})
INFERENCE_MODEL_PATH = INFERENCE_CONFIG.get("model_path", "small-qwen-exp")
INFERENCE_MAX_SEQ_LENGTH = INFERENCE_CONFIG.get("max_seq_length", 512)
INFERENCE_LOAD_IN_4BIT = INFERENCE_CONFIG.get("load_in_4bit", True)
INFERENCE_TEMPERATURE = INFERENCE_CONFIG.get("temperature", 0.7)
INFERENCE_TOP_P = INFERENCE_CONFIG.get("top_p", 0.9)
INFERENCE_DO_SAMPLE = INFERENCE_CONFIG.get("do_sample", True)
INFERENCE_MAX_NEW_TOKENS = INFERENCE_CONFIG.get("max_new_tokens", 512)
INFERENCE_ADAPTIVE_GENERATION = INFERENCE_CONFIG.get("adaptive_generation", False)
INFERENCE_ADAPTIVE_THRESHOLD_VRAM_GB = INFERENCE_CONFIG.get("adaptive_threshold_vram_gb", 10)
INFERENCE_SYSTEM_PROMPT = INFERENCE_CONFIG.get("system_prompt", "")

# Wandb
WANDB_CONFIG = _config.get("wandb", {})
USE_WANDB = WANDB_CONFIG.get("use", False)
WANDB_PROJECT = WANDB_CONFIG.get("project", "fitness-qa-bot")
WANDB_NAME = WANDB_CONFIG.get("name", "qwen-finetuning")

# API
API_CONFIG = _config.get("api", {})

# Logging
LOGGING_CONFIG = _config.get("logging", {})
LOG_LEVEL = LOGGING_CONFIG.get("level", "INFO")

# Logging setup
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)
