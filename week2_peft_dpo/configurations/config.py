import json
import torch

class Config:
    """
    Loads training configuration from a JSON file.
    All parameters (device, model, data paths, LoRA, training args)
    can be modified in config.json without changing the code.
    """
    def __init__(self, path="week2_peft_dop/configurations/config.json"):
        with open(path, "r") as f:
            cfg = json.load(f)

        # Device
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Model & Data
        self.model_name = cfg.get("model_name")
        self.train_file = cfg.get("train_file")
        self.eval_file = cfg.get("eval_file")
        self.max_length = cfg.get("max_length", 512)

        # LoRA configuration
        self.lora = cfg.get("lora", {})

        # Training arguments
        self.training = cfg.get("training", {})

