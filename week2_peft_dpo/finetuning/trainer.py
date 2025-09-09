from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def get_lora_config(cfg):
    lora_cfg = cfg.lora
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 64),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM")
    )

def get_training_args(cfg):
    tr_cfg = cfg.training
    return TrainingArguments(
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=tr_cfg.get("learning_rate", 5e-5),
        warmup_steps=tr_cfg.get("warmup_steps", 40),
        num_train_epochs=tr_cfg.get("num_train_epochs", 4),
        logging_steps=tr_cfg.get("logging_steps", 10),
        save_strategy=tr_cfg.get("save_strategy", "steps"),
        save_steps=tr_cfg.get("save_steps", 50),
        save_total_limit=tr_cfg.get("save_total_limit", 2),
        output_dir=tr_cfg.get("output_dir", "./output"),
        logging_dir=tr_cfg.get("logging_dir", "./logs"),
        report_to=tr_cfg.get("report_to", "none")
    )

def get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args):
    return SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        peft_config=lora_config,
        args=training_args
    )
