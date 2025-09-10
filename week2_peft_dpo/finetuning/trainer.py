from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def get_lora_config(cfg):
    """
    Create LoRA configuration for the model.
    Expects cfg.lora to be a dictionary with keys:
    r, lora_alpha, target_modules, lora_dropout, bias, task_type
    """
    lora_cfg = cfg.lora
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM")
    )


def get_training_args(cfg):
    """
    Create HuggingFace TrainingArguments from cfg.training dictionary.
    """
    tr_cfg = cfg.training
    return TrainingArguments(
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=tr_cfg.get("learning_rate", 5e-5),
        warmup_steps=tr_cfg.get("warmup_steps", 40),
        num_train_epochs=tr_cfg.get("num_train_epochs", 4),
        logging_steps=tr_cfg.get("logging_steps", 10),
        evaluation_strategy=tr_cfg.get("evaluation_strategy", "no"),
        eval_steps=tr_cfg.get("eval_steps", None),
        save_strategy=tr_cfg.get("save_strategy", "steps"),
        save_steps=tr_cfg.get("save_steps", 50),
        save_total_limit=tr_cfg.get("save_total_limit", 2),
        output_dir=tr_cfg.get("output_dir", "./output"),
        logging_dir=tr_cfg.get("logging_dir", "./logs"),
        load_best_model_at_end=tr_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=tr_cfg.get("metric_for_best_model", None),
        report_to=tr_cfg.get("report_to", "none")
    )


def get_trainer(model, tokenized_train_dataset, tokenized_eval_dataset, lora_config, training_args):
    """
    Create an SFTTrainer instance for fine-tuning with LoRA.
    """
    return SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        peft_config=lora_config,
        args=training_args
    )
