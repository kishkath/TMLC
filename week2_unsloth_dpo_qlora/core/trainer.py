from trl import DPOTrainer, DPOConfig
import wandb
import torch
from configurations.config import USE_WANDB, WANDB_PROJECT, WANDB_NAME

def create_dpo_trainer(model, ref_model, tokenizer, dataset, trainer_config):
    dpo_args = DPOConfig(
        per_device_train_batch_size=trainer_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=trainer_config.get("gradient_accumulation_steps", 4),
        warmup_ratio=trainer_config.get("warmup_ratio", 0.1),
        num_train_epochs=trainer_config.get("num_train_epochs", 3),
        learning_rate=trainer_config.get("learning_rate", 5e-6),
        logging_steps=trainer_config.get("logging_steps", 10),
        optim=trainer_config.get("optim", "adamw_8bit"),
        weight_decay=trainer_config.get("weight_decay", 0.0),
        lr_scheduler_type=trainer_config.get("lr_scheduler_type", "linear"),
        seed=trainer_config.get("seed", 42),
        output_dir=trainer_config.get("output_dir", "outputs"),
        report_to="wandb" if USE_WANDB else "none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        beta=trainer_config.get("beta", 0.1),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_length=trainer_config.get("max_length", 512),
        max_prompt_length=trainer_config.get("max_prompt_length", 128),
    )

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, config=trainer_config, name=WANDB_NAME)

    return trainer

def train_and_save(trainer, model, tokenizer, trainer_config, save_path="small-qwen-exp"):
    print("🚀 Starting training...")
    trainer.train()

    if USE_WANDB:
        wandb.finish()

    print(f"💾 Saving model to {save_path}...")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("✅ Model saved successfully.")
