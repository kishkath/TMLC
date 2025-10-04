from core.dataset_preparation import prepare_dataset
from core.model_loader import load_model, load_reference_model
from core.trainer import create_dpo_trainer, train_and_save
from configurations.config import TRAIN_FILE, VAL_FILE, TRAIN_LIMIT, TEST_LIMIT, TRAINER_CONFIG

if __name__ == "__main__":
    dataset = prepare_dataset(TRAIN_FILE, VAL_FILE, TRAIN_LIMIT, TEST_LIMIT)
    print(f"✅ Train size: {len(dataset['train'])}")
    print(f"✅ Test size: {len(dataset['test'])}")
    print("✅ Example record:\n", dataset["train"][0])

    model, tokenizer = load_model()
    ref_model = load_reference_model()
    print("✅ Models loaded successfully")

    trainer = create_dpo_trainer(model, ref_model, tokenizer, dataset, TRAINER_CONFIG)
    print("✅ Trainer created successfully")

    train_and_save(trainer, model, tokenizer, TRAINER_CONFIG, save_path="small-qwen-exp")
