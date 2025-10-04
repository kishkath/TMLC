from datasets import load_dataset, DatasetDict

def clean_record(example):
    example["prompt"] = str(example.get("question", "")).strip()
    example["chosen"] = str(example.get("chosen", "")).strip()
    example["rejected"] = str(example.get("rejected", "")).strip()
    return example

def prepare_dataset(train_file, val_file, train_limit=700, test_limit=100):
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "val": val_file}
    )
    dataset = DatasetDict({"train": dataset["train"], "test": dataset["val"]})
    dataset = dataset.map(clean_record)
    dataset = dataset.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )
    dataset["train"] = dataset["train"].select(range(min(train_limit, len(dataset["train"]))))
    dataset["test"] = dataset["test"].select(range(min(test_limit, len(dataset["test"]))))
    return dataset
