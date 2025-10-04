from unsloth import FastLanguageModel
from configurations.config import MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT, LORA_CONFIG

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG.get("r", 16),
        target_modules=LORA_CONFIG.get("target_modules", []),
        lora_alpha=LORA_CONFIG.get("lora_alpha", 64),
        lora_dropout=LORA_CONFIG.get("lora_dropout", 0),
        bias=LORA_CONFIG.get("bias", "none"),
        use_gradient_checkpointing=LORA_CONFIG.get("use_gradient_checkpointing", None),
        random_state=LORA_CONFIG.get("random_state", None),
        use_rslora=LORA_CONFIG.get("use_rslora", False),
        loftq_config=LORA_CONFIG.get("loftq_config", None),
    )

    return model, tokenizer

def load_reference_model():
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return ref_model

def load_finetuned_model(model_path, max_seq_length=512, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer
