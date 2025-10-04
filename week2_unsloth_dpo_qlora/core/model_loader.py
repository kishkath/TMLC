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

from unsloth import FastLanguageModel
from peft import LoraConfig

def load_model_with_lora(config):
    """
    Loads Qwen3-0.6B with Unsloth + QLoRA + DPO support.
    """
    # Extract parameters
    model_name = config["model_name"]
    max_seq_length = config.get("max_seq_length", 512)
    dtype = config.get("dtype", None)
    load_in_4bit = config.get("load_in_4bit", True)

    # Load base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # LoRA Configuration
    lora_params = config["lora"]
    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        target_modules=lora_params["target_modules"],
        task_type="CAUSAL_LM",
        use_rslora=lora_params.get("use_rslora", True)
    )

    # Apply LoRA adapters to model
    model = FastLanguageModel.get_peft_model(model, lora_config)

    print(f"✅ LoRA adapters successfully attached to {model_name}")
    return model, tokenizer

