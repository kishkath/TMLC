import torch
from configurations.config import INFERENCE_CONFIG, logger


def get_adaptive_max_tokens():
    if INFERENCE_CONFIG.get("adaptive_generation", False):
        try:
            if torch.cuda.is_available():
                vram_free = torch.cuda.mem_get_info()[0] / 1e9
                if vram_free < INFERENCE_CONFIG.get("adaptive_threshold_vram_gb", 10):
                    return max(512, int(INFERENCE_CONFIG.get("max_new_tokens", 512) / 2))
        except Exception as e:
            logger.warning(f"Adaptive token adjustment failed: {e}")
    return INFERENCE_CONFIG.get("max_new_tokens", 512)


def predict(model, tokenizer, input_prompt, system_prompt=None):
    system_prompt = system_prompt or INFERENCE_CONFIG.get("system_prompt", "You are a helpful assistant.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        max_new_tokens = get_adaptive_max_tokens()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=INFERENCE_CONFIG.get("temperature", 0.7),
            top_p=INFERENCE_CONFIG.get("top_p", 0.9),
            do_sample=INFERENCE_CONFIG.get("do_sample", True)
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
