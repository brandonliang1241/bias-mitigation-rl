import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

# -------------------------
# Bits & Bytes config (optional, if using 4-bit quant)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# -------------------------
# Batched response generation
# -------------------------
def batch_generate_responses(
    model,
    tokenizer,
    prompts,
    device="cuda",
    batch_size=8,
    max_new_tokens=128,
    greedy=True
):
    """
    Generates responses from a causal LM in batches.
    Returns a list of strings (responses).
    """
    model.eval()
    out = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            texts = []

            # Apply chat template if tokenizer supports it
            for prompt_text in batch_prompts:
                try:
                    chat_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    chat_text = prompt_text
                texts.append(chat_text)

            # Tokenize
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Generate
            if greedy:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9
                )

            # Decode outputs
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j, full_text in enumerate(decoded):
                # Extract the model's response only
                response = full_text.split(batch_prompts[j])[-1].strip()
                out.append(response)

    return out
