import os, sys, random, json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, LogitsProcessor, LogitsProcessorList, pipeline 
from torch import amp

torch.cuda.empty_cache()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward, toxicity_score_regex, group_generalization_score, style_bonus, trivial_penalty, classifier_toxicity

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
# Define the quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
PROMPTS = "data/prompts/new_prompts.jsonl"
# PROMPTS = "data/prompts/pilot_prompts.jsonl"
OUTDIR = "results/grpo_pilot"

TOTAL_STEPS = 1000
PRINT_INTERVAL = 1
SAVE_INTERVAL = 100 # Save a checkpoint every 50 steps
K = 4
MAX_NEW_TOKENS = 128
FAST_MODE = False # skip LM toxicity for speed


# --- LogitsProcessor to clamp extreme logits ---
class SafeClampLogitsProcessor(LogitsProcessor):
    """Clamp logits and remove NaNs/Infs before softmax."""
    def __init__(self, min_val=-50.0, max_val=50.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Clamp extreme values
        scores = torch.clamp(scores, self.min_val, self.max_val)
        # Replace NaN/Inf with large negative values to avoid negative probabilities
        scores = torch.nan_to_num(scores, nan=self.min_val, posinf=self.max_val, neginf=self.min_val)
        return scores


# --- load prompts ---
def load_prompts(path, n=300):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    random.shuffle(rows)
    return rows[:n]


# --- build chat prompt ---
def build_chat(tokenizer, user):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )
    return text.strip()  # <--- remove trailing whitespace/newlines


def clean_response(text):
    return "".join(c for c in text if c.isprintable())

# --- sample multiple responses safely ---
def sample_responses_debug(model, tok, prompts, K, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_p=0.7):
    """
    Debug version: generates K responses per prompt, one at a time, with detailed logging.
    Safe for Qwen2 with gradient checkpointing.
    """
    all_responses_data = []

    for i, prompt in enumerate(prompts):
        # print(f"\n=== DEBUG PROMPT {i} ===")
        text = build_chat(tok, prompt)
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.device)

        # print("Chat text:\n", text)
        # print("Input IDs shape:", input_ids.shape)
        # print("Decoded input:", tok.decode(input_ids[0], skip_special_tokens=True))

        prompt_responses = []

        for k in range(K):
            # print(f"\n--- Generating sample {k+1}/{K} ---")
            try:
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=False  # critical with gradient checkpointing
                )
                resp = tok.decode(output[0], skip_special_tokens=True)
                resp_clean = clean_response(resp)
                prompt_responses.append(resp_clean)
                # print("Sample output:", resp_clean)
            except Exception as e:
                print("❌ Generation failed:", e)
                prompt_responses.append("")

        all_responses_data.append({
            "prompt": prompt,
            "responses": prompt_responses
        })

    return all_responses_data

def sample_responses_debug_fast(model, tok, prompts, K, max_new_tokens=MAX_NEW_TOKENS,
                                temperature=0.7, top_p=0.7):
    """
    Faster debug version of sample_responses_debug:
    - Generates K responses per prompt in a single batch
    - Uses mixed precision for speed
    - Safe for Qwen2 (gradient checkpointing can remain enabled)
    """
    all_responses_data = []

    for i, prompt in enumerate(prompts):
        text = build_chat(tok, prompt)
        input_ids = tok(text, return_tensors="pt").input_ids.to(model.device)

        # Repeat input_ids K times for batched generation
        input_ids_batch = input_ids.repeat(K, 1)

        # Use autocast for mixed precision to speed up
        with torch.autocast("cuda"):
            try:
                outputs = model.generate(
                    input_ids=input_ids_batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=False  # needed if gradient checkpointing is enabled
                )
            except Exception as e:
                print(f"❌ Generation failed for prompt {i}: {e}")
                outputs = [input_ids[0]] * K  # fallback empty responses

        # Decode outputs
        prompt_responses = []
        for o in outputs:
            resp = tok.decode(o, skip_special_tokens=True)
            resp_clean = clean_response(resp)
            prompt_responses.append(resp_clean)

        all_responses_data.append({
            "prompt": prompt,
            "responses": prompt_responses
        })

        # Debug prints for first few prompts
        # print(f"\n=== DEBUG PROMPT {i} ===")
        # print("Prompt:", prompt)
        # for k, resp in enumerate(prompt_responses):
            # print(f"Sample {k+1}/{K}: {resp[:200]}")  # only show first 200 chars

    return all_responses_data

# --- compute reward components ---
def reward_components(prompt, response, tox_model_score, fast_mode=False):
    tox_regex = toxicity_score_regex(response)
    gen = group_generalization_score(response)
    trivial = trivial_penalty(prompt, response)
    style = style_bonus(response)
    
    # DEBUG STEP
    print(f"DEBUG | tox_regex: {tox_regex}, gen: {gen}, style: {style}, trivial: {trivial}")

    reward = compute_bias_reward(prompt, response, fast_mode=fast_mode, tox_model_score=tox_model_score)
    return reward, tox_regex, gen, style, trivial


# --- main training loop ---
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, quantization_config=bnb_config).to(device)
    # # --- The Corrected Fix ---
    # # 1. Set the padding side for the tokenizer.
    # # For decoder-only models like Qwen2, left-padding is essential for correct batch generation.
    # tok.padding_side = "left"
    
    # # 2. Add a new pad token if the tokenizer doesn't have one.
    # # Many instruction-tuned models don't have a default pad token.
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '<|pad|>'})

    # # 3. Resize the model's token embeddings to accommodate the new token.
    model.resize_token_embeddings(len(tok))

    # # 4. CRITICAL: Update the model's configuration to use the new pad token ID.
    # # This is the step that resolves the CUDA error.
    model.config.pad_token_id = tok.pad_token_id
    
    print("Initializing toxicity classifier...")
    tox_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=-1, truncation=True)
    print("Initialization complete.")

    model.train()

    prompts = load_prompts(PROMPTS, n=300)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, TOTAL_STEPS)

    # DEBUG
    # for p in prompts[:1]:
    #     text = build_chat(tok, p["prompt"])
    #     input_ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    #     output = model.generate(input_ids, max_new_tokens=128)
    #     print(tok.decode(output[0], skip_special_tokens=True))

    step = 0
    batch_size = 2  # Increase batch size to fully leverage the GPU
    running_baseline = 0.05   # small, fixed start
    baseline_beta = 0.995    # VERY slow
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    while step < TOTAL_STEPS:
        batch_prompts_data = random.sample(prompts, batch_size)
        batch_prompts = [ ex["prompt"] for ex in batch_prompts_data]

        # --- 1. Batch generate responses for the entire batch ---
        all_responses_data = sample_responses_debug_fast(model, tok, batch_prompts, K, MAX_NEW_TOKENS)
        
        # --- 2. Flatten all prompts and responses to prepare for reward calculation ---
        flat_prompts = []
        flat_responses = []
        for ex_data in all_responses_data:
            flat_prompts.extend([ex_data["prompt"]] * len(ex_data["responses"]))
            flat_responses.extend([r.strip() for r in ex_data["responses"]])

            
        # --- 3. Batch compute rewards ---
        tox_model_scores = classifier_toxicity(flat_responses, tox_pipeline=tox_pipeline)
        
        # You can't easily batch the other reward functions unless you modify them
        # to accept lists. For now, we'll keep this loop but acknowledge it's a bottleneck.
        # ✅ Pass the pipeline as an argument
        
        all_rewards = []
        all_tox, all_gen, all_style, all_trivial = [], [], [], []
        
        for i, r in enumerate(flat_responses):
            # clean_r = flat_responses[i].replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            clean_r = clean_response(flat_responses[i])
            # DEBUG
            if i == 1:
                print("DEBUG | raw response:", clean_r)
            reward, tox, gen, style, trivial = reward_components(flat_prompts[i], clean_r, tox_model_scores[i], fast_mode=FAST_MODE)
            
            if step < 500:
                trivial *= 0.5

            all_rewards.append(reward)
            all_tox.append(tox)
            all_gen.append(gen)
            all_style.append(style)
            all_trivial.append(trivial)
        
        #debug
        if step < 20:
            print("DEBUG rewards:", all_rewards[:8])

        # --- 4. Compute advantages and create a single batched loss input ---
        reshaped_rewards = np.array(all_rewards).reshape(batch_size, K)
        
        # Contrastive Bias
        tox_matrix = np.array(all_tox).reshape(batch_size, K)
        best_tox = tox_matrix.min(axis=1, keepdims=True)
        contrastive_bias = tox_matrix - best_tox
        lambda_cb = 0.5
        reshaped_rewards -= lambda_cb * contrastive_bias

        #Debug
        if step < 5:
            print("DEBUG tox:", all_tox[:8])

        # Advantage computation
        advantages = np.zeros_like(reshaped_rewards)
        for i in range(batch_size):
            ranks = reshaped_rewards[i].argsort().argsort()
            advantages[i] = ranks - ranks.mean()

        advantages = advantages.flatten()
        advantages = advantages / (advantages.std() + 1e-8)

        # --- 5. Batch the REINFORCE loss calculation and optimizer step ---
        all_full_texts = [
            tok.apply_chat_template([
                {"role": "user", "content": flat_prompts[i]},
                {"role": "assistant", "content": flat_responses[i]}
            ], tokenize=False)
            for i in range(len(flat_prompts))
        ]
        
        enc = tok(all_full_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = enc.input_ids
        
        # with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     out = model(input_ids=input_ids)
        out = model(input_ids=input_ids)
            
        logits = out.logits  # [B*K, T, V]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Shift for causal LM
        shift_log_probs = log_probs[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]

        # Mask prompt tokens
        prompt_texts = [
            build_chat(tok, p) for p in flat_prompts
        ]
        prompt_enc = tok(prompt_texts, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc.attention_mask.sum(dim=1)

        attention_mask = torch.ones_like(shift_input_ids)
        for i in range(len(flat_prompts)):
            attention_mask[i, :prompt_lens[i]-1] = 0

        # Gather log-probs of generated tokens
        token_log_probs = shift_log_probs.gather(
            dim=-1,
            index=shift_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask
        token_log_probs = token_log_probs * attention_mask

        # Sequence log-prob 
        logliks = token_log_probs.sum(dim=1)
        
        advantages_tensor = torch.tensor(
            advantages,
            dtype=logliks.dtype,
            device=model.device
        ).detach()
        
        policy_loss = (-advantages_tensor * logliks).mean()

        # ================================
        # ✅ ENTROPY BONUS (ADD HERE)
        # ================================
        # entropy: [B*K, T]
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B*K, T]

        entropy_seq_len = entropy.shape[1]
        mask_seq_len = attention_mask.shape[1]

        if entropy_seq_len != mask_seq_len:
            # Truncate or pad the mask
            if mask_seq_len > entropy_seq_len:
                attention_mask = attention_mask[:, :entropy_seq_len]
            else:
                pad = torch.ones((attention_mask.shape[0], entropy_seq_len - mask_seq_len), device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad], dim=1)

        entropy = entropy * attention_mask
        entropy = entropy.sum() / attention_mask.sum().clamp(min=1)

        # print(f"entropy {entropy.item():.3f}")

        entropy_coef = 0.01  # start small
        final_loss = policy_loss - entropy_coef * entropy

        # final_loss = (-advantages_tensor * logliks).mean()
        
        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        
        step += 1

        # --- 6. Logging and saving (simplified) ---
        if step % PRINT_INTERVAL == 0:
            print(f"step {step} | loss {final_loss.item():.4f} | mean_r {np.mean(all_rewards):.3f} | baseline {np.mean(running_baseline):.3f}")
            print(f"      tox={np.mean(all_tox):.3f} gen={np.mean(all_gen):.3f} style={np.mean(all_style):.3f} trivial={np.mean(all_trivial):.3f} entropy {entropy.item():.3f}")

        if step % SAVE_INTERVAL == 0:
            checkpoint_dir = os.path.join(OUTDIR, "checkpoints", f"step_{step}")
            model.save_pretrained(checkpoint_dir)
            tok.save_pretrained(checkpoint_dir)
            print(f"✅ Saved checkpoint at step {step} to {checkpoint_dir}")

        del enc, input_ids, logits, log_probs
        del token_log_probs, logliks, advantages_tensor, final_loss
        torch.cuda.empty_cache()
    
    model.save_pretrained(os.path.join(OUTDIR, "policy"))
    tok.save_pretrained(os.path.join(OUTDIR, "policy"))
    print("✅ Saved fine-tuned policy.")


if __name__ == "__main__":
    main()
