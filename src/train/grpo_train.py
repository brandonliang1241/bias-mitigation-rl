import os, sys, random, json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, get_linear_schedule_with_warmup, LogitsProcessor, LogitsProcessorList, pipeline, BitsAndBytesConfig
from torch import amp, nn
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward, toxicity_score_regex, group_generalization_score, style_bonus, trivial_penalty, compute_grpo_reward

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

TOTAL_STEPS = 2000
PRINT_INTERVAL = 5
SAVE_INTERVAL = 100 # Save a checkpoint every 100 steps
DEBUG_INTERVAL = 50 # Print and check model output and reward scores
K = 4
MAX_NEW_TOKENS = 64 # Low to speed up training
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
def sample_responses_debug(model, tok, prompts, K, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_p=0.85):
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


def strip_assistant_prefix(text: str) -> str:
    text = text.lstrip()
    for prefix in ["assistant", "Assistant", "?assistant", ".assistant"]:
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text


def sample_responses_debug_fast(
    model,
    tok,
    prompt_encodings,
    K,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.7,
):

    all_responses_data = []

    input_ids = prompt_encodings["input_ids"]
    attention_mask = prompt_encodings["attention_mask"]

    B = input_ids.shape[0]

    # Repeat prompts K times
    input_ids_batch = input_ids.repeat_interleave(K, dim=0)
    attention_mask_batch = attention_mask.repeat_interleave(K, dim=0)

    prompt_lengths = attention_mask_batch.sum(dim=1)

    with torch.autocast("cuda"):
        outputs = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=False,
        )

    # slice generated tokens only
    generated_tokens = []
    for i in range(outputs.size(0)):
        gen = outputs[i, prompt_lengths[i]:]
        generated_tokens.append(gen)

    # Decode responses
    decoded_responses = tok.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Repack per-prompt
    idx = 0
    for i in range(B):
        responses = []
        for _ in range(K):
            resp = clean_response(decoded_responses[idx])
            resp = strip_assistant_prefix(resp)
            responses.append(resp)
            idx += 1

        prompt_text = tok.decode(
            input_ids[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        all_responses_data.append({
            "prompt": prompt_text,
            "responses": responses
        })

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


# reward/deberta_reward.py
MODEL_DIR = "DeBERTaV3/ToxiGen/deberta_bias_scorer"
MAX_LEN = 128

class DebertaPairwiseReward(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 2)  # <-- change from 1 to 2

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.scorer(cls)  # shape [B,2]


def load_deberta_reward(device):
    """
    Load the DeBERTa pairwise reward model from checkpoint (.safetensors or .bin).
    Returns frozen model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = DebertaPairwiseReward("microsoft/deberta-v3-small")

    # Load checkpoint
    checkpoint_path_bin = f"{MODEL_DIR}/pytorch_model.bin"
    checkpoint_path_safetensors = f"{MODEL_DIR}/model.safetensors"

    if torch.cuda.is_available():
        map_location = device
    else:
        map_location = 'cpu'

    try:
        # Try safetensors first
        state = load_file(checkpoint_path_safetensors, device=map_location)
        model.load_state_dict(state)
    except Exception as e:
        print(f"❌ Failed to load safetensors: {e}, trying .bin")
        state = torch.load(checkpoint_path_bin, map_location=map_location)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer


def deberta_score_batch(model, tokenizer, texts, device, max_len=MAX_LEN):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])  # [B,2]

    # Convert pairwise logits → scalar bias score
    bias_score = logits[:, 1] - logits[:, 0]   # [B]

    return bias_score.cpu()


def debug_step(step, flat_prompts, flat_responses, advantages, deberta_model, deberta_tokenizer, device):
    """
    Debug printing for GRPO training, ensures each response matches its prompt.
    
    Prints the prompt, response, computed GRPO reward, and advantage.
    """
    print(f"\n🛠️  DEBUG STEP {step} 🛠️")

    # Compute DeBERTa bias scores for all responses
    deberta_scores = deberta_score_batch(deberta_model, deberta_tokenizer, flat_responses, device=device)

    for i, (prompt, resp, adv, bias_score) in enumerate(zip(flat_prompts, flat_responses, advantages, deberta_scores)):
        reward = compute_grpo_reward(prompt, resp, bias_score)  # Use actual GRPO scalar reward
        print(f"\nPrompt [{i}]: {prompt[:300]}...")  # truncate long prompts
        print(f"  ✅ Response: {resp[:300]}... | Reward: {reward:.4f} | Advantage: {adv.item():.4f}")

    # Batch-level stats
    all_rewards = [compute_grpo_reward(p, r, s) for p, r, s in zip(flat_prompts, flat_responses, deberta_scores)]
    batch_mean_reward = np.mean(all_rewards)
    batch_adv_std = advantages.std().item()
    print(f"\nBatch Mean Reward: {batch_mean_reward:.4f} | Advantage Std: {batch_adv_std:.4f}\n")

    return batch_mean_reward, batch_adv_std

# --- plotting code ---
import matplotlib.pyplot as plt

# Store metrics for plotting
training_log = {
    "steps": [],
    "mean_reward": [],
    "adv_std": []
}

def log_training_metrics(step, mean_reward, adv_std):
    training_log["steps"].append(step)
    training_log["mean_reward"].append(mean_reward)
    training_log["adv_std"].append(adv_std)


def plot_training_metrics():
    plt.figure(figsize=(10,5))
    
    # Mean reward
    plt.plot(training_log["steps"], training_log["mean_reward"], label="Mean Reward", marker='o')
    
    # Advantage std
    plt.plot(training_log["steps"], training_log["adv_std"], label="Advantage Std", marker='x')
    
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title("GRPO Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

import textstat
import json
LOG_FILE = os.path.join(OUTDIR, "training_log.jsonl")

def log_metrics(step, prompt, response, components, kl_value=None, deberta_score=None):
    log = {
        "step": step,
        "prompt": prompt,
        "response": response,
        "bias_reward": components["bias_reward"],
        "style_reward": components["style_reward"],
        "length_reward": components["length_reward"],
        "repeat_penalty": components["repeat_penalty"],
        "repetition_gate": components["repetition_gate"],
        "combined_reward": components["combined_reward"],
        "kl": kl_value,
        "mean_deberta_score": deberta_score,
        "response_length": len(response.split()),
        "readability": textstat.flesch_reading_ease(response)
    }

    # Append to file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
# -------------------------

# --- main training loop ---
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DeBERTa once, frozen
    deberta_model, deberta_tokenizer = load_deberta_reward(device)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        trust_remote_code=True, 
        quantization_config=bnb_config
    ).to(device)

    # =====================
    # Apply LoRA to policy model
    # =====================

    # Freeze base model weights
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=16,                    # good starting point
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # correct for Qwen2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Optional but VERY useful
    model.print_trainable_parameters()

    # ref model for comparison 
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        quantization_config=bnb_config
    ).to(device)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Ensure pad token is set properly
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': tok.eos_token})
    tok.pad_token = tok.eos_token
    model.resize_token_embeddings(len(tok))
    ref_model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    ref_model.config.pad_token_id = tok.pad_token_id

    model.train()
    # --- Load and pretokenize prompts ---
    prompts_data = load_prompts(PROMPTS, n=300)
    prompt_texts = [build_chat(tok, ex["prompt"]) for ex in prompts_data]

    # Tokenize ALL prompts ONCE and move to GPU
    prompt_encodings = tok(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # optional
    ).to(model.device)

    prompts = load_prompts(PROMPTS, n=300)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,   # LoRA needs MUCH higher LR
        betas=(0.9, 0.999),
        eps=1e-6
    )

    # DEBUG
    # for p in prompts[:1]:
    #     text = build_chat(tok, p["prompt"])
    #     input_ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    #     output = model.generate(input_ids, max_new_tokens=128)
    #     print(tok.decode(output[0], skip_special_tokens=True))

    step = 0
    batch_size = 4  # Increase batch size to fully leverage the GPU

    model.config.use_cache = False
    import time

    while step < TOTAL_STEPS:
        t0 = time.time()
        # 1️⃣ Sample batch indices instead of Python objects
        batch_indices = np.random.choice(len(prompts), batch_size, replace=False)
        batch_encodings = {
            "input_ids": prompt_encodings.input_ids[batch_indices],
            "attention_mask": prompt_encodings.attention_mask[batch_indices]
        }

        # --- 2️⃣ Batch generate responses using optimized function ---
        all_responses_data = sample_responses_debug_fast(model, tok, batch_encodings, K, MAX_NEW_TOKENS)

        # --- 3️⃣ Flatten all prompts and responses for reward calculation ---
        flat_prompts = []
        flat_responses = []
        
        for ex_data in all_responses_data:
            prompt_text = ex_data["prompt"].strip()
            for r in ex_data["responses"]:
                flat_prompts.append(prompt_text)
                flat_responses.append(r)   # ← THIS is the missing line

        t1 = time.time()
            
        # --- 3. Batch compute rewards ---
        # --- 3.1: Batch DeBERTa scores for all responses ---
        deberta_scores = deberta_score_batch(
            deberta_model,
            deberta_tokenizer,
            flat_responses,
            device=device,
            max_len=MAX_LEN
        )
        # ---------- DEBUGGING -----------
        # responses_per_prompt = [
        #     ex_data["responses"]
        #     for ex_data in all_responses_data
        # ]  # shape: [B][K]
        # rep_responses = [
        #     clean_response(responses[0])
        #     for responses in responses_per_prompt
        # ]
        # print("DEBUG DeBERTa inputs:", len(rep_responses))
        # rep_deberta_scores = deberta_score_batch(
        #     deberta_model,
        #     deberta_tokenizer,
        #     rep_responses,
        #     device=device,
        #     max_len=MAX_LEN
        # )  # shape: [B]
        # # Repeat each prompt's score K times
        # deberta_scores = torch.repeat_interleave(
        #     rep_deberta_scores,
        #     repeats=K
        # )
        # ----------------------
        
        t2 = time.time()
        # --- 3.2: Compute final rewards ---
        all_rewards = [
            compute_grpo_reward(p, r, s)
            for p, r, s in zip(flat_prompts, flat_responses, deberta_scores)
        ]

        #debug
        if step < 20:
            print("DEBUG rewards:", all_rewards[:8])

        # --- 4. Compute advantages and create a single batched loss input ---
        reshaped_rewards = np.array(all_rewards).reshape(batch_size, K)
        
        # Convert to torch tensor on GPU
        rewards = torch.tensor(reshaped_rewards, device=model.device, dtype=torch.float32)

        with torch.no_grad():
            reward_std_per_prompt = rewards.std(dim=1).mean().item()
            reward_range = (rewards.max() - rewards.min()).item()
        print(
            f"reward_std/prompt {reward_std_per_prompt:.4f} | "
            f"reward_range {reward_range:.3f}"
        )

        group_mean = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - group_mean
        group_std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = advantages / group_std

        advantages = advantages.view(-1)                  # [B*K]
        print("adv per prompt:\n", advantages.view(batch_size, K))
        print(f"adv_mean {advantages.mean().item():.3f} | adv_std {advantages.std().item():.3f}")

        # --- 5. Batch the REINFORCE loss calculation and optimizer step ---
        # ----------------------------
        # Vectorized tokenization (GPU-friendly)
        # ----------------------------
        MAX_SEQ_LEN = min(512, model.config.max_position_embeddings)
        assistant_max_len = MAX_NEW_TOKENS

        # 1. Tokenize user prompts in batch
        prompt_enc = {
            "input_ids": batch_encodings["input_ids"].repeat_interleave(K, dim=0),
            "attention_mask": batch_encodings["attention_mask"].repeat_interleave(K, dim=0),
        }
        prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)

        # 2. Tokenize assistant responses in batch
        response_enc = tok(
            flat_responses,
            add_special_tokens=False,
            truncation=True,
            max_length=assistant_max_len,
            padding="max_length",  # <-- pad responses to max length in batch
            return_tensors="pt"
        ).to(model.device)

        # 3. Combine prompt + response, truncate to MAX_SEQ_LEN
        combined_input_ids = torch.cat([prompt_enc["input_ids"], response_enc["input_ids"]], dim=1)
        combined_input_ids = combined_input_ids[:, :MAX_SEQ_LEN]

        prompt_lengths = torch.clamp(prompt_lengths, max=combined_input_ids.size(1)-1)

        # 4. Attention mask
        attention_mask = (combined_input_ids != tok.pad_token_id).long()

        # ----------------------------
        # --- 6. Forward + REINFORCE ---
        # ----------------------------

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_ids=combined_input_ids, attention_mask=attention_mask)
            logits = out.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            ref_out = ref_model(
                input_ids=combined_input_ids,
                attention_mask=attention_mask
            )
            ref_log_probs = torch.log_softmax(ref_out.logits, dim=-1)

        # shift for causal LM
        shift_log_probs = log_probs[:, :-1, :]
        shift_ref_log_probs = ref_log_probs[:, :-1, :]
        shift_input_ids = combined_input_ids[:, 1:]
        shift_attention_mask = attention_mask[:, 1:]

        # generated token mask
        gen_mask = torch.zeros_like(shift_attention_mask)
        for i, plen in enumerate(prompt_lengths):
            if plen < combined_input_ids.size(1):
                gen_mask[i, plen:] = 1  # start right after prompt
        gen_mask = gen_mask * shift_attention_mask  # mask out padding
        # print("gen_mask first row:", gen_mask[0].cpu().numpy())
        # print("prompt_length:", prompt_lengths[0])
        # print(f"combined_input_ids: {combined_input_ids.shape}, shift_input_ids: {shift_input_ids.shape}, gen_mask: {gen_mask.shape}"

        # gather log-probs of generated tokens
        token_log_probs = shift_log_probs.gather(
            dim=-1,
            index=shift_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # mask prompt tokens
        # token_log_probs = token_log_probs * gen_mask 
        
        # token-wise KL on sampled actions
        # 1. compute token-wise KL over vocab
        kl_tokens = shift_log_probs.exp() * (shift_log_probs - shift_ref_log_probs)  # [B, T, V]
        kl_per_token = kl_tokens.sum(dim=-1)  # [B, T]

        # 2. mask prompt tokens
        kl_masked = kl_per_token * gen_mask  # [B, T]

        # 3. average
        kl = kl_masked.sum() / gen_mask.sum().clamp(min=1)
        print(f"KL {kl.item():.4f}")
        if kl.item() > 0.3: # hard skip if kl exceeds
            print("⚠️ KL too large, skipping optimizer step")
            optimizer.zero_grad()
            del combined_input_ids, attention_mask, log_probs, logits
            del shift_log_probs, shift_input_ids, shift_attention_mask, token_log_probs
            torch.cuda.empty_cache()
            continue

        # sequence log-likelihoods
        token_counts = gen_mask.sum(dim=1).clamp(min=1)
        logliks = (token_log_probs * gen_mask).sum(dim=1) / token_counts

        # REINFORCE loss
        advantages_tensor = advantages.detach().clamp(-1.0, 1.0) #added clamp
        policy_loss = (-advantages_tensor * logliks).mean()

        # entropy bonus
        entropy = -(shift_log_probs.exp() * shift_log_probs).sum(dim=-1)
        entropy = (entropy * gen_mask).sum() / gen_mask.sum().clamp(min=1)

        entropy_coef = 0.005

        # kl controller
        TARGET_KL = 0.1
        if kl.item() > TARGET_KL * 2:
            kl_coef = 0.2
        elif kl.item() > TARGET_KL:
            kl_coef = 0.1
        else:
            kl_coef = 0.02

        final_loss = policy_loss - entropy_coef * entropy + kl_coef * kl

        # backward + step
        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t3 = time.time()
        step += 1

        # --- 6. Logging and saving (simplified) ---
        # Assume flat_prompts, flat_responses, deberta_scores are ready
        for prompt, response, bias_score in zip(flat_prompts, flat_responses, deberta_scores):
            combined_reward, components = compute_grpo_reward(prompt, response, bias_score, return_components=True)

            # Optionally, KL per batch (use per sample if you have)
            kl_value = kl.item()  # or per-response if you compute it individually

            log_metrics(
                step=step,
                prompt=prompt,
                response=response,
                components=components,
                kl_value=kl_value,
                deberta_score=bias_score.item() if isinstance(bias_score, torch.Tensor) else bias_score
            )
        
        if step % PRINT_INTERVAL == 0:
            print(f"step {step} | loss {final_loss.item():.4f} | mean_r {np.mean(all_rewards):.3f} | mean_r_scaled {rewards.mean().item():.3f}")
            print(f"entropy {entropy.item():.3f}")

        if step % SAVE_INTERVAL == 0:
            checkpoint_dir = os.path.join(OUTDIR, "checkpoints", f"step_{step}")
            model.save_pretrained(checkpoint_dir)
            tok.save_pretrained(checkpoint_dir)
            print(f"✅ Saved checkpoint at step {step} to {checkpoint_dir}")
        
        if step % DEBUG_INTERVAL == 0 or step == 1:
            # Call debug function
            batch_mean_r, batch_adv_std = debug_step(
                step,
                flat_prompts=flat_prompts,       # flattened, matches flat_responses
                flat_responses=flat_responses,
                advantages=advantages.detach(),
                deberta_model=deberta_model,
                deberta_tokenizer=deberta_tokenizer,
                device=device
            )

            # Optionally track metrics for plotting
            log_training_metrics(step, batch_mean_r, batch_adv_std)
        
        del combined_input_ids, attention_mask, log_probs, logits
        del shift_log_probs, shift_input_ids, shift_attention_mask, token_log_probs, logliks, advantages_tensor, final_loss
        torch.cuda.empty_cache()

        t4 = time.time()
        print(f"""
        Timing:
        generation: {t1 - t0:.2f}s
        reward:     {t2 - t1:.2f}s
        backward:   {t3 - t2:.2f}s
        print & delete:   {t4 - t3:.2f}s
        """)
    
    model.save_pretrained(os.path.join(OUTDIR, "policy"))
    tok.save_pretrained(os.path.join(OUTDIR, "policy"))
    print("✅ Saved fine-tuned policy.")


if __name__ == "__main__":
    main()
