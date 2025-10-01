import os, sys, random, json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, LogitsProcessor, pipeline
from torch import amp

torch.cuda.empty_cache()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward, toxicity_score_regex, group_generalization_score, style_bonus, trivial_penalty, classifier_toxicity

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
# Define the quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
PROMPTS = "data/prompts/pilot_prompts.jsonl"
OUTDIR = "results/grpo_pilot"

TOTAL_STEPS = 1000
PRINT_INTERVAL = 1
SAVE_INTERVAL = 100 # Save a checkpoint every 50 steps
K = 4
MAX_NEW_TOKENS = 120
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
    return tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                        tokenize=False, add_generation_prompt=True)


# --- sample multiple responses safely ---
def sample_responses_batched(model, tok, prompts, K, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_p=0.9):
    """
    Sample K responses for a batch of prompts.
    """
    # 1. Prepare batched inputs
    batched_texts = [build_chat(tok, p) for p in prompts]
    inputs = tok(batched_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # 2. Generate responses in a single call
    logits_processor = SafeClampLogitsProcessor()

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=K,
            max_new_tokens=max_new_tokens,
            # Add the processor here
            logits_processor=[logits_processor] 
        )
    
    # 3. Decode the batched outputs
    # The output tensor has shape (batch_size * K, sequence_length)
    responses_list = tok.batch_decode(outs, skip_special_tokens=True)
    
    # 4. Restructure responses to match the original prompts
    responses_by_prompt = []
    for i in range(len(prompts)):
        start_idx = i * K
        end_idx = start_idx + K
        prompt_responses = [
            r.split(prompts[i])[-1].strip() for r in responses_list[start_idx:end_idx]
        ]
        responses_by_prompt.append({"prompt": prompts[i], "responses": prompt_responses})

    return responses_by_prompt


# --- compute reward components ---
def reward_components(prompt, response, tox_model_score, fast_mode=True):
    tox_regex = toxicity_score_regex(response)
    gen = group_generalization_score(response)
    trivial = trivial_penalty(prompt, response)
    style = style_bonus(response)
    
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
    # if tok.pad_token is None:
    #     tok.add_special_tokens({'pad_token': '<|pad|>'})

    # # 3. Resize the model's token embeddings to accommodate the new token.
    # model.resize_token_embeddings(len(tok))

    # # 4. CRITICAL: Update the model's configuration to use the new pad token ID.
    # # This is the step that resolves the CUDA error.
    # model.config.pad_token_id = tok.pad_token_id
    
    print("Initializing toxicity classifier...")
    tox_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=-1, truncation=True)
    print("Initialization complete.")

    model.train()

    prompts = load_prompts(PROMPTS, n=300)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, TOTAL_STEPS)

    step = 0
    batch_size = 1  # Increase batch size to fully leverage the GPU

    while step < TOTAL_STEPS:
        print(f"Step {step}:")
        batch_prompts_data = random.sample(prompts, batch_size)
        batch_prompts = [ex["prompt"] for ex in batch_prompts_data]
        
        allocated_before = torch.cuda.memory_allocated()
        print(f"Memory allocated before empty_cache(): {allocated_before / (1024**2):.2f} MB")
        torch.cuda.empty_cache()
        allocated_after = torch.cuda.memory_allocated()
        print(f"Memory allocated after empty_cache(): {allocated_after / (1024**2):.2f} MB")

        # --- 1. Batch generate responses for the entire batch ---
        all_responses_data = sample_responses_batched(model, tok, batch_prompts, K, MAX_NEW_TOKENS)
        
        # --- 2. Flatten all prompts and responses to prepare for reward calculation ---
        flat_prompts = []
        flat_responses = []
        for ex_data in all_responses_data:
            flat_prompts.extend([ex_data["prompt"]] * len(ex_data["responses"]))
            flat_responses.extend(ex_data["responses"])
            
        # --- 3. Batch compute rewards ---
        tox_model_scores = classifier_toxicity(flat_responses, tox_pipeline=tox_pipeline)
        
        # You can't easily batch the other reward functions unless you modify them
        # to accept lists. For now, we'll keep this loop but acknowledge it's a bottleneck.
        # ✅ Pass the pipeline as an argument
        
        all_rewards = []
        all_tox, all_gen, all_style, all_trivial = [], [], [], []
        
        for i, r in enumerate(flat_responses):
            reward, tox, gen, style, trivial = reward_components(flat_prompts[i], r, tox_model_scores[i], fast_mode=FAST_MODE)
            all_rewards.append(reward)
            all_tox.append(tox)
            all_gen.append(gen)
            all_style.append(style)
            all_trivial.append(trivial)
            
        # --- 4. Compute advantages and create a single batched loss input ---
        reshaped_rewards = np.array(all_rewards).reshape(batch_size, K)
        baselines = np.median(reshaped_rewards, axis=1, keepdims=True)
        advantages = (reshaped_rewards - baselines).flatten()
        
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
        
        labels = input_ids.clone()
        user_lens_per_seq = [len(tok(build_chat(tok, flat_prompts[i]))["input_ids"]) for i in range(len(flat_prompts))]
        
        for i in range(input_ids.size(0)):
            labels[i, :user_lens_per_seq[i]] = -100

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
        
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        per_sequence_loss = losses.view(input_ids.size(0), -1).mean(dim=1)
        logliks = -per_sequence_loss
        
        advantages_tensor = torch.tensor(advantages, dtype=logliks.dtype).to(model.device)
        # Normalize the advantages to prevent large loss spikes
        if advantages_tensor.std() > 1e-6: # Check to prevent division by zero
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / advantages_tensor.std()
        else:
            advantages_tensor = torch.zeros_like(advantages_tensor)
        
        final_loss = (-advantages_tensor * logliks).mean()
        
        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        torch.cuda.empty_cache()
        step += 1

        # --- 6. Logging and saving (simplified) ---
        if step % PRINT_INTERVAL == 0:
            print(f"step {step} | loss {final_loss.item():.4f} | mean_r {np.mean(all_rewards):.3f} | baseline {np.mean(baselines):.3f}")
            print(f"      tox={np.mean(all_tox):.3f} gen={np.mean(all_gen):.3f} style={np.mean(all_style):.3f} trivial={np.mean(all_trivial):.3f}")

        if step % SAVE_INTERVAL == 0:
            checkpoint_dir = os.path.join(OUTDIR, "checkpoints", f"step_{step}")
            model.save_pretrained(checkpoint_dir)
            tok.save_pretrained(checkpoint_dir)
            print(f"✅ Saved checkpoint at step {step} to {checkpoint_dir}")
    
    model.save_pretrained(os.path.join(OUTDIR, "policy"))
    tok.save_pretrained(os.path.join(OUTDIR, "policy"))
    print("✅ Saved fine-tuned policy.")


if __name__ == "__main__":
    main()
