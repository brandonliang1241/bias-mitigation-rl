import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from analysis.utils.gen import batch_generate_responses

def evaluate_truthfulqa(base_model_name, finetuned_model_name, split="validation", device="cuda"):
    # 1. Load dataset
    dataset = load_dataset("truthful_qa", "generation")[split]
    prompts = [ex["question"] for ex in dataset]

    # 2. Load models
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    base_tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,padding_side='left')

    tuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, trust_remote_code=True).to(device)
    tuned_tok = AutoTokenizer.from_pretrained(finetuned_model_name, trust_remote_code=True,padding_side='left')

    # 3. Generate responses
    base_responses = batch_generate_responses(base_model, base_tok, prompts, device=device)
    tuned_responses = batch_generate_responses(tuned_model, tuned_tok, prompts, device=device)

    # 4. Score with reference answers
    # The dataset has "best_answer" (truthful reference)
    refs = [ex["best_answer"] for ex in dataset]

    def score_response(response, ref):
        """Simple truthfulness heuristic: 1 if ref substring appears in response, else 0."""
        return int(ref.lower() in response.lower())

    base_scores = [score_response(r, ref) for r, ref in zip(base_responses, refs)]
    tuned_scores = [score_response(r, ref) for r, ref in zip(tuned_responses, refs)]

    # 5. Aggregate
    return {
        "truthfulqa_baseline_mean": float(np.mean(base_scores)),
        "truthfulqa_finetuned_mean": float(np.mean(tuned_scores)),
    }
