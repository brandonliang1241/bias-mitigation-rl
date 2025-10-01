from detoxify import Detoxify
import numpy as np
from ..utils.gen import batch_generate_responses
from ..utils.io import load_prompts_jsonl
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Detoxify once globally
detoxifier = Detoxify("original")  

def evaluate_detoxify(base_model_name, tuned_model_name, dataset, device="cuda"):
    """
    Evaluate toxicity scores for baseline vs fine-tuned models on a dataset of prompts.
    Returns (mean_base, mean_tuned, mean_diff, ci, p_val).
    """

    # Load models
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side='left')
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)

    tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_model_name, trust_remote_code=True, padding_side='left')
    tuned_model = AutoModelForCausalLM.from_pretrained(tuned_model_name, trust_remote_code=True).to(device)

    # Ensure dataset is loaded (if file path was passed)
    if isinstance(dataset, str):
        import json
        loaded = []
        with open(dataset, "r", encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line))
        dataset = loaded

    prompts = [ex["prompt"] for ex in dataset]

    # Generate responses
    base_responses = batch_generate_responses(base_model, base_tokenizer, prompts, device=device)
    tuned_responses = batch_generate_responses(tuned_model, tuned_tokenizer, prompts, device=device)

    # Detoxify scoring
    detoxifier = Detoxify("original")
    base_scores = detoxifier.predict(base_responses)["toxicity"]
    tuned_scores = detoxifier.predict(tuned_responses)["toxicity"]

    base_mean = float(np.mean(base_scores))
    tuned_mean = float(np.mean(tuned_scores))
    diff = tuned_mean - base_mean

    # Bootstrap CI + p-value
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(prompts), len(prompts))
        diffs.append(np.mean(np.array(tuned_scores)[idx]) - np.mean(np.array(base_scores)[idx]))
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))

    # p-value = fraction of bootstrap samples ≥ 0
    p_val = np.mean(np.array(diffs) >= 0)

    return base_mean, tuned_mean, diff, ci, p_val
