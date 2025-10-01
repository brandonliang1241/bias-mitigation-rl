import re
from transformers import pipeline

# --- Regex-based lightweight rules ---
GROUP_PATTERNS = [
    r"\b(asians|blacks|black people|whites|white people|women|men|muslims|jews|christians|immigrants)\b",
]

# Expanded TOXIC_MARKERS
TOXIC_MARKERS = [
    r"\b(stupid|idiotic|inferior|superior)\b",
    r"\b(criminals|thugs|scum|parasites|vermin)\b",
    r"\b(lazy|unintelligent|smarter|weaker)\b",
    r"\b(violent|aggressive|brutal)\b",
    r"\b(better|worse)\b",
    r"\b(naturally|tend to|prone to|inherently)\b"
]

# Expanded GENERALIZATION_WORDS
GENERALIZATION_WORDS = [
    r"\b(all|every|everyone)\b",
    r"\b(most|majority|many)\b",
    r"\b(always|never|constantly|invariably)\b",
    r"\b(generally|usually|typically|on average)\b"
]

STYLE_KEYWORDS = [
    r"\bdepends\b", r"\bcontext\b", r"\bnuanced\b",
    r"\bcannot generalize\b", r"\bevidence\b", r"\bsources\b",
    r"\bavoid stereotypes\b"
]

# --- Optional pretrained classifier ---
# Note: For this example, we will assume tox_pipeline is available or fast_mode is True.
try:
    tox_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=0)
except ImportError:
    print("Warning: transformers or toxic-bert model not found. Using fast_mode=True.")
    tox_pipeline = None

# --- Component functions ---
def toxicity_score_regex(text: str) -> float:
    """Calculates a penalty based on the number of toxic markers found."""
    t = text.lower()
    # Now counts every occurrence of a toxic marker for more granularity
    matches = sum(len(re.findall(pat, t)) for pat in TOXIC_MARKERS)
    # Continuous, non-linear penalty based on the count
    return min(matches * 0.4, 1.0) ** 1.5

def group_generalization_score(text: str) -> float:
    """Calculates a penalty for group generalizations."""
    t = text.lower()
    # This function was already using a good, granular approach with findall
    group_count = sum(len(re.findall(p, t)) for p in GROUP_PATTERNS)
    gen_count = sum(len(re.findall(p, t)) for p in GENERALIZATION_WORDS)
    # Continuous, non-linear penalty
    return min(group_count * gen_count * 0.7, 1.0) ** 1.5

def style_bonus(text: str) -> float:
    """Calculates a bonus for using nuanced and balanced language."""
    matches = sum(len(re.findall(pat, text.lower())) for pat in STYLE_KEYWORDS)
    return min(matches * 0.08, 0.2) # Adjusted to be more lenient and higher

def length_bonus(response: str, min_len: int = 20, max_len: int = 80) -> float:
    length = len(response.split())
    if min_len <= length <= max_len:
        return 0.1  # A fixed, small bonus
    return 0.0 # Return 0 for responses outside the range

def classifier_toxicity(text_list: list, tox_pipeline) -> list:
    """Uses a pretrained model to score toxicity for a batch of texts."""
    if tox_pipeline is None:
        return [0.0] * len(text_list)
    
    preds = tox_pipeline(text_list)
    scores = []
    for pred in preds:
        if pred["label"].lower() == "toxic":
            scores.append(pred["score"])
        else:
            scores.append(0.0)
    return scores

def trivial_penalty(prompt: str, response: str) -> float:
    """Calculates a penalty for trivial or repeated responses."""
    penalty = 0.0
    if len(response.split()) < 5:
        penalty += 0.3
    if response.strip().lower() in prompt.strip().lower():
        penalty += 0.4
    return penalty

# --- Normalized, variance-maximized reward ---
def compute_bias_reward(prompt: str, response: str, fast_mode=True, tox_model_score: float = 0.0) -> float:
    """
    Returns [0,1] reward for RL, normalized and variance-maximized.
    - fast_mode=True disables heavy transformer toxicity, but keeps light signal.
    """
    # --- Base components (all normalized to [0,1]) ---
    tox_regex = min(max(toxicity_score_regex(response), 0.0), 1.0)
    gen = min(max(group_generalization_score(response), 0.0), 1.0)
    trivial = min(max(trivial_penalty(prompt, response), 0.0), 1.0)
    style = min(max(style_bonus(response), 0.0), 1.0)
    length = min(max(length_bonus(response), 0.0), 1.0)

    if fast_mode:
        tox_model = tox_regex * 0.5  # light toxicity signal
    else:
        tox_model = min(max(tox_model_score, 0.0), 1.0)

    # --- Amplify stronger violations ---
    tox_score = tox_model ** 1.5      # amplify high toxicity
    gen_score = gen ** 1.3            # amplify biased generation
    trivial_score = trivial            # linear
    style_score = style ** 0.8         # slightly compress high style
    length_score = length ** 0.8      # slightly compress long responses

    # --- Weighted sum ---
    # Add a small, non-zero baseline reward to prevent zero advantages
    baseline_reward = 0.2  # Increased baseline reward to provide a stronger signal
    
    reward = (
        baseline_reward
        + 0.2 * (1 - tox_score)      # Lowered weight for the toxicity term
        + 0.1 * (1 - gen_score)    # Lowered weight for the generalization term
        + 0.3 * style_score         # Increased style bonus weight
        + 0.2 * length_score       # Increased length bonus weight
        - 0.1 * trivial_score      # Lowered trivial penalty
    )

    # clamp to [0, 0.95] to avoid saturation
    reward = max(0.0, min(0.95, reward))
    return reward

