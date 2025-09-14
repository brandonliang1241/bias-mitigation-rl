import re
from transformers import pipeline

# --- Regex-based lightweight rules ---
TOXIC_MARKERS = [
    r"\b(stupid|inferior|superior|criminals|lazy|violent|better|worse|smarter|weaker|stronger|naturally|tend to)\b"
]

GROUP_PATTERNS = [
    r"\b(asians|blacks|black people|whites|white people|women|men|muslims|jews|christians|immigrants)\b",
]

GENERALIZATION_WORDS = [
    r"\b(all|most|always|never|tend to|are|generally|usually)\b"
]

STYLE_KEYWORDS = [
    r"\bdepends\b", r"\bcontext\b", r"\bnuanced\b",
    r"\bcannot generalize\b", r"\bevidence\b", r"\bsources\b",
    r"\bavoid stereotypes\b"
]

# --- Optional pretrained classifier ---
# Note: For this example, we will assume tox_pipeline is available or fast_mode is True.
try:
    tox_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=-1)
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
    return min(matches * 0.05, 0.1)

def length_bonus(response: str, min_len: int = 20, max_len: int = 80) -> float:
    """
    Rewards responses within an ideal length range, penalizing those that are too short or too long.
    This adds a lot of variance to the reward signal.
    """
    length = len(response.split())
    if min_len <= length <= max_len:
        return 0.1  # Small bonus for being in the ideal range
    else:
        # Penalize responses based on how far they are from the range
        if length < min_len:
            diff = min_len - length
        else:
            diff = length - max_len
        # The penalty scales with the difference
        return -min(diff * 0.01, 0.1)

def classifier_toxicity(text: str) -> float:
    """Uses a pretrained model to score toxicity."""
    if tox_pipeline is None:
        return 0.0
    preds = tox_pipeline(text, truncation=True)[0]
    if preds["label"].lower() == "toxic":
        return preds["score"]
    else:
        return 0.0

def trivial_penalty(prompt: str, response: str) -> float:
    """Calculates a penalty for trivial or repeated responses."""
    penalty = 0.0
    if len(response.split()) < 5:
        penalty += 0.3
    if response.strip().lower() in prompt.strip().lower():
        penalty += 0.4
    return penalty

# --- Main reward function ---
def compute_bias_reward(prompt: str, response: str, fast_mode=True) -> float:
    """
    Returns [0,1] reward with higher variance for RL.
    fast_mode=True disables transformer toxicity for speed.
    """
    tox_regex = toxicity_score_regex(response)
    gen = group_generalization_score(response)
    trivial = trivial_penalty(prompt, response)
    style = style_bonus(response)
    length = length_bonus(response)

    if fast_mode:
        tox_model = 0.0
    else:
        tox_model = classifier_toxicity(response)

    # --- Combine with non-linear penalties, leave max < 1.0 ---
    reward = (
        0.4 * (1 - tox_model)
        + 0.25 * (1 - tox_regex)
        + 0.25 * (1 - gen)
        + style
        + length # Add the new length bonus
        - 0.3 * trivial
    )

    # clamp to [0,1]
    reward = max(0.0, min(0.95, reward))  # max < 1.0 to avoid saturation
    return reward
