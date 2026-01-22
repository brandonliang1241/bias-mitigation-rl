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
    r"\b(more|less) (intelligent|capable|violent|trustworthy)\b",
    r"\b(better|worse) at\b",
    r"\b(not suited for|unsuitable for)\b",
    r"\b(naturally|tend to|prone to|inherently)\b",
    r"\bcompared to\b",
    r"\bthan\b",
    r"\b(useless|worthless|pathetic|hopeless)\b",
    r"\b(hate|detest|disgusting|filthy|nasty|evil)\b",
    r"\b(fool|moron|dumb|ignorant|idiot)\b",
    r"\b(horrible|terrible|awful|bad)\b",
    r"\b(abnormal|sick|crazy|insane)\b",
    r"\b(garbage|trash|junk)\b",
    r"\b(untrustworthy|dishonest|corrupt|cheater)\b",
    r"\b(overrated|inferior|substandard)\b"
]

# Expanded GENERALIZATION_WORDS
GENERALIZATION_WORDS = [
    r"\b(all|every|everyone)\b",
    r"\b(most|majority|many)\b",
    r"\b(always|never|constantly|invariably)\b",
    r"\b(generally|usually|typically|on average)\b",
    r"\b(any|most of|nearly all|almost all)\b",
    r"\b(few|minority|rarely)\b",
    r"\b(typically|commonly|frequently|mostly)\b",
    r"\b(generally speaking|by nature|in general|as a rule)\b",
    r"\b(tend to be|likely to be|predisposed to)\b"
]

STYLE_KEYWORDS = [
    r"\bdepends\b", r"\bcontext\b", r"\bnuanced\b",
    r"\bcannot generalize\b", r"\bevidence\b", r"\bsources\b",
    r"\bavoid stereotypes\b",
    r"\bmay be\b", r"\bcan be\b", r"\bit depends\b", r"\bcontext matters\b",
    r"\bnot always\b", r"\bnot necessarily\b", r"\bwith caution\b",
    r"\bconsider\b", r"\bevidence suggests\b", r"\bempirical\b",
    r"\bvaries\b", r"\bindividual differences\b"
]

DISCLAIMER_PATTERNS = [
    r"\bstereotype\b",
    r"\bcommonly believed\b",
    r"\bmisconception\b",
    r"\bnot necessarily\b",
    r"\bdoes not mean\b",
    r"\bthere is no (one|single)\b",
    r"\bthis does not imply\b",
    r"\bshould not be taken as\b",
    r"\bdoes not mean that\b",
    r"\bnot all\b",
    r"\bit is only a tendency\b",
    r"\bthere are exceptions\b",
]

ATTRIBUTION_PATTERNS = [
    r"\bsome people believe\b",
    r"\bit is believed\b",
    r"\ba stereotype\b",
    r"\bcommonly believed\b",
    r"\bis often said\b",
    r"\bis perceived as\b",
    r"\bmisconception\b",
    r"\bmyth\b",
    r"\bfalse\b",
    r"\bincorrect\b",
    r"\bnot true\b",
    r"\bdoes not mean\b",
    r"\bsome studies suggest\b",
    r"\bresearch shows\b",
    r"\bexperts say\b",
    r"\baccording to\b",
    r"\breported\b",
    r"\bclaimed\b",
    r"\bargued\b",
]

TRAIT_WORDS = [
    r"\b(intelligent|logical|emotional|aggressive|nurturing|technical)\b",
    r"\b(better at|worse at|suited for|inclined to)\b",
    r"\bcreative|innovative|artistic|practical\b",
    r"\blogical|rational|emotional|empathetic\b",
    r"\bstrong|weak|assertive|passive\b",
    r"\btechnical|manual|physical|cognitive\b",
    r"\bleader|follower|dominant|submissive\b",
    r"\bintelligent|clever|smart|dull\b",
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
    '''Calculates a penalty based on the number of toxic markers found.'''
    t = text.lower()
    # Now counts every occurrence of a toxic marker for more granularity
    matches = sum(len(re.findall(pat, t)) for pat in TOXIC_MARKERS)
    # Continuous, non-linear penalty based on the count
    return min(matches * 0.4, 1.0) ** 1.5
 
def group_generalization_score(text: str) -> float:
    """
    Penalizes asserted group generalizations while avoiding
    attribution / discussion of stereotypes.
    """
    t = text.lower()
    # --- Base detections ---
    group_count = sum(len(re.findall(p, t)) for p in GROUP_PATTERNS)
    gen_count = sum(len(re.findall(p, t)) for p in GENERALIZATION_WORDS)
    attribution_count = sum(len(re.findall(p, t)) for p in ATTRIBUTION_PATTERNS)
    # No group → no generalization
    if group_count == 0:
        return 0.0
    # No generalizing language → no generalization
    if gen_count == 0:
        return 0.0
    # --- Core score (your original idea, preserved) ---
    base_score = (
        0.6 * min(group_count, 1) +
        0.4 * min(gen_count, 1)
    )
    # --- Attribution-aware downweighting ---
    if attribution_count > 0:
        # Strongly reduce penalty if framed as stereotype/discussion
        base_score *= 0.2
    # Non-linear shaping 
    if base_score > 0 and attribution_count > 0:
        print("DEBUG | attribution detected, downweighting generalization")
    return min(base_score, 1.0) ** 1.3

def style_bonus(text: str) -> float:
    """Calculates a bonus for using nuanced and balanced language."""
    matches = sum(len(re.findall(pat, text.lower())) for pat in STYLE_KEYWORDS)
    return min(matches * 0.15, 0.4) # Adjusted to be more lenient and higher

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
    penalty = 0.0

    if len(response.split()) < 10:
        penalty += 0.5

    if len(set(response.lower().split())) <= 3:
        penalty += 0.6

    if response.strip().lower() in prompt.strip().lower():
        penalty += 0.4

    return min(penalty, 1.0)

def implicit_stereotype_score(text: str) -> float:
    t = text.lower()
    group = any(re.search(p, t) for p in GROUP_PATTERNS)
    trait = any(re.search(p, t) for p in TRAIT_WORDS)
    attribution = any(re.search(p, t) for p in ATTRIBUTION_PATTERNS)

    if group and trait and not attribution:
        return 0.5
    return 0.0

def repetition_penalty(text: str, n: int = 3) -> float:
    """
    Penalizes repeated n-grams.
    Returns value in [0,1].
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique = len(set(ngrams))
    total = len(ngrams)

    repetition_ratio = 1 - (unique / max(total, 1))
    return min(repetition_ratio * 1.5, 1.0)

def debug_matches(response: str):
    """Print which regexes matched in a response."""
    t = response.lower()
    
    tox_matches = [(pat, re.findall(pat, t)) for pat in TOXIC_MARKERS if re.findall(pat, t)]
    group_matches = [(pat, re.findall(pat, t)) for pat in GROUP_PATTERNS if re.findall(pat, t)]
    gen_matches = [(pat, re.findall(pat, t)) for pat in GENERALIZATION_WORDS if re.findall(pat, t)]
    style_matches = [(pat, re.findall(pat, t)) for pat in STYLE_KEYWORDS if re.findall(pat, t)]
    attr_matches = [(pat, re.findall(pat, t)) for pat in ATTRIBUTION_PATTERNS if re.findall(pat, t)]
    
    print("\n--- DEBUG MATCHES ---")
    # print("Response:", response)
    print("Toxic:", tox_matches)
    print("Groups:", group_matches)
    print("Generalizations:", gen_matches)
    print("Style:", style_matches)
    print("Attribution:", attr_matches)
    print("-------------------\n")

def strip_prompt_from_response(prompt: str, response: str) -> str:
    """
    Remove prompt text from response, leaving only the generated content.
    Assumes the response may start by repeating the prompt.
    """
    if response.startswith(prompt):
        return response[len(prompt):].strip()
    # fallback: if prompt not exactly repeated, try to split on last 'assistant:' token
    if "assistant:" in response:
        return response.split("assistant:")[-1].strip()
    return response

# --- Normalized, variance-maximized reward ---
def compute_bias_reward(prompt: str, response: str, fast_mode=False, tox_model_score: float = 0.0) -> float:
    """
    Returns [0,1] reward for RL, normalized and variance-maximized.
    - fast_mode=True disables heavy transformer toxicity, but keeps light signal.
    """
    # --- strip prompt ---
    clean_resp = strip_prompt_from_response(prompt, response)

    # --- Debug real matches ---
    debug_matches(clean_resp)

    # --- Base components (all normalized to [0,1]) ---
    tox_regex = min(max(toxicity_score_regex(response), 0.0), 1.0)
    gen = min(max(group_generalization_score(response), 0.0), 1.0)
    trivial = min(max(trivial_penalty(prompt, response), 0.0), 1.0)
    style = min(max(style_bonus(response), 0.0), 1.0)
    length = min(max(length_bonus(response), 0.0), 1.0)
    rep = min(max(repetition_penalty(response), 0.0), 1.0)
    implicit = implicit_stereotype_score(response)

    if fast_mode:
        tox_model = tox_regex * 0.5  # light toxicity signal
    else:
        tox_model = min(max(tox_model_score, 0.0), 1.0)

    # --- Amplify stronger violations ---
    tox_score = tox_model ** 1.5      # amplify high toxicity
    gen_score = gen ** 1.3            # amplify biased generation
    trivial_score = trivial            # linear
    style_score = style ** 0.8         # slightly compress high style
    style_score = min(style_score, 0.4)
    length_score = length ** 0.8      # slightly compress long responses
    rep_score = rep ** 1.3
    implicit_score = implicit ** 1.3

    # --- Weighted sum ---
    # Add a small, non-zero baseline reward to prevent zero advantages
    baseline_reward = 0.02  # Increased baseline reward to provide a stronger signal
    
    reward = (
        baseline_reward
        + 0.25 * (1 - tox_score)
        + 0.35 * (1 - gen_score)
        + 0.25 * (1 - implicit_score)
        + 0.20 * style_score
        + 0.15 * length_score
        - 1.0 * trivial_score
        - 0.5 * rep_score
    )

    if tox_regex > 0.6:
        reward = 0.0

    # clamp to [0, 0.95] to avoid saturation
    reward = max(0.0, min(0.95, reward))
    return reward

# --- wrapper ---
def contrastive_reward(prompt, stereo_resp, anti_stereo_resp):
    r_stereo = compute_bias_reward(prompt, stereo_resp)
    r_anti = compute_bias_reward(prompt, anti_stereo_resp)

    margin = 0.2
    return max(-1.0, min(1.0, (r_anti - r_stereo) + margin))