from .detox_eval import evaluate_detoxify
from .crows_eval import evaluate_crows_for_model
from .truthfulqa_eval import evaluate_truthfulqa

BENCHMARKS = {
    "detoxify": evaluate_detoxify,
    "crows": evaluate_crows_for_model,
    "truthfulqa": evaluate_truthfulqa,
}
