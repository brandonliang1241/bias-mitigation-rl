import json
from pathlib import Path

BASE = Path("data/model_comparisons_bias_1800/baseline.jsonl")
FT = Path("data/model_comparisons_bias_1800/finetuned.jsonl")

OUTL = Path("data/model_comparisons_bias_1800/side_by_side.jsonl")
OUT = Path("data/model_comparisons_bias_1800/side_by_side.json")

# Clear output file ONCE
OUTL.write_text("", encoding="utf-8")

with open(BASE, "r", encoding="utf-8") as f1, open(FT, "r", encoding="utf-8") as f2:
    for line_b, line_f in zip(f1, f2):
        b = json.loads(line_b)
        f = json.loads(line_f)

        with open(OUTL, "a", encoding="utf-8") as o:
            o.write(json.dumps({
                "prompt": b["prompt"],
                "baseline": b["response"],
                "finetuned": f["response"],
                "improvement": ""   # ← manual annotation field
            }, ensure_ascii=False) + "\n")

print("✅ Side-by-side JSONL created")

data = []

with open(OUTL, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Converted to JSON array")
