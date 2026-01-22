import json
import re

input_file = r"data\prompts\prompts_raw_v1.txt"
output_file = r"data\prompts\new_prompts.jsonl"

prompts = []

# --- Read and clean prompts ---
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Remove leading numbers like "1. " or "23. "
        clean_line = re.sub(r"^\d+\.\s*", "", line)
        prompts.append(clean_line)

# --- Write to JSONL ---
with open(output_file, "w", encoding="utf-8") as f:
    for prompt_text in prompts:
        obj = {"prompt": prompt_text}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ {len(prompts)} prompts written to {output_file}")
