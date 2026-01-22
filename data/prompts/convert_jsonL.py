import json

data = []
with open("data/prompts/response_pairs_raw.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open("data/prompts/response_pairs_raw.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
