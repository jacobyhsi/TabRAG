#!/usr/bin/env python3
import json
import os
import random
from collections import defaultdict

# --------------------------------------------------
# Config
# --------------------------------------------------
SEED = 42
MAX_QA_PER_ENTRY = 2

INPUT_JSON = "/data2/users/js2723/TabRAG/datasets/comtqa/annotated.json"
GEN_DIR = "/data2/users/js2723/TabRAG/datasets/comtqa/generation"
OUT_JSON = "/data2/users/js2723/TabRAG/datasets/comtqa/qa.json"

random.seed(SEED)

# --------------------------------------------------
# Collect available generation IDs
# --------------------------------------------------
gen_ids = {
    d for d in os.listdir(GEN_DIR)
    if os.path.isdir(os.path.join(GEN_DIR, d))
}

print(f"[INFO] Found {len(gen_ids)} generation entries")

# --------------------------------------------------
# Load QA annotations
# --------------------------------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# --------------------------------------------------
# Group QAs by generation entry
# --------------------------------------------------
grouped = defaultdict(list)

for item in qa_data:
    # FinTabNet
    if item.get("table_id") is not None:
        entry_id = str(item["table_id"])
        if entry_id in gen_ids:
            grouped[entry_id].append(item)
        continue

    # PubTables
    if item.get("image_name") is not None:
        entry_id = os.path.splitext(item["image_name"])[0]
        if entry_id in gen_ids:
            grouped[entry_id].append(item)

# --------------------------------------------------
# Randomly sample up to N QAs per entry
# --------------------------------------------------
filtered = []

for entry_id, qas in grouped.items():
    if len(qas) <= MAX_QA_PER_ENTRY:
        filtered.extend(qas)
    else:
        filtered.extend(random.sample(qas, MAX_QA_PER_ENTRY))

# --------------------------------------------------
# Write output
# --------------------------------------------------
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"[DONE] Wrote {len(filtered)} QA pairs to {OUT_JSON}")
print(f"[INFO] Avg QAs per entry: {len(filtered) / max(len(grouped), 1):.2f}")
