import os
import json
from datasets import load_dataset

# Define ROOT as the project's base directory (two levels up from this script)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ds = load_dataset("terryoo/TableVQA-Bench")

def save_images(example):
    img = example["image"]
    img_id = example['qa_id'].split('.')[0]
    
    # Save to ROOT/datasets/tablevqa_bench_processed/
    save_dir = os.path.join(ROOT, f"datasets/tablevqa/generation/{img_id}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img.save(os.path.join(save_dir, f"{img_id}.png"))
    return example

def extract_metadata(example, split_name):
    return {
        "split": split_name,
        "qa_id": example["qa_id"],
        "question": example["question"],
        "text_markdown_table": example["text_markdown_table"],
        "text_html_table": example["text_html_table"],
        "gt": example["gt"],
    }

ds["fintabnetqa"].map(save_images, load_from_cache_file=False)
ds["vtabfact"].map(save_images, load_from_cache_file=False)

out_path = os.path.join(ROOT, "datasets/tablevqa/qa.json")
all_records = []

for split_name in ["fintabnetqa", "vtabfact"]:
    for ex in ds[split_name]:
        all_records.append(extract_metadata(ex, split_name))
with open(out_path, "w") as f:
    json.dump(all_records, f, indent=2)
