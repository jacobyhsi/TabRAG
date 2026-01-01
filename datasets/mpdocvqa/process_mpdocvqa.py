import os
import sys
import shutil
import json
import warnings
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# sys.path.append(os.path.abspath("object_detection"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)
from src.layout import LayoutProcessor

warnings.filterwarnings("ignore")

# ============================================================
# PART 1 — Group MPDocVQA images by prefix
# ============================================================
def group_images_by_prefix(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    # Get list of files first so tqdm knows the total count
    file_list = os.listdir(src_dir)

    # Wrap the list in tqdm for a progress bar
    for fname in tqdm(file_list, desc="Grouping images"):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if "_p" not in fname:
            continue

        prefix = fname.split("_p")[0]
        prefix_folder = os.path.join(dst_dir, prefix)
        os.makedirs(prefix_folder, exist_ok=True)

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(prefix_folder, fname)
        shutil.copy(src_path, dst_path)

    print("\nDone grouping images by prefix.")


# ============================================================
# PART 2 — LayoutProcessor (table detection)
# ============================================================
lp = LayoutProcessor()

def extract_table_images(base_dir, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)

    print("\n=== Running table detection ===\n")

    for root, subdirs, files in tqdm(os.walk(base_dir), desc="Scanning directories", unit="index"):
        if not any(f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")) for f in files):
            continue

        print(f"Processing directory: {root}")

        image_files = sorted([
            f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        for f in tqdm(image_files, desc="Processing files", leave=False):
            file_path = os.path.join(root, f)
            comps = lp.preprocess(file_path)

            save_whole_file = False
            for page in comps:
                for comp in comps[page]:
                    if comp["class"] == 3:  # TABLE
                        save_whole_file = True
                        break
                if save_whole_file:
                    break

            if save_whole_file:
                rel_path = os.path.relpath(file_path, base_dir)
                dest_path = os.path.join(tables_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)

    print("\n✔ Finished extracting table images.")


# ============================================================
# PART 3 — Compute ratios + delete low-value folders
# ============================================================
def prune_tables_by_ratio(val_json_path, tables_dir, file_limit=500):

    print("\n=== Pruning tables by matched question ratio ===\n")

    with open(val_json_path, "r") as f:
        data = json.load(f)["data"]

    folder_page_files = {}
    for folder in os.listdir(tables_dir):
        folder_path = os.path.join(tables_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files = [
            f[:-4] for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        folder_page_files[folder] = set(files)

    folder_question_matches = defaultdict(int)

    for q in data:
        for pid in q["page_ids"]:
            doc_id = pid.split("_p")[0]
            if doc_id in folder_page_files:
                if pid in folder_page_files[doc_id]:
                    folder_question_matches[doc_id] += 1

    ratios = []
    for folder, pages in folder_page_files.items():
        num_pages = len(pages)
        matched_qs = folder_question_matches.get(folder, 0)
        if num_pages > 0:
            ratios.append((folder, matched_qs, num_pages, matched_qs / num_pages))

    ratios_sorted = sorted(ratios, key=lambda x: x[3])

    total_files = sum(1 for _ in Path(tables_dir).rglob("*") if _.is_file())
    print(f"Total files currently in {tables_dir}: {total_files}")

    print("\nBottom 10 folders by matched_questions/pages ratio (before deletion):\n")
    for folder, q_count, p_count, ratio in ratios_sorted[:10]:
        print(f"{folder}: {q_count} / {p_count} = {ratio:.2f}")

    delete_list = []
    remaining_files = total_files

    for folder, q_count, p_count, ratio in ratios_sorted:
        if remaining_files <= file_limit:
            break
        delete_list.append((folder, q_count, p_count, ratio))
        remaining_files -= p_count

    print(f"\nFolders to delete to reach <{file_limit} files (remaining {remaining_files}):\n")
    for folder, q_count, p_count, ratio in delete_list:
        print(f"{folder}: {q_count}/{p_count} = {ratio:.2f}")

    deleted_folders = {f for f, _, _, _ in delete_list}
    remaining_question_count = 0

    for q in data:
        for pid in q["page_ids"]:
            doc_id = pid.split("_p")[0]
            if doc_id in folder_page_files and doc_id not in deleted_folders:
                if pid in folder_page_files[doc_id]:
                    remaining_question_count += 1

    print(f"\nTotal remaining matched questions after deletion: {remaining_question_count}")

    for folder, q_count, p_count, ratio in delete_list:
        folder_path = os.path.join(tables_dir, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted: {folder_path}")
        else:
            print(f"Skipped (not found): {folder_path}")

    print("\n✔ Pruning complete.\n")


# ============================================================
# PART 4 — Build generation/ tree
# ============================================================
def build_generation_tree(tables_dir, generation_dir):
    os.makedirs(generation_dir, exist_ok=True)

    print("\n=== Step 4: Building generation/directory structure ===\n")

    for folder in os.listdir(tables_dir):
        folder_path = os.path.join(tables_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            page_id = fname[:-4]
            prefix = page_id.split("_p")[0]

            out_dir = os.path.join(generation_dir, prefix, page_id)
            os.makedirs(out_dir, exist_ok=True)

            src_path = os.path.join(folder_path, fname)
            dst_path = os.path.join(out_dir, fname)

            shutil.copy(src_path, dst_path)

    print("✔ Finished building generation/ tree.\n")


# ============================================================
# PART 5 — Pretty-print val.json
# ============================================================
def pretty_print_val_json(val_json_path):
    print("\n=== Pretty-printing val.json ===\n")

    with open(val_json_path, 'r') as f:
        data = json.load(f)

    with open(val_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("✔ val.json pretty-printed successfully.\n")


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    raw_images = os.path.join(ROOT, "datasets/mpdocvqa/images")
    grouped_images = os.path.join(ROOT, "datasets/mpdocvqa/images_test")
    tables_output = os.path.join(ROOT, "datasets/mpdocvqa/retrieval")
    generation_output = os.path.join(ROOT, "datasets/mpdocvqa/generation")
    val_json = os.path.join(ROOT, "datasets/mpdocvqa/val.json")

    print("\n=== Step 1: Grouping images by prefix ===")
    group_images_by_prefix(raw_images, grouped_images)

    print("\n=== Step 2: Detecting tables and copying images ===")
    extract_table_images(grouped_images, tables_output)

    print("\n=== Step 3: Pruning low-ratio folders ===")
    prune_tables_by_ratio(val_json, tables_output, file_limit=500)

    print("\n=== Step 4: Building generation/ tree ===")
    build_generation_tree(tables_output, generation_output)

    print("\n=== Step 5: Pretty-printing val.json ===")
    pretty_print_val_json(val_json)

    print("\nAll tasks completed successfully.\n")
