import sys
import os
import heapq
import json
from tqdm import tqdm
import argparse

import random
random.seed(0)


# Add object_detection to path so src.layout works
sys.path.append(os.path.abspath("object_detection"))

from src.llm import VLLMVLMClient
from src.prompts import VLMPrompts
from src.layout import LayoutProcessor


def main(args):
    # Initialize Models
    print("Initializing models...")
    try:
        vlm = VLLMVLMClient(args.vlm_model, ip=args.vlm_ip, port=args.vlm_port)
        lp = LayoutProcessor()
        vlmp = VLMPrompts()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    data_dir = f"datasets/{args.dataset}"
    folders_processed = 0
    candidate_tables = []  # List of (area, component_dict)

    print(f"Scanning {args.num_folders} folders in {data_dir}...")

    # Walk through directories
    for root, subdirs, files in os.walk(data_dir):
        random.shuffle(subdirs)
        # files.sort()

        valid_files = [
            f for f in files
            if f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))
        ]

        if not valid_files:
            continue

        file_name = valid_files[0]
        file_path = os.path.join(root, file_name)

        print(f"Processing folder: {root}, file: {file_name}")

        try:
            comps = lp.preprocess(file_path)

            largest_table = None
            max_area = -1

            # comps: {page_num: [components]}
            for page_num, components in comps.items():
                for comp in components:
                    if comp["class"] == 3:  # Table class
                        bbox = comp["bbox"]
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        area = width * height

                        if area > max_area:
                            max_area = area
                            largest_table = comp

            if largest_table:
                candidate_tables.append((max_area, largest_table))
                print(f"  Found table with area {max_area}")
            else:
                print("  No table found in this file.")

        except Exception as e:
            print(f"  Error processing file {file_path}: {e}")
            continue

        folders_processed += 1
        if folders_processed >= args.num_folders:
            break

    print(f"\nFound {len(candidate_tables)} tables total.")

    # Select top N largest tables
    top_tables = heapq.nlargest(
        args.num_icl, candidate_tables, key=lambda x: x[0]
    )

    print(f"Generating ICL examples for top {len(top_tables)} tables...")

    icl_examples = []  # List[str]

    for i, (area, comp) in enumerate(top_tables):
        print(f"Generating example {i+1}/{len(top_tables)} (Area: {area})...")

        vlm_image = comp["path"]

        # Generate Markdown
        markdown_output = vlm.generate(
            vlmp.vlm_table_icl_markdown_prompt,
            vlm_image
        )

        # Generate JSON
        json_output = vlm.generate(
            vlmp.vlm_table_icl_json_prompt,
            vlm_image
        )

        example = (
            f"Input:\n{markdown_output}\n\n"
            f"Output:\n{json_output}"
        )

        icl_examples.append(example)

    # Console output (optional)
    print("\n" + "=" * 50)
    print("GENERATED ICL EXAMPLES")
    print("=" * 50)

    for i, example in enumerate(icl_examples):
        print(f"\n--- Example {i+1} ---")
        print(example)
        print("\n" + "-" * 20)

    # Save JSON (list of example strings)

    if not os.path.exists("icl"):
        os.makedirs("icl")

    with open(f"icl/{args.dataset}_{args.vlm_model.split('/')[-1]}_icl.json", "w") as f:
        json.dump(icl_examples, f, indent=2)

    print(f"\nSaved examples to icl/{args.dataset}_{args.vlm_model.split('/')[-1]}_icl.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folders", type=int, default=10, help="Number of folders to iterate over")
    parser.add_argument("--num_icl", type=int, default=3, help="Number of ICL examples to generate")

    parser.add_argument("--dataset", type=str, default="tatdqa", help="Dataset name") # modify

    # parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-32B-Instruct") # modify
    # parser.add_argument("--vlm_ip", type=str, default="146.169.26.172") # modify
    # parser.add_argument("--vlm_port", type=str, default="3232") # modify

    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct") # modify
    parser.add_argument("--vlm_ip", type=str, default="146.169.1.69") # modify
    parser.add_argument("--vlm_port", type=str, default="6200") # modify

    args = parser.parse_args()
    main(args)
