import sys
import os
import heapq
import json
from tqdm import tqdm
import argparse
import random

random.seed(0)

import warnings
warnings.filterwarnings("ignore")

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
    candidate_tables = []  # List of (area, component_dict)

    print(f"Scanning for {args.num_folders} files containing tables in {data_dir}...")

    # Walk through directories
    pbar = tqdm(total=args.num_folders, desc="Tables Found")

    for root, subdirs, files in os.walk(data_dir):
        if len(candidate_tables) >= args.num_folders:
            break
            
        random.shuffle(subdirs)

        valid_files = [
            f for f in files
            if f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))
        ]

        if not valid_files:
            continue

        for file_name in valid_files:
            file_path = os.path.join(root, file_name)
            
            try:
                comps = lp.preprocess(file_path)
                largest_table = None
                max_area = -1

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
                    pbar.update(1)
                    break 

            except Exception as e:
                continue
        
        if len(candidate_tables) >= args.num_folders:
            break

    pbar.close()
    print(f"\nFound {len(candidate_tables)} tables total.")

    # --- MODIFIED SECTION ---
    # Sort by area descending
    candidate_tables.sort(key=lambda x: x[0], reverse=True)
    
    # Pick ranks 4, 5, and 6 (indices 3, 4, 5)
    top_tables = candidate_tables[3:6]
    # ------------------------

    print(f"Generating ICL examples for {len(top_tables)} tables (Ranks 4-6)...")

    icl_examples = []

    for i, (area, comp) in enumerate(top_tables):
        print(f"Generating example {i+1}/{len(top_tables)} (Area: {area})...")
        vlm_image = comp["path"]

        markdown_output = vlm.generate(vlmp.vlm_table_icl_markdown_prompt, vlm_image)
        json_output = vlm.generate(vlmp.vlm_table_icl_json_prompt, vlm_image)

        example = f"Input:\n{markdown_output}\n\nOutput:\n{json_output}"
        icl_examples.append(example)

    # Console output
    print("\n" + "=" * 50)
    print("GENERATED ICL EXAMPLES")
    print("=" * 50)

    for i, example in enumerate(icl_examples):
        print(f"\n--- Example {i+1} ---")
        print(example)
        print("\n" + "-" * 20)

    if not os.path.exists(f"icl/{args.dataset}"):
        os.makedirs(f"icl/{args.dataset}")

    vlm_name = args.vlm_model.split('/')[-1]

    with open(f"icl/{args.dataset}/{vlm_name}.json", "w") as f:
        json.dump(icl_examples, f, indent=2)

    print(f"\nSaved examples to icl/{args.dataset}/{vlm_name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folders", type=int, default=10, help="Number of folders to iterate over")
    parser.add_argument("--num_icl", type=int, default=3, help="Number of ICL examples to generate")

    parser.add_argument("--dataset", type=str, default="mpdocvqa", help="Dataset name") 
    # tatdqa, tablevqa, mpdocvqa, wikitablequestions, spiqa

    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-32B-Instruct") # modify
    parser.add_argument("--vlm_ip", type=str, default="146.169.26.172") # modify
    parser.add_argument("--vlm_port", type=str, default="3232") # modify

    # parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct") # modify
    # parser.add_argument("--vlm_ip", type=str, default="146.169.1.69") # modify
    # parser.add_argument("--vlm_port", type=str, default="6200") # modify

    args = parser.parse_args()
    main(args)