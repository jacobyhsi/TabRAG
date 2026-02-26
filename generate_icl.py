import sys
import os
import json
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from transformers import AutoTokenizer

random.seed(1)

# Add object_detection to path so src.layout works
sys.path.append(os.path.abspath("object_detection"))

from src.llm import VLLMVLMClient, HFVLMClient
from src.prompts import VLMPrompts
from src.layout import LayoutProcessor


# --------------------------------------------------
# Utils
# --------------------------------------------------
def collect_image_paths(data_dir):
    """
    Collect all image paths under:
    generation/<doc>/<doc_page>/<doc_page>.jpg
    """
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    # ----------------------------
    # Initialize models
    # ----------------------------
    print("Initializing models...")
    try:
        lp = LayoutProcessor()
        vlmp = VLMPrompts()
        model = args.model
        if args.use_vllm is None and args.use_hf is None and args.use_openai is None:
            print("Error: You must use one of HuggingFace or VLLM as a VLM inference provider.")
            return
        if args.use_hf:
            vlm = HFVLMClient(model)
        elif args.use_vllm:
            vlm = VLLMVLMClient(model, ip=args.vllm_ip, port=args.vllm_port)
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    data_dir = f"datasets/{args.dataset}/generation"
    candidate_tables = []  # (area, component)

    # ----------------------------
    # Collect images
    # ----------------------------
    image_paths = collect_image_paths(data_dir)
    random.shuffle(image_paths)

    print(f"Found {len(image_paths)} images under {data_dir}")
    print(f"Scanning for {args.num_folders} files containing tables...")

    pbar = tqdm(total=args.num_folders, desc="Tables Found")

    # ----------------------------
    # Detect tables
    # ----------------------------
    for file_path in image_paths:
        if len(candidate_tables) >= args.num_folders:
            break

        try:
            comps = lp.preprocess(file_path)

            largest_table = None
            max_area = -1

            for _, components in comps.items():
                for comp in components:
                    if comp["class"] == 3:  # table class
                        bbox = comp["bbox"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                        if area > max_area:
                            max_area = area
                            largest_table = comp

            if largest_table is not None:
                candidate_tables.append((max_area, largest_table))
                pbar.update(1)

        except Exception:
            continue

    pbar.close()
    print(f"\nFound {len(candidate_tables)} tables total.")

    # ----------------------------
    # Rank tables by area
    # ----------------------------
    candidate_tables.sort(key=lambda x: x[0], reverse=True)

    # ----------------------------
    # Token-budget-aware selection
    # ----------------------------
    MAX_TOKENS = 5000
    MIN_TOKENS = 100
    K = args.num_icl

    print(
        f"Selecting top-{K} tables with "
        f"{MIN_TOKENS} ≤ tokens per field < {MAX_TOKENS} total..."
    )

    selected_examples = []
    cursor = 0

    while cursor < len(candidate_tables) and len(selected_examples) < K:
        area, comp = candidate_tables[cursor]

        try:
            vlm_image = comp["path"]

            markdown_output = vlm.generate(
                vlmp.vlm_table_icl_markdown_prompt, vlm_image
            )
            json_output = vlm.generate(
                vlmp.vlm_table_prompt, vlm_image
            )
            md_tokens = count_tokens(tokenizer, markdown_output)
            json_tokens = count_tokens(tokenizer, json_output)
            total_tokens = md_tokens + json_tokens

            combined_text = (
                "Input:\n"
                f"{markdown_output}\n\n"
                "Output:\n"
                f"{json_output}"
            )

            if md_tokens < MIN_TOKENS:
                print(
                    f"Skipped (markdown too short: "
                    f"{md_tokens} < {MIN_TOKENS})"
                )

            elif json_tokens < MIN_TOKENS:
                print(
                    f"Skipped (json too short: "
                    f"{json_tokens} < {MIN_TOKENS})"
                )

            elif md_tokens >= MAX_TOKENS:
                print(
                    f"Skipped (markdown tokens {md_tokens} "
                    f">= {MAX_TOKENS})"
                )

            elif json_tokens >= MAX_TOKENS:
                print(
                    f"Skipped (json tokens {json_tokens} "
                    f">= {MAX_TOKENS})"
                )

            else:
                selected_examples.append(combined_text)
                print(
                    f"Accepted (md={md_tokens}, json={json_tokens}, "
                    f"total={total_tokens}, area={area})"
                )

        except Exception:
            pass

        cursor += 1

    if len(selected_examples) < K:
        print(
            f"Warning: only {len(selected_examples)} valid "
            f"ICL examples found."
        )

    # ----------------------------
    # Print results
    # ----------------------------
    print("\n" + "=" * 50)
    print("GENERATED ICL EXAMPLES")
    print("=" * 50)

    for i, example in enumerate(selected_examples):
        print(f"\n--- Example {i + 1} ---")
        print(example)
        print("\n" + "-" * 20)

    # ----------------------------
    # Save to disk
    # ----------------------------
    save_dir = f"icl/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    vlm_name = model.split("/")[-1]
    save_path = f"{save_dir}/{vlm_name}.json"

    with open(save_path, "w") as f:
        json.dump(selected_examples, f, indent=2)

    print(f"\nSaved examples to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folders", type=int, default=20, help="Number of folders to iterate over")
    parser.add_argument("--num_icl", type=int, default=3, help="Number of ICL examples to generate")

    parser.add_argument("--dataset", type=str, default="tatdqa", help="Dataset name") 
    # tatdqa, tablevqa, mpdocvqa, wikitablequestions, spiqa, comtqa
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    parser.add_argument("--use_hf", action='store_true')

    parser.add_argument("--use_vllm", action='store_true')
    parser.add_argument("--vllm_ip", type=str, default="146.169.26.172") # modify
    parser.add_argument("--vllm_port", type=str, default="1111") # modify

    args = parser.parse_args()
    main(args)