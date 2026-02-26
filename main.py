import argparse
import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.abspath("object_detection"))

from src.llm import HFLLMClient, HFVLMClient, VLLMLLMClient, VLLMVLMClient
from src.prompts import VLMPrompts, LLMPrompts
from src.layout import LayoutProcessor
from src.vector_store import VectorStore
from src.embedder import SentenceTransformerEmbedder, HFEmbedder, VLLMEmbedder
from src.ragstore import Ragstore

import warnings
warnings.filterwarnings("ignore")

def main(args):
    model = args.model
    mode = args.mode
    dataset = args.dataset
    # -----------------------------
    # Model setup
    # -----------------------------

    # Embedder
    if args.use_vllm:
        embedder = VLLMEmbedder(args.embedder, tensor_parallel_size=1, gpu_memory_utilization=0.6)
        vlm = VLLMVLMClient(args.vlm, ip=args.vllm_ip, port=args.vllm_port)
    if args.use_hf:
        embedder = HFEmbedder(args.embedder)
        vlm = HFVLMClient(args.vlm)

    # Initialize Model
    vlmp = VLMPrompts()
    lp = LayoutProcessor()
    vlm_name = args.vlm.split('/')[-1]

    icl_path = f"icl/{args.dataset}/{vlm_name}.json"
    with open(icl_path, "r") as f:
        icl = json.load(f)

    # -----------------------------
    # Base directory setup
    # -----------------------------
    data_dir = f"datasets/{dataset}/{mode}"

    # ----------------------------------
    # Recursive walk through directories
    # ----------------------------------
    for i, (root, subdirs, files) in enumerate(os.walk(data_dir)):
        subdirs.sort()
        files.sort()

        if any(f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")) for f in files):
            print(f"Processing directory: {root}")

            relative_path = os.path.relpath(root, data_dir)
            save_dir = os.path.join(f"storages/{dataset}/{mode}/{model}/{vlm_name}", relative_path)

            rs = Ragstore(
                lp=lp,
                embedder=embedder,
                vlm=vlm,
                vlm_prompts=vlmp,
                icl=icl,
                model=model,
                data_dir=root,
                save_dir=save_dir
            )
            rs.build_index_per_folder()
            print("Saved to:", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tabrag", help="e.g. tabrag, pymupdf, pytesseract, vlm")
    parser.add_argument("--mode", type=str,  default="generation", help="generation or retrieval")
    parser.add_argument("--embedder", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-8B-Instruct") # modify

    parser.add_argument("--use_hf", action='store_true')

    parser.add_argument("--use_vllm", action='store_true')
    parser.add_argument("--vllm_ip", type=str, default="localhost") # modify
    parser.add_argument("--vllm_port", type=str, default="2222") # modify

    parser.add_argument("--dataset", type=str,  default="tatdqa", help="tatdqa, mpdocvqa, wikitablequestions, spiqa, tablevqa, comtqa")

    args = parser.parse_args()
    main(args)
