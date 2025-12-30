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
    # VLLM
    # vlm = VLLMVLMClient('Qwen/Qwen3-VL-8B-Instruct', ip='146.169.1.69', port='6200')
    # vlm = VLLMVLMClient('Qwen/Qwen3-VL-8B-Instruct', ip='146.169.1.172', port='3232')
    # vlm = VLLMVLMClient('Qwen/Qwen3-VL-8B-Instruct', ip='localhost', port='3232')
    vlm = VLLMVLMClient('Qwen/Qwen3-VL-32B-Instruct', ip='localhost', port='3232')
    llm = VLLMLLMClient('Qwen/Qwen3-14B', ip='146.169.1.68', port='1707')

    # HuggingFace
    # llm = HFLLMClient('Qwen/Qwen3-8B')
    # vlm = HFVLMClient('Qwen/Qwen2.5-VL-7B-Instruct')

    # Embedder
    embedder = VLLMEmbedder('Qwen/Qwen3-Embedding-8B', tensor_parallel_size=1, gpu_memory_utilization=0.7)
    # embedder = HFEmbedder('Qwen/Qwen3-Embedding-8B')
    # embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')
    # embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-4B')

    # Initialize Model
    llmp = LLMPrompts()
    vlmp = VLMPrompts()
    lp = LayoutProcessor()

    icl_path = f"icl/{args.dataset}_icl.json"
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
            save_dir = os.path.join(f"storages/{dataset}/{mode}/{model}", relative_path)

            rs = Ragstore(
                lp=lp,
                embedder=embedder,
                vlm=vlm,
                llm=llm,
                vlm_prompts=vlmp,
                llm_prompts=llmp,
                icl=icl,
                model=model,
                data_dir=root,
                save_dir=save_dir
            )
            # rs.build_index_per_file()
            rs.build_index_per_folder()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tabrag", help="e.g. tabrag, pymupdf, pytesseract, vlm")
    parser.add_argument("--mode", type=str,  default="generation", help="generation or retrieval")
    parser.add_argument("--dataset", type=str,  default="tatdqa", help="tatdqa, mpdocvqa, wikitablequestions, spiqa, tablevqa")
    args = parser.parse_args()
    main(args)