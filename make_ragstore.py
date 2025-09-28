import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath("object_detection"))

from src.llm import HFLLMClient, HFVLMClient, VLLMLLMClient, VLLMVLMClient
from src.prompts import VLMPrompts, LLMPrompts
from src.layout import LayoutProcessor
from src.vector_store import VectorStore
from src.embedder import SentenceTransformerEmbedder
from src.ragstore import Ragstore

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Model setup
# -----------------------------
llm = VLLMLLMClient('Qwen/Qwen3-14B', ip='146.169.1.69', port='1707')
vlm = VLLMVLMClient('Qwen/Qwen2.5-VL-32B-Instruct', ip='146.169.1.69', port='1708')

# Alternative clients if needed
# llm = HFLLMClient('Qwen/Qwen3-8B')
# vlm = HFVLMClient('Qwen/Qwen2.5-VL-7B-Instruct')

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')
llmp = LLMPrompts()
vlmp = VLMPrompts()
lp = LayoutProcessor()

# -----------------------------
# Base directory setup (general)
# -----------------------------
# base_dir = "datasets/fintabnet/expts"
base_dir = "datasets/tatdqa/test"
# base_dir = "datasets/mp-docvqa/jjvg0027"
# base_dir = "datasets/mp-docvqa/images_test" # for holding

# -----------------------------
# Recursive walk for any subfolder containing PDFs
# -----------------------------
for i, (root, subdirs, files) in enumerate(os.walk(base_dir)):
    # if i < 10:
    #     continue
    # print(root, subdirs, files)

    if any(f.lower().endswith((".pdf", ".jpg", ".jpeg")) for f in files):
        print(f"Processing directory: {root}")

        relative_path = os.path.relpath(root, base_dir)
        save_dir = os.path.join("storages/tatdqa", relative_path)

        rs = Ragstore(
            lp=lp,
            embedder=embedder,
            vlm=vlm,
            llm=llm,
            vlm_prompts=vlmp,
            llm_prompts=llmp,
            data_dir=root,
            save_dir=save_dir
        )
        # rs.build_index_for_folders()
        rs.build_index_per_file()
