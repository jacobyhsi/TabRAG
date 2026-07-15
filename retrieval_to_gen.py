import os
import json
import argparse
import faiss

from src.vector_store import VectorStore

# Splits a "retrieval" store (one docstore per group, containing all pages)
# back out into a "generation"-style store (one docstore per page/item),
# mirroring the folder layout used by e.g.
# storages/mpdocvqa/generation/vlm/Qwen3-VL-8B-Instruct.
#
# Embeddings are reused directly from the retrieval FAISS index (via
# reconstruct) rather than being recomputed, since the retrieval store was
# built from the same page_vlm text.

RETRIEVAL_DIR = '/home/js2723/PROJECTS/TabRAG/storages/tablevqa/retrieval/vlm/Qwen3-VL-32B-Instruct'
GENERATION_DIR = '/home/js2723/PROJECTS/TabRAG/storages/tablevqa/generation/vlm/Qwen3-VL-32B-Instruct'


def item_name_from_meta(meta):
    """Derive the per-item folder name from a doc's meta['file'], e.g.
    'flpp0227_p14.jpg' -> 'flpp0227_p14', '105377.png' -> '105377'."""
    file_field = meta['file']
    return os.path.splitext(file_field)[0]


def split_group(retrieval_dir, generation_dir, group, flatten_groups=False):
    group_path = os.path.join(retrieval_dir, group)
    jsonl_path = os.path.join(group_path, "docstore_data.jsonl")
    index_path = os.path.join(group_path, "docstore.index")

    if not (os.path.exists(jsonl_path) and os.path.exists(index_path)):
        print(f"Skipping {group}: missing docstore files")
        return

    index = faiss.read_index(index_path)
    dim = index.d

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()]

    for i, doc in enumerate(docs):
        item_name = item_name_from_meta(doc['meta'])
        if flatten_groups:
            out_path = os.path.join(generation_dir, item_name)
        else:
            out_path = os.path.join(generation_dir, group, item_name)
        os.makedirs(out_path, exist_ok=True)

        emb = index.reconstruct(i)[None, :]
        store = VectorStore(dim)
        store.add(emb, [doc['text']], [doc['meta']])
        store.save(os.path.join(out_path, "docstore"))


def split_retrieval_to_generation(retrieval_dir, generation_dir, flatten_groups=False):
    groups = sorted(os.listdir(retrieval_dir))
    print(f"Found {len(groups)} groups in {retrieval_dir}")
    for group in groups:
        print(f"Processing group: {group}")
        split_group(retrieval_dir, generation_dir, group, flatten_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_dir', type=str, default=RETRIEVAL_DIR)
    parser.add_argument('--generation_dir', type=str, default=GENERATION_DIR)
    parser.add_argument(
        '--flatten_groups',
        action='store_true',
        help='write each item directly under generation_dir instead of preserving shard folders',
    )
    args = parser.parse_args()
    split_retrieval_to_generation(
        args.retrieval_dir, args.generation_dir, args.flatten_groups
    )
