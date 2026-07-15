import os
import json
import argparse
from src.vector_store import VectorStore
from src.embedder import SentenceTransformerEmbedder, HFEmbedder, VLLMEmbedder

RETRIEVAL_DIR = '/vol/bitbucket/mml324/TabRAG/datasets/finqa/retrieval/'
STORAGE_DIR = '/vol/bitbucket/mml324/TabRAG/storages/finqa/generation/tabrag/Qwen3-VL-8B-Instruct'
STORAGE_OUT = 'storages/finqa/retrieval/tabrag/Qwen3-VL-8B-Instruct'

def collect_docs_from_storage(storage_dir, group):
    """Walk group dir and collect all docstore_data.jsonl entries.

    Handles both 2-level (group/page_id) and 3-level (group/year/page_id) layouts
    by scanning recursively for docstore_data.jsonl files.
    """
    data = []
    group_path = os.path.join(storage_dir, group)
    for dirpath, dirnames, filenames in os.walk(group_path):
        if "docstore_data.jsonl" not in filenames:
            continue
        jsonl_path = os.path.join(dirpath, "docstore_data.jsonl")
        rel_path = os.path.relpath(dirpath, storage_dir)
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc['meta']['file'] = rel_path
                    data.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    return data

def collect_docs_from_retrieval(ret_dir, storage_dir, group, has_subgroup):
    """Use retrieval/ dir file listing to find and collect docstore_data.jsonl entries."""
    data = []
    group_path = os.path.join(ret_dir, group)
    for filename in os.listdir(group_path):
        print(filename)
        file_id = filename.split('.')[0]
        if has_subgroup:
            jsonl_path = os.path.join(storage_dir, group, file_id, "docstore_data.jsonl")
            rel_path = os.path.join(group, file_id)
        else:
            jsonl_path = os.path.join(storage_dir, file_id, "docstore_data.jsonl")
            rel_path = file_id
        if not os.path.exists(jsonl_path):
            print(f"File not found: {jsonl_path}, skipping.")
            continue
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    doc['meta']['file'] = rel_path
                    data.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    return data

def embed_and_save(data, embedder, out_path):
    emb_dim = embedder.get_dims()
    index = VectorStore(emb_dim)
    output_data = []
    output_meta = []
    output_emb = []
    for doc in data:
        page_text = doc['text']
        page_meta = {
            "file": doc['meta']['file'],  # e.g. ABMD/2003/page_22
            "page": doc['meta']['page'],
            "mode": "page_vlm",
        }
        output_data.append(page_text)
        output_meta.append(page_meta)

        MAX_CHARS = 40960
        if len(page_text) > MAX_CHARS:
            print(f"Page text length {len(page_text)} exceeds {MAX_CHARS}, splitting into chunks.")

        chunks = [page_text[i : i + MAX_CHARS] for i in range(0, len(page_text), MAX_CHARS)] if page_text else [""]
        chunk_embs = [embedder.encode([chunk])[0] for chunk in chunks if chunk.strip()]

        if not chunk_embs:
            continue

        page_emb = sum(chunk_embs) / len(chunk_embs)
        output_emb.append(page_emb[None, :])

    for d, m, e in zip(output_data, output_meta, output_emb):
        index.add(e, d, m)
    os.makedirs(out_path, exist_ok=True)
    index.save(os.path.join(out_path, "docstore"))

def extract_and_transform_generation_jsonl(ret_dir, storage_dir, out_dir, has_subgroup, from_storage=False):
    embedder = VLLMEmbedder('Qwen/Qwen3-Embedding-8B', tensor_parallel_size=1, gpu_memory_utilization=0.8)

    all_data = []
    if from_storage:
        groups = os.listdir(storage_dir)
        print(f"Groups found in storage: {groups}")
        for group in groups:
            print(f"Processing group: {group}")
            all_data.extend(collect_docs_from_storage(storage_dir, group))
    else:
        print(os.listdir(ret_dir))
        for group in os.listdir(ret_dir):
            print(f"Processing group: {group}")
            all_data.extend(collect_docs_from_retrieval(ret_dir, storage_dir, group, has_subgroup))

    out_path = out_dir
    print(f"Embedding {len(all_data)} total docs into {out_path}")
    embed_and_save(all_data, embedder, out_path)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--retrieval_dir', type=str, default=RETRIEVAL_DIR)
    argparse.add_argument('--storage_dir', type=str, default=STORAGE_DIR)
    argparse.add_argument('--storage_out', type=str, default=STORAGE_OUT)
    argparse.add_argument('--has_subgroup', action='store_true', default=False)
    argparse.add_argument('--from_storage', action='store_true', default=False)
    args = argparse.parse_args()
    extract_and_transform_generation_jsonl(args.retrieval_dir, args.storage_dir, args.storage_out, args.has_subgroup, args.from_storage)
