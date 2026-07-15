import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore

GT_PATH = '/vol/bitbucket/mml324/TabRAG/datasets/finqa2/metadata.jsonl'
STORE_PATH = '/vol/bitbucket/mml324/TabRAG/storages/finqa2/retrieval/DeepSeek-OCR'

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

parser = argparse.ArgumentParser(description='Evaluate FinQA MRR')
parser.add_argument('--gt_path', type=str, default=GT_PATH)
parser.add_argument('--store_path', type=str, default=STORE_PATH)
args = parser.parse_args()

with open(args.gt_path, 'r') as f:
    gt = [json.loads(line) for line in f]

def fmt_gt(doc):
    name = doc.get('file_name', '')
    if name.endswith('.pdf'):
        name = name[:-4]
    name = name.replace('\\', '')
    if name.startswith('pdf/'):
        name = name[4:]
    return name  # e.g. "ABMD/2012/page_75"

def normalise(fname):
    return fname.rsplit('.', 1)[0].replace('/', '_')

# Detect storage style
is_flat = os.path.isfile(os.path.join(args.store_path, 'docstore.index'))
folder_paths = sorted([
    os.path.join(args.store_path, d)
    for d in os.listdir(args.store_path)
    if d.startswith('folder') and os.path.isdir(os.path.join(args.store_path, d))
    and os.path.isfile(os.path.join(args.store_path, d, 'docstore.index'))
])
has_folders = len(folder_paths) > 0

rr_list = []
skipped = 0

if is_flat:
    print(f"Flat index detected at {args.store_path}")
    index = VectorStore.load(os.path.join(args.store_path, 'docstore'))
    pages_in_index = set(normalise(m['file']) for m in index.metadata)
    print(f"Pages in index: {len(pages_in_index)}")

    for doc in tqdm(gt):
        doc_name = normalise(fmt_gt(doc))  # e.g. "ABMD_2012_page_75"
        if doc_name not in pages_in_index:
            skipped += 1
            continue

        q_embed = embedder.encode([doc.get('question', '')]).astype("float32")
        retrieved = index.search(q_embed, k=10)

        found = False
        for rank, r in enumerate(retrieved):
            if normalise(r['meta']['file']) == doc_name:
                rr_list.append(1 / (rank + 1))
                found = True
                break
        if not found:
            rr_list.append(0)

elif has_folders:
    # folder0/, folder1/, ... — merge all into one search pool
    print(f"Folder index detected ({len(folder_paths)} folders) at {args.store_path}")
    indexes = [VectorStore.load(os.path.join(fp, 'docstore')) for fp in folder_paths]
    pages_in_index = set(normalise(m['file']) for idx in indexes for m in idx.metadata)
    print(f"Pages in index: {len(pages_in_index)}")

    for doc in tqdm(gt):
        doc_name = normalise(fmt_gt(doc))
        if doc_name not in pages_in_index:
            skipped += 1
            continue

        q_embed = embedder.encode([doc.get('question', '')]).astype("float32")

        # Search all folders, collect top-10 across all, re-rank by score
        all_results = []
        for idx in indexes:
            all_results.extend(idx.search(q_embed, k=10))
        all_results.sort(key=lambda r: r['score'])  # L2: lower = better

        found = False
        for rank, r in enumerate(all_results[:10]):
            if normalise(r['meta']['file']) == doc_name:
                rr_list.append(1 / (rank + 1))
                found = True
                break
        if not found:
            rr_list.append(0)

else:
    # Sharded: TICKER/YEAR/page_N/docstore.index — search within company/year
    print(f"Sharded index detected at {args.store_path}")
    total_shards = sum(
        1 for ticker in os.listdir(args.store_path)
        for year in (os.listdir(os.path.join(args.store_path, ticker))
                     if os.path.isdir(os.path.join(args.store_path, ticker)) else [])
        for page in (os.listdir(os.path.join(args.store_path, ticker, year))
                     if os.path.isdir(os.path.join(args.store_path, ticker, year)) else [])
        if os.path.isfile(os.path.join(args.store_path, ticker, year, page, 'docstore.index'))
    )
    print(f"Pages in index: {total_shards}")

    for doc in tqdm(gt):
        doc_name = fmt_gt(doc)           # e.g. "ABMD/2012/page_75"
        page_shard = os.path.join(args.store_path, doc_name)

        if not os.path.isfile(os.path.join(page_shard, 'docstore.index')):
            skipped += 1
            continue

        company_year_dir = os.path.dirname(page_shard)  # .../ABMD/2012
        gt_page = os.path.basename(doc_name)             # page_75

        page_shards = [
            os.path.join(company_year_dir, p)
            for p in sorted(os.listdir(company_year_dir))
            if os.path.isfile(os.path.join(company_year_dir, p, 'docstore.index'))
        ]

        q_embed = embedder.encode([doc.get('question', '')]).astype("float32")

        scores = []
        for shard in page_shards:
            results = VectorStore.load(os.path.join(shard, 'docstore')).search(q_embed, k=1)
            if results:
                scores.append((results[0]['score'], os.path.basename(shard)))

        scores.sort(key=lambda x: x[0])  # L2: lower = better

        found = False
        for rank, (score, page_name) in enumerate(scores):
            if page_name == gt_page:
                rr_list.append(1 / (rank + 1))
                found = True
                break
        if not found:
            rr_list.append(0)

rr_arr = np.array(rr_list)
print(f'Evaluated {len(rr_list)} queries, skipped {skipped} (not in index)')
print(f'MRR = {np.mean(rr_arr):.4f}, stdev = {np.std(rr_arr):.4f}')
print(f'eval_finqaMRR = {np.mean(rr_arr)}, stdev = {np.std(rr_arr)} from {args.store_path}')
