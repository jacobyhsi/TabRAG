import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore


#############
# Retrieval MRR evaluation
#
# Consolidates the per-dataset eval_*_mrr.py scripts into one entry point.
# Each eval_<dataset>_mrr(gt_path, store_path, embedder) returns (rr_arr, num_questions).
#############

# Default ground-truth files (local datasets/ layout, matching eval_generation.py)
DEFAULT_GT = {
    'mpdocvqa': 'datasets/mpdocvqa/val.json',
    'tatdqa':   'datasets/tatdqa/tatdqa_dataset_test_gold.json',
    'wikitq':   'datasets/wikitablequestions/qa.json',
    'tablevqa': 'datasets/tablevqa/qa.json',
    'comtqa':   'datasets/comtqa/qa3.json',
    'finqa':    'datasets/finqa/metadata.jsonl',
}


def _list_folder_shards(store_path):
    """Return the folder0/, folder1/, ... shard subdirs under store_path (may be empty)."""
    return [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]


def _find_shard(store_path, shards, matches):
    """Return the first shard whose docstore_data.jsonl has a meta.file satisfying matches(fname)."""
    for s in shards:
        full_path = os.path.join(store_path, s)
        jsonl_path = os.path.join(full_path, "docstore_data.jsonl")
        if not os.path.isfile(jsonl_path):
            continue
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if matches(obj["meta"]["file"]):
                    return full_path
    return None


###############
# Per-dataset evaluators
###############

def eval_wikitq_mrr(gt_path, store_path, embedder):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    rr_list = []
    num_questions = 0
    for q in tqdm(gt, desc="wikitq MRR"):
        num_questions += 1
        doc_q = q['utterance']
        store_q = q['context'].split('/')[-2]
        ref_q = q['context'].split('/')[-1].split('.csv')[0]

        storage_path = os.path.join(store_path, f'{store_q}')
        if not os.path.exists(storage_path):
            continue
        index = VectorStore.load(os.path.join(storage_path, 'docstore'))

        q_embed = embedder.encode([doc_q]).astype("float32")
        retrieved_pages = index.search(q_embed, k=10)
        for rank, retrieved_page in enumerate(retrieved_pages):
            if retrieved_page['meta']['page'] == int(ref_q):
                rr_list.append(1 / (rank + 1))
                break
        else:
            rr_list.append(0)

    return np.array(rr_list), num_questions


def eval_mpdocvqa_mrr(gt_path, store_path, embedder):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    rr_list = []
    for question in tqdm(gt['data'], desc="mpdocvqa MRR"):
        doc_id = question['doc_id']
        page_ids = question['page_ids']

        storage_path = os.path.join(store_path, f'{doc_id}')
        if not os.path.exists(storage_path):
            continue
        index = VectorStore.load(os.path.join(storage_path, 'docstore'))

        q_embed = embedder.encode([question['question']]).astype("float32")
        retrieved_pages = index.search(q_embed, k=10)
        for rank, retrieved_page in enumerate(retrieved_pages):
            retrieved_page = retrieved_page['meta']['file'].split('.')[0]
            if retrieved_page in page_ids:
                rr_list.append(1 / (rank + 1))
                break
        else:
            rr_list.append(0)

    return np.array(rr_list), len(rr_list)


def eval_comtqa_mrr(gt_path, store_path, embedder):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    shards = _list_folder_shards(store_path)
    rr_list = []
    num_questions = 0

    for item in tqdm(gt, desc="comtqa MRR"):
        if 'image_name' in item:
            doc_id = item['image_name'].replace('.jpg', '')
        else:
            doc_id = str(item['table_id'])

        shard_path = _find_shard(
            store_path, shards,
            lambda fname: fname.startswith(doc_id + "_p") or fname.startswith(doc_id + "."),
        )
        if shard_path is None:
            print(f"doc {doc_id} not found in any shard")
            continue

        index = VectorStore.load(os.path.join(shard_path, "docstore"))
        num_questions += 1

        q_embed = embedder.encode([item['question']]).astype("float32")
        retrieved = index.search(q_embed, k=10)

        for rank, r in enumerate(retrieved):
            file_name = r["meta"]["file"]
            if "_p" in file_name:
                pred_id = file_name.split("_p")[0]
            else:
                pred_id = file_name.rsplit('.', 1)[0]
            if pred_id == doc_id:
                rr_list.append(1 / (rank + 1))
                break
        else:
            rr_list.append(0)

    return np.array(rr_list), num_questions


def eval_tablevqa_mrr(gt_path, store_path, embedder):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    shards = _list_folder_shards(store_path)
    rr_list = []
    num_questions = 0

    for item in tqdm(gt, desc="tablevqa MRR"):
        qa_id = item.get('qa_id', '')
        page_id = qa_id.split('.')[0]

        shard_path = _find_shard(
            store_path, shards,
            lambda fname: page_id in fname or fname.startswith(page_id),
        )
        if shard_path is None:
            print(f"WARNING: doc {page_id} not found in any shard")
            continue

        index = VectorStore.load(os.path.join(shard_path, "docstore"))
        num_questions += 1

        q_embed = embedder.encode([item.get('question', '')]).astype("float32")
        retrieved = index.search(q_embed, k=10)

        for rank, r in enumerate(retrieved):
            file_name = r["meta"]["file"]
            if page_id in file_name or file_name.startswith(page_id):
                rr_list.append(1 / (rank + 1))
                break
        else:
            rr_list.append(0)

    return np.array(rr_list), num_questions


def eval_tatdqa_mrr(gt_path, store_path, embedder):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    shards = _list_folder_shards(store_path)
    rr_list = []
    num_questions = 0

    for item in tqdm(gt, desc="tatdqa MRR"):
        doc_id = item["doc"]["uid"]
        ref_page = int(item["doc"]["page"])

        shard_path = _find_shard(
            store_path, shards,
            lambda fname: fname.startswith(doc_id + "_p"),
        )
        if shard_path is None:
            print(f"WARNING: doc {doc_id} not found in any shard")
            continue

        index = VectorStore.load(os.path.join(shard_path, "docstore"))

        for q in item["questions"]:
            num_questions += 1
            q_embed = embedder.encode([q["question"]]).astype("float32")
            retrieved = index.search(q_embed, k=10)

            for rank, r in enumerate(retrieved):
                file_name = r["meta"]["file"]
                pred_id = file_name.split("_p")[0]
                pred_page = int(file_name.split("_p")[1].split(".")[0])
                if pred_id == doc_id and pred_page == ref_page:
                    rr_list.append(1 / (rank + 1))
                    break
            else:
                rr_list.append(0)

    return np.array(rr_list), num_questions


def eval_finqa_mrr(gt_path, store_path, embedder):
    """FinQA supports three storage layouts, auto-detected:
       - flat:    store_path/docstore.index
       - folders: store_path/folder0, folder1, ...   (merged into one search pool)
       - sharded: store_path/TICKER/YEAR/page_N/docstore.index (searched per company/year)
    """
    with open(gt_path, 'r') as f:
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

    is_flat = os.path.isfile(os.path.join(store_path, 'docstore.index'))
    folder_paths = sorted([
        os.path.join(store_path, d)
        for d in os.listdir(store_path)
        if d.startswith('folder') and os.path.isdir(os.path.join(store_path, d))
        and os.path.isfile(os.path.join(store_path, d, 'docstore.index'))
    ])
    has_folders = len(folder_paths) > 0

    rr_list = []
    skipped = 0

    if is_flat:
        print(f"Flat index detected at {store_path}")
        index = VectorStore.load(os.path.join(store_path, 'docstore'))
        pages_in_index = set(normalise(m['file']) for m in index.metadata)
        print(f"Pages in index: {len(pages_in_index)}")

        for doc in tqdm(gt, desc="finqa MRR (flat)"):
            doc_name = normalise(fmt_gt(doc))
            if doc_name not in pages_in_index:
                skipped += 1
                continue

            q_embed = embedder.encode([doc.get('question', '')]).astype("float32")
            retrieved = index.search(q_embed, k=10)

            for rank, r in enumerate(retrieved):
                if normalise(r['meta']['file']) == doc_name:
                    rr_list.append(1 / (rank + 1))
                    break
            else:
                rr_list.append(0)

    elif has_folders:
        print(f"Folder index detected ({len(folder_paths)} folders) at {store_path}")
        indexes = [VectorStore.load(os.path.join(fp, 'docstore')) for fp in folder_paths]
        pages_in_index = set(normalise(m['file']) for idx in indexes for m in idx.metadata)
        print(f"Pages in index: {len(pages_in_index)}")

        for doc in tqdm(gt, desc="finqa MRR (folders)"):
            doc_name = normalise(fmt_gt(doc))
            if doc_name not in pages_in_index:
                skipped += 1
                continue

            q_embed = embedder.encode([doc.get('question', '')]).astype("float32")

            all_results = []
            for idx in indexes:
                all_results.extend(idx.search(q_embed, k=10))
            all_results.sort(key=lambda r: r['score'])  # L2: lower = better

            for rank, r in enumerate(all_results[:10]):
                if normalise(r['meta']['file']) == doc_name:
                    rr_list.append(1 / (rank + 1))
                    break
            else:
                rr_list.append(0)

    else:
        print(f"Sharded index detected at {store_path}")
        total_shards = sum(
            1 for ticker in os.listdir(store_path)
            for year in (os.listdir(os.path.join(store_path, ticker))
                         if os.path.isdir(os.path.join(store_path, ticker)) else [])
            for page in (os.listdir(os.path.join(store_path, ticker, year))
                         if os.path.isdir(os.path.join(store_path, ticker, year)) else [])
            if os.path.isfile(os.path.join(store_path, ticker, year, page, 'docstore.index'))
        )
        print(f"Pages in index: {total_shards}")

        for doc in tqdm(gt, desc="finqa MRR (sharded)"):
            doc_name = fmt_gt(doc)
            page_shard = os.path.join(store_path, doc_name)

            if not os.path.isfile(os.path.join(page_shard, 'docstore.index')):
                skipped += 1
                continue

            company_year_dir = os.path.dirname(page_shard)
            gt_page = os.path.basename(doc_name)

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

            for rank, (score, page_name) in enumerate(scores):
                if page_name == gt_page:
                    rr_list.append(1 / (rank + 1))
                    break
            else:
                rr_list.append(0)

    print(f'Evaluated {len(rr_list)} queries, skipped {skipped} (not in index)')
    return np.array(rr_list), len(rr_list)


EVALUATORS = {
    'mpdocvqa': eval_mpdocvqa_mrr,
    'tatdqa':   eval_tatdqa_mrr,
    'wikitq':   eval_wikitq_mrr,
    'tablevqa': eval_tablevqa_mrr,
    'comtqa':   eval_comtqa_mrr,
    'finqa':    eval_finqa_mrr,
}


def main(args):
    dataset = args.dataset
    method = args.method
    vlm_model = args.vlm_model

    if dataset not in EVALUATORS:
        print(f"[Error] Unknown dataset '{dataset}'. Choices: {sorted(EVALUATORS)}")
        return

    gt_path = args.gt_path or DEFAULT_GT[dataset]
    store_path = args.store_path or f'storages/{dataset}/retrieval/{method}/{vlm_model}'

    if not os.path.exists(store_path):
        print(f"[Error] Store path does not exist: {store_path}")
        return

    embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

    rr_arr, num_questions = EVALUATORS[dataset](gt_path, store_path, embedder)

    mrr = float(np.mean(rr_arr)) if len(rr_arr) else 0.0
    stdev = float(np.std(rr_arr)) if len(rr_arr) else 0.0
    print(f'eval_{dataset}MRR = {mrr}, stdev = {stdev} from {store_path}')
    print(f'MRR = {mrr:.4f}, stdev = {stdev:.4f}, num_questions = {num_questions}, evaluated = {len(rr_arr)}')

    record = {
        "dataset": dataset,
        "method": method,
        "vlm_model": vlm_model,
        "store_path": store_path,
        "result": {
            "mrr": mrr,
            "stdev": stdev,
            "num_questions": num_questions,
            "n_evaluated": int(len(rr_arr)),
        },
    }

    # Save alongside generation results, if store_path is under storages/
    try:
        rel = os.path.relpath(store_path, "storages")
        if not rel.startswith(".."):
            results_path = os.path.join("results", rel, "mrr.json")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            print(f"Saved results to {results_path}")
    except ValueError:
        pass  # store_path on a different drive than storages/; skip saving


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Retrieval MRR Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='wikitq',
                        help='dataset to evaluate on: ' + ', '.join(sorted(EVALUATORS)))
    parser.add_argument('--method', type=str, default='tabrag',
                        help='method/baseline, e.g. tabrag, vlm, pymupdf')
    parser.add_argument('--vlm_model', type=str, default='Qwen3-VL-8B-Instruct',
                        help='VLM model: Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, gpt-5.2, ...')
    parser.add_argument('--store_path', type=str, default=None,
                        help='override for the retrieval store dir '
                             '(default: storages/{dataset}/retrieval/{method}/{vlm_model})')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='override for the ground-truth file (default: DEFAULT_GT[dataset])')
    args = parser.parse_args()
    main(args)
