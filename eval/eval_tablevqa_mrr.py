from tqdm import tqdm
import json
import numpy as np
import os
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore

JSON_PATH = '/vol/bitbucket/mml324/TabRAG/datasets/tablevqa/qa.json'
STORE_PATH = '/vol/bitbucket/mml324/TabRAG/storages/tablevqa/retrieval/gemma/Qwen3-VL-8B-Instruct'

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

with open(JSON_PATH, 'r') as f:
    gt = json.load(f)

rr_list = []
num_questions = 0

shards = [
    d for d in os.listdir(STORE_PATH)
    if d.startswith("folder") and os.path.isdir(os.path.join(STORE_PATH, d))
]


def shard_contains_doc(shard_path, page_id):
    jsonl_path = os.path.join(shard_path, "docstore_data.jsonl")
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            fname = obj["meta"]["file"]
            if page_id in fname or fname.startswith(page_id):
                return True
    return False


for item in tqdm(gt):

    qa_id = item.get('qa_id', '')
    q_text = item.get('question', '')

    parts = qa_id.split('.')
    page_id = parts[0]

    shard_path = None
    for s in shards:
        full_path = os.path.join(STORE_PATH, s)
        if shard_contains_doc(full_path, page_id):
            shard_path = full_path
            break

    if shard_path is None:
        print(f"WARNING: doc {page_id} not found in any shard")
        continue

    index = VectorStore.load(os.path.join(shard_path, "docstore"))

    num_questions += 1

    q_embed = embedder.encode([q_text]).astype("float32")

    retrieved = index.search(q_embed, k=10)

    found = False
    for rank, r in enumerate(retrieved):

        file_name = r["meta"]["file"]
        retrieved_page_id = file_name.split("_p")[0] + "_p" + file_name.split("_p")[1].split(".")[0] if "_p" in file_name else file_name.rsplit('.', 1)[0]
        print(f"Retrieved file: {file_name}, extracted page_id: {retrieved_page_id}, expected page_id: {page_id}")
        if page_id in file_name or file_name.startswith(page_id):
            rr_list.append(1/(rank+1))
            found = True
            break

    if not found:
        rr_list.append(0)

rr_arr = np.array(rr_list)
print("MRR =", np.mean(rr_arr))
print("stdev =", np.std(rr_arr))
print("num_questions=", num_questions)
print(f'eval_tablevqaMRR = {np.mean(rr_arr)}, stdev = {np.std(rr_arr)} from {STORE_PATH}')
