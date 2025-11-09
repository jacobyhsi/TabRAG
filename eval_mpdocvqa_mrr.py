from tqdm import tqdm
import json
import numpy as np
import os
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore

JSON_PATH = 'datasets/mpdocvqa/val.json'
STORE_PATH = 'storages/mpdocvqa/retrieval/tabrag'

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

with open(JSON_PATH, 'r') as f:
    gt = json.load(f)

rr_list = []
for question in tqdm(gt['data']):
    doc_id = question['doc_id']
    page_ids = question['page_ids']
    storage_path = os.path.join(STORE_PATH, f'{doc_id}')
    if not os.path.exists(storage_path):
        # print(f'ERROR: storage path {storage_path} not found')
        continue
    index = VectorStore.load(os.path.join(storage_path, 'docstore'))
    doc_q = question['question']
    q_embed = embedder.encode([doc_q]).astype("float32")
    retrieved_pages = index.search(q_embed, k=10)
    for rank, retrieved_page in enumerate(retrieved_pages):
        retrieved_page = retrieved_page['meta']['file'].split('.')[0]
        if retrieved_page in page_ids:
            rr_list.append(1/(rank + 1))
            break
    else:
        rr_list.append(0)
rr_arr = np.array(rr_list)
print(f'MRR = {np.mean(rr_arr)}, stdev = {np.std(rr_arr)}')