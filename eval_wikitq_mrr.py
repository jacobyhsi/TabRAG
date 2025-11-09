from tqdm import tqdm
import json
import numpy as np
import os
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore

JSON_PATH = 'datasets/wikitablequestions/qa.json'
STORE_PATH = 'storages/wikitablequestions/retrieval/pymupdf'

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

with open(JSON_PATH, 'r') as f:
    gt = json.load(f)

rr_list = []
num_questions = 0
for q in tqdm(gt):
    num_questions += 1
    doc_q = q['utterance']
    store_q = q['context'].split('/')[-2]
    ref_q = q['context'].split('/')[-1]
    ref_q = ref_q.split('.csv')[0]
    
    storage_path = os.path.join(STORE_PATH, f'{store_q}')
    if not os.path.exists(storage_path):
        continue
    index = VectorStore.load(os.path.join(storage_path, 'docstore'))

    q_embed = embedder.encode([doc_q]).astype("float32")
    retrieved_pages = index.search(q_embed, k=10)
    for rank, retrieved_page in enumerate(retrieved_pages):
        retrieved_pagenum = retrieved_page['meta']['page']
        # print(f'{retrieved_pagenum}, {ref_q}')
        if retrieved_pagenum == int(ref_q):
            rr_list.append(1/(rank + 1))
            break
    else:
        rr_list.append(0)
rr_arr = np.array(rr_list)
print(f'MRR = {np.mean(rr_arr)}, stdev = {np.std(rr_arr)}')
print(num_questions)