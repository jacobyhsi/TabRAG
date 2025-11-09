from tqdm import tqdm
import json
import numpy as np
import os
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore

JSON_PATH = 'datasets/spiqa/test-A/SPIQA_testA_wpage.json'
STORE_PATH = 'storages/spiqa/retrieval/tabrag'

embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')

with open(JSON_PATH, 'r') as f:
    gt = json.load(f)

paper_ids = ['1805.06431v4', '1805.04687v2', '1805.01216v3', '1906.06589v3', '1704.07121v2', '1707.08608v3', '1901.00398v2', '1811.02721v3', '1803.01128v3', '1707.01922v5', '1805.06447v3', '1706.00827v2', '1703.02507v3', '1812.00281v3', '1811.08481v2', '1803.04572v2', '1708.00160v2', '1901.00056v2', '1809.03550v3', '1805.00912v4', '1802.07351v2', '1706.00633v4', '1705.09296v2', '1611.04684v1', '1906.10843v1', '1811.09393v4', '1811.08257v1', '1809.04276v2', '1809.02731v3', '1804.05936v2', '1803.03467v4', '1709.08294v3', '1707.06320v2', '1707.01917v2', '1705.10667v4', '1705.02798v6', '1704.05426v4', '1611.03780v2', '1611.02654v2', '1812.10735v2']

rr_list = []
num_questions = 0
for doc_id in tqdm(gt.keys()):
    if doc_id not in paper_ids:
        continue
    figures = gt[doc_id]['all_figures']
    figure_map = {}
    for fig in figures.keys():
        figure_map[fig] = figures[fig]['page']
    
    storage_path = os.path.join(STORE_PATH, f'{doc_id}')
    if not os.path.exists(storage_path):
        # print(f'ERROR: storage path {storage_path} not found')
        continue
    index = VectorStore.load(os.path.join(storage_path, 'docstore'))
    for q in gt[doc_id]['qa']:
        doc_q = q['question']
        ref_q = q['reference']
        num_questions += 1

        q_embed = embedder.encode([doc_q]).astype("float32")
        retrieved_pages = index.search(q_embed, k=10)
        for rank, retrieved_page in enumerate(retrieved_pages):
            retrieved_pagenum = retrieved_page['meta']['page']
            # print(f'{doc_id}: {retrieved_pagenum}, {figure_map[ref_q]}')
            # exit()
            if retrieved_pagenum == figure_map[ref_q]:
                rr_list.append(1/(rank + 1))
                break
rr_arr = np.array(rr_list)
print(f'MRR = {np.mean(rr_arr)}, stdev = {np.std(rr_arr)}')
print(num_questions)