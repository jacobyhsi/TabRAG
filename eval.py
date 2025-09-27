import argparse
import json
import os
import re
from src.llm import HFLLMClient, VLLMLLMClient
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStore
from src.utils import normalize_answer

def preprocess_mpdocvqa(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(data.keys())
    data = data['data']
    print(data)

# def eval_TATDQA(gt_path, dataset_dir, storage_dir, llm, emb, metric):
#     with open(gt_path, 'r') as f:
#         gt = json.load(f)

#     correct = 0 # accuracy calc
#     rr = 0 # reciprocal rank calc
#     total = 0
#     mistakes = []
#     for doc in gt:
#         doc_doc = doc['doc']
#         doc_id = doc['doc']['uid']
#         doc_src = doc['doc']['source']
#         with open(os.path.join(dataset_dir, doc_id + '.json'), 'r') as f:
#             doc_meta = json.load(f)

#         storage_path = os.path.join(storage_dir, doc_id)
#         if not os.path.exists(storage_path):
#             # print(f'ERROR: {storage_path} not found')
#             continue
#         index = VectorStore.load(os.path.join(storage_path, 'docstore'))

#         doc_qs = doc['questions']
#         for q in doc_qs:
#             q_q = q['question']
#             q_a = q['answer']
#             q_embed = emb.encode([q_q]).astype("float32")
#             gt_components = q['block_mapping']
#             if len(gt_components) == 0: # incomplete information
#                 continue
#             total += 1

#             # QA interaction
#             if metric == 'acc' or metric is None:
#                 retr = index.search(q_embed, k=1)
#                 system_prompt = f"""You are a helpful assistant. Use the information from the documents below to answer the question."""
#                 user_prompt = f"""/no_think {retr[0]['text']} \n Question: {q} \n Answer: """

#                 raw_response = llm.generate(system_prompt, user_prompt)
#                 raw_response = output_sanitizer(raw_response).lower() # clean up <think> tags and change to lower case
#                 raw_response = re.sub(r'(?<=\d),(?=\d)', '', raw_response) # regex match number formatting to remove commas
#                 print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

#                 if not isinstance(q_a, list): # ensure q_a is a list even even if there is only one entry
#                     q_a = [q_a]

#                 if all(str(ans).lower() in raw_response for ans in q_a):
#                     correct += 1
#                     print("correct!")
#                 else:
#                     mistakes.append((q_q, q_a, raw_response))
#                     print("incorrect!")
#                 print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')

#             if metric == 'mrr10' or metric is None:
#                 # retrieve bounding boxes of ground truth answers
#                 gt_component_ids = [list(comp.keys())[0] for comp in gt_components]
#                 gt_component_bboxes = []
#                 for page in doc_meta['pages']:
#                     for block in page['blocks']:
#                         if block['uuid'] in gt_component_ids:
#                             gt_component_bboxes.append([coord / 2 for coord in block['bbox']])

#                 gt_component_bbox = gt_component_bboxes[0]
#                 gt_component_bbox[0], gt_component_bbox[1] = gt_component_bbox[0] - 5, gt_component_bbox[1] - 5
#                 gt_component_bbox[2], gt_component_bbox[3] = gt_component_bbox[2] + 5, gt_component_bbox[3] + 5
#                 gt_component_area = (gt_component_bbox[2] - gt_component_bbox[0]) * (gt_component_bbox[3] - gt_component_bbox[1])
                
#                 # for MRR only pay attention to the rank of one retrieval
#                 ret_components = index.search(q_embed, k=10)
#                 for rank, ret_component in enumerate(ret_components):
#                     ret_component_bbox = ret_component['meta']['bbox']
#                     ret_component_score = ret_component['score']
#                     inter_x0 = max(ret_component_bbox[0], gt_component_bbox[0])
#                     inter_y0 = max(ret_component_bbox[1], gt_component_bbox[1])
#                     inter_x1 = min(ret_component_bbox[2], gt_component_bbox[2])
#                     inter_y1 = min(ret_component_bbox[3], gt_component_bbox[3])
#                     inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
#                     if (inter_area > 0.1 * gt_component_area):
#                         print(f'MATCH: rank {rank + 1}, {ret_component_bbox} | {ret_component_score}')
#                         # print(ret_component['text'])
#                         rr += 1/(rank + 1)
#                         break
#     results = {}
#     results['mrr@10'] = rr / total
#     results['accuracy'] = correct / total
#     return results


def eval_TATDQA(gt_path, dataset_dir, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for doc in gt:
        doc_doc = doc['doc']
        doc_id = doc['doc']['uid']
        doc_src = doc['doc']['source']
        page = doc_doc.get("page")

        folder_name = f"{doc_id}_p{page}" if page is not None else doc_id
        storage_path = os.path.join(storage_dir, folder_name)

        if not os.path.isdir(storage_path):
            continue

        docstore_path = os.path.join(storage_path, "docstore")

        index = VectorStore.load(docstore_path)

        doc_qs = doc['questions']
        for q in doc_qs:
            q_q = q['question']
            q_a = q['answer']
            q_embed = emb.encode([q_q]).astype("float32")
            gt_components = q['block_mapping']
            if len(gt_components) == 0: # incomplete information
                continue
            total += 1

            # QA interaction
            if metric == 'acc' or metric is None:
                retr = index.search(q_embed, k=1)
                system_prompt = f"""You are a helpful assistant. Use the information from the documents below to answer the question."""
                user_prompt = f"""/no_think {retr[0]['text']} \n Question: {q} \n Answer: """

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                raw_response = normalize_answer(raw_response)

                if not isinstance(q_a, list): # ensure q_a is a list even even if there is only one entry
                    q_a = [q_a]

                q_a = [normalize_answer(str(ans)) for ans in q_a]

                if all(ans in raw_response for ans in q_a):
                    correct += 1
                    print("correct!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("incorrect!")
                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')

          
    results = {}
    results['accuracy'] = correct / total
    return results

def main(args):
    dataset = args.dataset
    metric = args.metric
    llm = HFLLMClient('Qwen/Qwen3-8B')
    # llm = VLLMLLMClient('Qwen/Qwen3-8B')
    # llm = None
    embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')
    # embedder = None
    if (dataset == 'MP-DocVQA'):
        if (args.dataset_path is None):
            print('MP-DocVQA: Missing --dataset_path argument')
            exit()
        preprocess_mpdocvqa(args.dataset_path)
    if (dataset == 'tatdqa'):
        gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
        dataset_dir = f'datasets/tatdqa/test'
        storage_dir =  f'storages/tatdqa'
        result = eval_TATDQA(gt_path, dataset_dir, storage_dir, llm, embedder, metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation Pipeline')
    parser.add_argument('--model', type=str, default='tabrag', help='model to use')
    parser.add_argument('--dataset', type=str, default='tatdqa', help='dataset to evaluate on')
    parser.add_argument('--metric', type=str, default='acc', help='metric to eval on: acc, mrr10, ndcg10')
    args = parser.parse_args()
    main(args)