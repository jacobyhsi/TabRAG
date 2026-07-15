import os
import sys
import json
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import HFLLMClient, VLLMLLMClient, OpenAILLMClient
from src.embedder import SentenceTransformerEmbedder, VLLMEmbedder
from src.vector_store import VectorStore
from src.utils import normalize_answer, normalize_answer_tablevqa
import numpy as np
import re


#############
# Eval TabRAG
#############

def accuracy_with_std(correct, total):
    if total == 0:
        return {
            "accuracy": 0.0,
            "std": 0.0
        }
    acc = correct / total
    std = (acc * (1 - acc) / total) ** 0.5
    return {
        "accuracy": acc,
        "std": std,
        "n": total
    }


def get_page_text_for_baseline(page_path, method=None):
    """
    Return the page text stored in the docstore_data.jsonl file.

    The top-level ``text`` field already contains the overview and all component
    text for TabRAG records, so it can be used for every extraction method.
    """
    page_text = ""
    with open(page_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                page_text += obj.get("text", "") + "\n"
            except Exception as e:
                continue
    return page_text

def eval_tatdqa(gt_path, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for doc in tqdm(gt, desc="Evaluating TATDQA"):
        doc_doc = doc['doc']
        doc_id = doc['doc']['uid']
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
                    print("CORRECT!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("INCORRECT!")
                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
                print()

    results = {}
    results['accuracy'] = correct / total
    return accuracy_with_std(correct, total)

def eval_comtqa_baseline(gt_path, storage_dir, llm, emb, metric, method=None):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for q in tqdm(gt, desc="Evaluating ComTQA"):
        # Map to the correct folder name using image_name or table_id
        if 'image_name' in q:
            folder_name = q['image_name'].replace('.jpg', '')
        else:
            folder_name = str(q['table_id'])

        storage_path = os.path.join(storage_dir, folder_name)

        if not os.path.isdir(storage_path):
            print(f"Skipping {storage_path} because it doesn't exist")
            continue

        page_path = os.path.join(storage_path, "docstore_data.jsonl")
        if not os.path.isfile(page_path):
            print(f"[Skip] Missing {page_path}")
            continue

        page_text = get_page_text_for_baseline(page_path, method)

        q_q = q['question']
        q_a = q['answer']
        
        # comtqa is a flat list, so we process each item directly
        total += 1

        # QA interaction
        try:
            if metric == 'acc' or metric is None:
                system_prompt = f"""You are a helpful assistant. Use the information from the documents below to answer the question."""
                user_prompt = f"""/no_think {page_text} \n Question: {q_q} \n Answer: """

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                raw_response = normalize_answer(raw_response)

                # Handle multi-part answers often found in comtqa
                if isinstance(q_a, str) and '\n' in q_a:
                    q_a_list = [ans.strip() for ans in q_a.split('\n') if ans.strip()]
                elif not isinstance(q_a, list):
                    q_a_list = [q_a]
                else:
                    q_a_list = q_a

                q_a_norm = [normalize_answer(str(ans)) for ans in q_a_list]

                if all(ans in raw_response for ans in q_a_norm):
                    correct += 1
                    print("CORRECT!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("INCORRECT!")
                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
                print()
        except Exception as e:
            print(f"Error during QA interaction: {e}")
            continue
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break

    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3) if total > 0 else 0
    
    return results

################
# Eval Baselines (PyMuPDF, PyTesseract, VLM, etc.)
################

def eval_tatdqa_baseline(gt_path, storage_dir, llm, metric, method=None):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
 
    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for doc in tqdm(gt, desc="Evaluating TATDQA baseline"):
        doc_doc = doc['doc']
        doc_id = doc['doc']['uid']
        page = doc_doc.get("page")

        folder_name = f"{doc_id}_p{page}" if page is not None else doc_id
        storage_path = os.path.join(storage_dir, folder_name)

        if not os.path.isdir(storage_path):
            continue

        page_path = os.path.join(storage_path, "docstore_data.jsonl")
        if not os.path.isfile(page_path):
            continue
 
        page_text = get_page_text_for_baseline(page_path, method)
 
        # Loop through questions
        doc_qs = doc['questions']
        for q in doc_qs:
            q_q = q['question']
            q_a = q['answer']
 
            if len(q.get('block_mapping', [])) == 0:  # incomplete information
                continue
 
            total += 1
 
            # QA interaction
            if metric == 'acc' or metric is None:
                system_prompt = ("You are a helpful assistant. Use the information from the documents below to answer the question.")
                user_prompt = f"""/no_think {page_text}\nQuestion: {q_q}\nAnswer:"""
 
                raw_response = llm.generate(system_prompt, user_prompt)
 
                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')
 
                raw_response = normalize_answer(raw_response)
 
                if not isinstance(q_a, list):  # ensure q_a is list
                    q_a = [q_a]
 
                q_a = [normalize_answer(str(ans)) for ans in q_a]
 
                if all(ans in raw_response for ans in q_a):
                    correct += 1
                    print("CORRECT!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("INCORRECT!")
 
                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
                print()
 
    results = {}
    results['accuracy'] = correct / total
    return accuracy_with_std(correct, total)

def eval_mpdocvqa_baseline(gt_path, storage_dir, llm, metric, method=None):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
 
    correct = 0
    total = 0
    mistakes = []

    for qa_item in tqdm(gt['data'], desc="Evaluating MPDocVQA baseline"):
        doc_id = qa_item['doc_id']
        answer_page_idx = qa_item['answer_page_idx']
 
        if not qa_item.get('page_ids') or answer_page_idx >= len(qa_item['page_ids']):
            continue
 
        page_id = qa_item['page_ids'][answer_page_idx] 
        storage_path = os.path.join(storage_dir, doc_id, page_id)
 
        if not os.path.isdir(storage_path):
            print(f"Missing folder: {storage_path}")
            continue
 
        page_path = os.path.join(storage_path, "docstore_data.jsonl")
        if not os.path.isfile(page_path):
            print(f"Missing jsonl: {page_path}")
            continue
 
        page_text = get_page_text_for_baseline(page_path, method)
 
        q_q = qa_item['question']
        q_a = qa_item['answers']
        total += 1
 
        if metric == 'acc' or metric is None:
            system_prompt = ("You are a helpful assistant. Use the information from the documents below to answer the question.")
 
            user_prompt = f"""/no_think {page_text}\nQuestion: {q_q}\nAnswer:"""
 
            raw_response = llm.generate(system_prompt, user_prompt)
            print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')
 
            normalized_response = normalize_answer(raw_response)
            q_a = [normalize_answer(str(ans)) for ans in q_a]
 
            if any(ans in normalized_response for ans in q_a):
                correct += 1
                print("CORRECT!")
            else:
                mistakes.append((q_q, q_a, raw_response))
                print("INCORRECT!")
 
            print(f'Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}')
 
 
    results = {}
    results['accuracy'] = correct / total if total > 0 else 0.0
    results['mistakes'] = mistakes
    return accuracy_with_std(correct, total)
 
# def eval_SPIQA_baseline(gt_path, storage_dir, llm, judge_llm, method=None): 
#     with open(gt_path, 'r') as f:
#         gt_data = json.load(f)
 
#     all_questions = []
#     for paper_id, paper_data in gt_data.items():
#         qa_list = paper_data.get('qa', [])
#         figures = paper_data.get('all_figures', {})
 
#         for qa_item in qa_list:
#             ref_file = qa_item.get('reference')
#             if not ref_file:
#                 continue
 
#             fig_meta = figures.get(ref_file, {})
#             if 'page' not in fig_meta:
#                 continue  
 
#             qa_item['paper_id'] = paper_id
#             qa_item['page'] = fig_meta['page']
#             all_questions.append(qa_item)
 
#     print(f"Loaded {len(all_questions)} questions that have valid page numbers.")
 
#     total_items = 0
#     total_l3score = 0.0
#     results = {}
 
#     for i, qa_item in enumerate(all_questions):
#         question = qa_item.get('question')
#         gt_answer = qa_item.get('answer')
#         paper_id = qa_item.get('paper_id')
#         reference_file = qa_item.get('reference')
#         page = qa_item.get('page')
 
#         page_id = f"{paper_id}_p{page}"
#         storage_path = os.path.join(storage_dir, paper_id, page_id)
#         jsonl_path = os.path.join(storage_path, "docstore_data.jsonl")
 
#         if not os.path.exists(jsonl_path):
#             print(f"Warning: Skipping {paper_id} (page {page}) — no docstore_data.jsonl found.")
#             continue
 
#         page_text = get_page_text_for_baseline(jsonl_path, method)
 
#         total_items += 1
 
#         system_prompt = (
#             "You are a helpful assistant. Use the information from the documents below and any provided images to answer the question."
#         )
#         user_prompt = f"/no_think {page_text}\nQuestion: {question}\nAnswer: "
 
#         raw_response = llm.generate(
#             system_message=system_prompt,
#             user_message=user_prompt,
#         )
 
#         print(f"Question: {question}")
#         print()
#         print(f"Ground Truth: {gt_answer}")
#         print()
#         print(f"Response: {raw_response}")
#         print()
 
#         score, _ = judge_llm.generate(
#             question=question,
#             gt_answer=gt_answer,
#             candidate_answer=raw_response,
#         )
 
#         total_l3score += score
#         print(f"L3Score: {score:.4f} | Current Avg L3Score: {total_l3score/total_items:.4f}")
 
#     results['average_l3score'] = total_l3score / total_items if total_items > 0 else 0.0
#     return results

def eval_wikitq_baseline(gt_path, storage_dir, llm, metric='acc', method=None):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
 
    correct = 0
    total = 0
    mistakes = []
 
    for qa_item in tqdm(gt, desc="Evaluating WikiTableQuestions"):
        questions = qa_item.get('utterance', '')
        context = qa_item.get('context', '')
        answers = qa_item.get('targetValue', [])
 
        parts = context.split('/')
        if len(parts) < 3:
            continue
 
        doc_id = parts[-2]               
        csv_name = os.path.splitext(parts[-1])[0] 
        page_id = f"{doc_id}_p{csv_name}"      
 
        storage_path = os.path.join(storage_dir, doc_id, page_id)
        page_path = os.path.join(storage_path, "docstore_data.jsonl")
 
        if not os.path.isfile(page_path):
            print(f"[Skip] Missing {page_path}")
            continue
 
        page_text = get_page_text_for_baseline(page_path, method)
 
 
        total += 1
 
        system_prompt = (
                    "You are a helpful assistant. "
                    "Use the following page of information to answer the question."
                )
        user_prompt = f"""/no_think{page_text}\nQuestion: {questions}\nAnswer:"""
 
        # Generate model answer
        raw_response = llm.generate(system_prompt, user_prompt)
 
        print(f"Response: {raw_response}\nGround Truth: {answers}")
 
        # Normalize and compare
        norm_response = normalize_answer(raw_response)
        norm_answers = [normalize_answer(str(a)) for a in answers]
 
        if all(ans in norm_response for ans in norm_answers):
            correct += 1
            print("Correct")
        else:
            mistakes.append({
                "id": qa_item["id"],
                "question": questions,
                "gold": answers,
                "pred": raw_response
            })
            print("Incorrect")
 
        print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
        print()
 
    results = {}
    results['accuracy'] = correct / total
    return accuracy_with_std(correct, total)

def eval_tablevqabench_baseline(gt_path, storage_dir, llm, emb, metric = 'acc', method=None):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
 
    correct = 0
    total = 0
    mistakes = []
 
    for qa_item in tqdm(gt, desc="Evaluating tablevqa_bench"):
        questions = qa_item.get('question', '')
        context = qa_item.get('qa_id', '')
 
        answers = qa_item.get('gt', [])
        if not isinstance(answers, list):
            answers = [answers]
 
        parts = context.split('.')
        # if len(parts) < 3:
        #     continue
 
        page_id = parts[0]               
 
        storage_path = os.path.join(storage_dir, page_id)
        page_path = os.path.join(storage_path, "docstore_data.jsonl")
 
        if not os.path.isfile(page_path):
            print(f"[Skip] Missing {page_path}")
            continue
 
        page_text = get_page_text_for_baseline(page_path, method)
 
 
        total += 1
 
        system_prompt = (
                    "You are a helpful assistant. "
                    "Use the following page of information to answer the question."
                )
        user_prompt = f"""/no_think{page_text}\nQuestion: {questions}\nAnswer:"""
 
        # Generate model answer
        raw_response = llm.generate(system_prompt, user_prompt)
 
        print(f"Response: {raw_response}\nGround Truth: {answers}")
 
        # Compare (tablevqa-specific: numeric-only match when GT has a number)
        if all(normalize_answer_tablevqa(a, raw_response) for a in answers):
            correct += 1
            print("Correct")
        else:
            mistakes.append({
                "question": questions,
                "gold": answers,
                "pred": raw_response
            })
            print("Incorrect")
 
        print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
        print()
 
    results = accuracy_with_std(correct, total)
    results['mistakes'] = mistakes
    return results

def eval_finqa_baseline(gt_path, storage_dir, llm, metric, method=None):
    with open(gt_path, 'r') as f:
        if gt_path.endswith('.jsonl'):
            gt = [json.loads(line) for line in f]
        else:
            gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for doc in tqdm(gt, desc="Evaluating FinQA baseline"):
        q_q = doc.get('question', '')
        doc_name = doc.get('file_name', '')
        if doc_name.endswith('.pdf'):
            doc_name = doc_name.replace('.pdf', '')
        doc_name = doc_name.replace('\\', '')
        #ignore the first pdf/ in the path
        if doc_name.startswith('pdf/'):
            doc_name = doc_name[4:] 
        q_a = doc.get('program_answer', [])
        if not isinstance(q_a, list):
            q_a = [q_a]

        storage_path = os.path.join(storage_dir, doc_name)
        page_path = os.path.join(storage_path, "docstore_data.jsonl")

        if not os.path.isfile(page_path):
            print(f"[Skip] Missing {page_path}")
            continue

        page_text = get_page_text_for_baseline(page_path, method)
        total += 1

         # QA interaction
        try:
            if metric == 'acc' or metric is None:
                system_prompt = f"""You are a helpful assistant. Use the information from the documents below to answer the question."""
                user_prompt = f"""/no_think {page_text} \n Question: {q_q} \n Answer: """

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                 # Normalize and compare
                _DECREASE_WORDS = ['decrease', 'decline', 'drop', 'fell', 'reduced',
                                    'reduction', 'lower', 'loss', 'percentage decrease']

                def numeric_match(gt_str, response_text, tol=0.02):
                    """Return True if any number in response_text matches gt_str within tol.
                    Handles: % vs decimal, negative currency (-$X), rounding, sign+context."""
                    try:
                        gt_val = float(gt_str)
                    except ValueError:
                        return False
                    # Fix "-$5,821" -> "-5821" before normalization
                    prepped = re.sub(r'-\$\s*', '-', response_text)
                    prepped = re.sub(r'\+\$\s*', '', prepped)
                    norm_resp = normalize_answer(prepped)
                    raw_nums = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', norm_resp)
                    resp_floats = []
                    for n in raw_nums:
                        try:
                            resp_floats.append(float(n))
                        except ValueError:
                            pass
                    # Extract percentages (with commas) and convert to decimal
                    for m in re.finditer(r'(-?\d[\d,]*\.?\d*)\s*%', response_text):
                        try:
                            resp_floats.append(float(m.group(1).replace(',', '')) / 100)
                        except ValueError:
                            pass
                    resp_lower = response_text.lower()
                    has_decrease = any(w in resp_lower for w in _DECREASE_WORDS)
                    denom = abs(gt_val) if abs(gt_val) > 1e-9 else 1.0
                    for rv in resp_floats:
                        if abs(rv - gt_val) / denom <= tol:
                            return True
                        # GT is negative but LLM expressed magnitude + decrease word
                        if gt_val < 0 and has_decrease and abs(-abs(rv) - gt_val) / denom <= tol:
                            return True
                    return False

                def answer_matched_finqa(gt_val, response_text):
                    gt_str = str(gt_val)
                    if numeric_match(gt_str, response_text):
                        return True
                    norm_resp = normalize_answer(response_text)
                    norm_gt = normalize_answer(gt_str)
                    if norm_gt in norm_resp:
                        return True
                    # Integer-valued GT: "2825000.0" should match "2825000"
                    if norm_gt.endswith('.0') and norm_gt[:-2] in norm_resp:
                        return True
                    return False

                if all(answer_matched_finqa(a, raw_response) for a in q_a):
                    correct += 1
                    print("Correct")
                else:
                    mistakes.append({
                        "question": q_q,
                        "gold": q_a,
                        "pred": raw_response
                    })
                    print("Incorrect")

                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
                print()
        except Exception as e:
            print(f"Error during QA interaction: {e}")
            continue
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break

    results = {}
    results['accuracy'] = correct / total if total > 0 else 0.0
    return accuracy_with_std(correct, total)

def main(args):
    dataset = args.dataset
    metric = args.metric
    method = args.method
    vlm_model = args.vlm_model
    # llm = OpenAILLMClient('gpt-5.5')
    llm = HFLLMClient('Qwen/Qwen3.6-27B')
    # llm = VLLMLLMClient('Qwen/Qwen3-8B', ip='146.169.1.214', port='6000')

    # llm = None
    embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')
    # embedder = VLLMEmbedder('Qwen/Qwen3-Embedding-8B', tensor_parallel_size=2, gpu_memory_utilization=0.7)
    # embedder = None

    if (dataset == 'mpdocvqa'):
        gt_path = f'datasets/mpdocvqa/val.json'
        storage_dir =  f'storages/mpdocvqa/generation/{method}/{vlm_model}'
        result = eval_mpdocvqa_baseline(gt_path, storage_dir, llm, metric, method=method)

    if (dataset == 'tatdqa'):
        gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
        storage_dir =  f'storages/tatdqa/generation/{method}/{vlm_model}'
        # result = eval_tatdqa_baseline(gt_path, storage_dir, llm, metric, method=method)
        result = eval_tatdqa(gt_path, storage_dir, llm, embedder, metric)

    if (dataset == 'wikitq'):
        gt_path = f'datasets/wikitablequestions/qa.json'
        storage_dir =  f'storages/wikitablequestions/generation/{method}/{vlm_model}'
        result = eval_wikitq_baseline(gt_path, storage_dir, llm, metric, method=method)

    if (dataset == 'tablevqa'):
        gt_path = f'datasets/tablevqa/qa.json'
        storage_dir =  f'storages/tablevqa/generation/{method}/{vlm_model}'
        result = eval_tablevqabench_baseline(gt_path, storage_dir, llm, embedder, metric, method=method)

    if (dataset == 'tablevqa2'):
        gt_path = f'datasets/tablevqa/qa.json'
        storage_dir =  f'storages/tablevqa/generation/{method}/{vlm_model}_2'
        result = eval_tablevqabench_baseline(gt_path, storage_dir, llm, embedder, metric, method=method)
    
    if (dataset == 'comtqa'):
        gt_path = f'datasets/comtqa/qa3.json'
        storage_dir =  f'storages/comtqa/generation/{method}/{vlm_model}'
        result = eval_comtqa_baseline(gt_path, storage_dir, llm, embedder, metric, method=method)

    if (dataset == 'comtqa2'):
        gt_path = f'datasets/comtqa/qa3.json'
        storage_dir =  f'storages/comtqa/generation/{method}/{vlm_model}_2'
        result = eval_comtqa_baseline(gt_path, storage_dir, llm, embedder, metric, method=method)

    if (dataset == 'finqa'):
        gt_path = f'datasets/finqa/metadata.jsonl'
        storage_dir =  f'storages/{dataset}/generation/{method}/{vlm_model}'
        result = eval_finqa_baseline(gt_path, storage_dir, llm, metric, method=method)
    
    if 'storage_dir' not in locals() or 'result' not in locals():
        print(f"[Error] No matching dataset/model branch for dataset='{dataset}'")
        return
    print(storage_dir, result)
    record = {
        "dataset": dataset,
        "method": method,
        "vlm_model": vlm_model,
        "storage_dir": storage_dir,
        "result": result,
    }

    results_path = os.path.join(
        "results",
        os.path.relpath(storage_dir, "storages"),
        "results.json",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"Saved results to {results_path}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='wikitq', help='dataset to evaluate on')
    parser.add_argument('--model', type=str, default='tabrag', help='model to use, baseline or tabrag')
    parser.add_argument('--method', type=str, default='pymupdf', help='baseline model to use, e.g. pymupdf, pytesseract, vlm, deepseek, gemma_vlm')
    parser.add_argument('--metric', type=str, default='acc', help='metric to eval on: acc, mrr10, ndcg10')
    parser.add_argument('--vlm_model', type=str, default='Qwen3-VL-8B-Instruct', help='VLM model to use: Qwen3-VL-8B-Instruct,Qwen3-VL-32B-Instruct')
    args = parser.parse_args()
    main(args)

