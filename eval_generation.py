import os
import json
from tqdm import tqdm
import argparse
from src.llm import HFLLMClient, VLLMLLMClient, L3ScoreVLLMLLMClient
from src.embedder import SentenceTransformerEmbedder, VLLMEmbedder
from src.vector_store import VectorStore
from src.utils import normalize_answer

#############
# Eval TabRAG
#############

def eval_TATDQA(gt_path, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for doc in gt:
        doc_doc = doc['doc']
        doc_id = doc['doc']['uid']
        page = doc_doc.get("page")

        folder_name = f"{doc_id}_p{page}" if page is not None else doc_id
        storage_path = os.path.join(storage_dir, folder_name)

        if not os.path.isdir(storage_path):
            print(f"Skipping {storage_path} because it doesn't exist")
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
            try:
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
            except Exception as e:
                print(f"Error during QA interaction: {e}")
                continue
            except KeyboardInterrupt:
                print("Evaluation interrupted by user.")
                break

    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)
    
    return results

def eval_MPDocVQA(gt_path, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for qa_item in gt['data']:
        doc_id = qa_item['doc_id']
        answer_page_idx = qa_item['answer_page_idx']

        if not qa_item.get('page_ids') or answer_page_idx >= len(qa_item['page_ids']):
            continue

        page_id = qa_item['page_ids'][answer_page_idx]
        storage_path = os.path.join(storage_dir, doc_id, page_id)

        if not os.path.isdir(storage_path):
            continue

        docstore_path = os.path.join(storage_path, "docstore")
        index = VectorStore.load(docstore_path)

        q_q = qa_item['question']
        q_a = qa_item['answers']

        q_embed = emb.encode([q_q]).astype("float32")
        total += 1

        # QA interaction
        try:
            if metric == 'acc' or metric is None:
                retr = index.search(q_embed, k=1)

                system_prompt = (
                    "You are a helpful assistant. "
                    "Use the information from the documents below to answer the question."
                )
                user_prompt = f"/no_think {retr[0]['text']} \n Question: {q_q} \n Answer: "

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f"\nResponse: {raw_response} \nGround Truth: {q_a}")

                normalized_response = normalize_answer(raw_response)
                q_a_norm = [normalize_answer(str(ans)) for ans in q_a]

                if any(ans in normalized_response for ans in q_a_norm):
                    correct += 1
                    print("correct!")
                else:
                    mistakes.append((q_q, q_a_norm, raw_response))
                    print("incorrect!")

                print(
                    f"Current Accuracy: {correct/total:.4f} | "
                    f"# correct: {correct}, # total: {total}"
                )

        except Exception as e:
            print(f"Error during QA interaction: {e}")
            continue

        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break

    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)
    
    return results

def eval_WikiTQ(gt_path, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for doc in gt:
        q_id = doc['id']
        q_q = doc['utterance']
        q_a = doc['targetValue']
        context_csv = doc['context']

        # Parse context: "csv/201-csv/21.csv" → doc_id="201-csv", page_id="201-csv_p21"
        parts = context_csv.split('/')
        if len(parts) < 3:
            continue

        doc_id = parts[-2]
        csv_name = os.path.splitext(parts[-1])[0]
        page_id = f"{doc_id}_p{csv_name}"

        storage_path = os.path.join(storage_dir, doc_id, page_id)
        docstore_path = os.path.join(storage_path, "docstore")

        index = VectorStore.load(docstore_path)

        q_embed = emb.encode([q_q]).astype("float32")
        total += 1

        # QA interaction
        try:
            if metric == 'acc' or metric is None:
                retr = index.search(q_embed, k=1)
                system_prompt = (
                    "You are a helpful assistant. "
                    "Use the information from the documents below to answer the question."
                )
                user_prompt = f"/no_think {retr[0]['text']} \n Question: {q_q} \n Answer: "

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f"\nResponse: {raw_response} \nGround Truth: {q_a}")

                normalized_response = normalize_answer(raw_response)

                if not isinstance(q_a, list):
                    q_a = [q_a]

                q_a_norm = [normalize_answer(str(ans)) for ans in q_a]

                if all(ans in normalized_response for ans in q_a_norm):
                    correct += 1
                    print("correct!")
                else:
                    mistakes.append((q_q, q_a_norm, normalized_response))
                    print("incorrect!")

                print(
                    f"Current Accuracy: {correct/total} | "
                    f"# correct: {correct}, # total: {total}"
                )

        except Exception as e:
            print(f"Error during QA interaction: {e}")
            continue

        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break

    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)
    
    return results


def eval_TableVQA(gt_path, storage_dir, llm, emb, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for doc in gt:
        q_q = doc.get('question', '')
        print(f"Question: {q_q}")
        context = doc.get('qa_id', '')
        q_a = doc.get('gt', [])

        parts = context.split('.')
        # if len(parts) < 3:
        #     continue
 
        page_id = parts[0]

        storage_path = os.path.join(storage_dir, page_id)
        index_prefix = os.path.join(storage_path, "docstore")

        if not os.path.isfile(index_prefix + ".index"):
            print(f"[Skip] Missing {index_prefix}.index")
            continue

        index = VectorStore.load(index_prefix)

        q_embed = emb.encode([q_q]).astype("float32")
        total += 1

        # QA interaction
        try:
            if metric == 'acc' or metric is None:
                retr = index.search(q_embed, k=1)
                system_prompt = f"""You are a helpful assistant. Use the information from the documents below to answer the question."""
                user_prompt = f"""/no_think {retr[0]['text']} \n Question: {q_q} \n Answer: """

                raw_response = llm.generate(system_prompt, user_prompt)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                if not isinstance(q_a, list):
                    q_a = [q_a]

                q_a = [normalize_answer(str(ans)) for ans in q_a]
                raw_response = normalize_answer(raw_response)

                if all(ans in raw_response for ans in q_a):
                    correct += 1
                    print("correct!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("incorrect!")

                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')

        except Exception as e:
            print(f"Error during QA interaction: {e}")
            continue
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break

    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)
    
    return results


def eval_SPIQA(gt_path, storage_dir, llm, emb, judge_llm):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # Flatten the dictionary of papers into a single list of questions that have page numbers
    all_questions = []
    for paper_id, paper_data in gt_data.items():
        qa_list = paper_data.get('qa', [])
        figures = paper_data.get('all_figures', {})

        for qa_item in qa_list:
            ref_file = qa_item.get('reference')
            if not ref_file:
                continue

            fig_meta = figures.get(ref_file, {})
            if 'page' not in fig_meta:
                continue  # only include items with a page number

            qa_item['paper_id'] = paper_id
            qa_item['page'] = fig_meta['page']
            all_questions.append(qa_item)

    print(f"Loaded {len(all_questions)} questions that have valid page numbers.")

    total_items = 0
    total_l3score = 0.0
    results = {}

    for i, qa_item in enumerate(all_questions):
        question = qa_item.get('question')
        gt_answer = qa_item.get('answer')
        paper_id = qa_item.get('paper_id')
        reference_file = qa_item.get('reference')
        page = qa_item.get('page')


        # --- 1. Locate and load the Vector Store ---
        # page index naming convention: 1611.02654v2_p1
        page_id = f"{paper_id}_p{page}"
        storage_path = os.path.join(storage_dir, paper_id, page_id)
        docstore_path = os.path.join(storage_path, "docstore.index")

        if not os.path.exists(docstore_path):
            print(f"Warning: Skipping item {i+1} (paper {paper_id}, page {page}) — docstore not found at '{docstore_path}'.")
            continue

        index = VectorStore.load(os.path.join(storage_path, "docstore"))

        total_items += 1

        # --- 3. Retrieve Context ---
        q_embed = emb.encode([question]).astype("float32")
        retrieved_docs = index.search(q_embed, k=1)
        retrieved_context = retrieved_docs[0]['text']

        print(f"\n--- Item {i+1}/{len(all_questions)} (Page {page}) ---")
        print(f"Generating response for paper {paper_id}...")

        # --- 4. Generate Candidate Answer ---
        system_prompt = (
            "You are a helpful assistant. Use the information from the documents below and any provided images to answer the question."
        )
        user_prompt = f"/no_think {retrieved_context} \n Question: {question} \n Answer: "

        raw_response = llm.generate(
            system_message=system_prompt,
            user_message=user_prompt,
        )

        print(f"Question: {question}")
        print()
        print(f"Ground Truth: {gt_answer}")
        print()
        print(f"Response: {raw_response}")
        print()

        # --- 5. Judge ---
        score, _ = judge_llm.generate(
            question=question,
            gt_answer=gt_answer,
            candidate_answer=raw_response,
        )

        total_l3score += score
        print(f"L3Score: {score:.4f} | Current Avg L3Score: {total_l3score/total_items:.4f}")

    # --- 6. Finalize ---
    results['average_l3score'] = total_l3score / total_items if total_items > 0 else 0.0
    return results


##################################################
# Eval Baselines (PyMuPDF, PyTesseract, VLM, etc.)
##################################################

def eval_TATDQA_baseline(gt_path, storage_dir, llm, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
 
    correct = 0 # accuracy calc
    total = 0
    mistakes = []
    for doc in gt:
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
 
        page_text = ""
        with open(page_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    page_text += obj["text"] + "\n"
                except:
                    continue
 
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
                    print("correct!")
                else:
                    mistakes.append((q_q, q_a, raw_response))
                    print("incorrect!")
 
                print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')
 
    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)

    return results

def eval_MPDocVQA_baseline(gt_path, storage_dir, llm, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
 
 
    correct = 0
    total = 0
    mistakes = []
 
    for qa_item in gt['data']:
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
 
        page_text = ""
        with open(page_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    page_text += obj["text"] + "\n"
                except:
                    continue
 
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
                print("correct!")
            else:
                mistakes.append((q_q, q_a, raw_response))
                print("incorrect!")
 
            print(f'Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}')
 
 
    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)

    return results
 
def eval_SPIQA_baseline(gt_path, storage_dir, llm, judge_llm): 
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
 
    all_questions = []
    for paper_id, paper_data in gt_data.items():
        qa_list = paper_data.get('qa', [])
        figures = paper_data.get('all_figures', {})
 
        for qa_item in qa_list:
            ref_file = qa_item.get('reference')
            if not ref_file:
                continue
 
            fig_meta = figures.get(ref_file, {})
            if 'page' not in fig_meta:
                continue  
 
            qa_item['paper_id'] = paper_id
            qa_item['page'] = fig_meta['page']
            all_questions.append(qa_item)
 
    print(f"Loaded {len(all_questions)} questions that have valid page numbers.")
 
    total_items = 0
    total_l3score = 0.0
    results = {}
 
    for i, qa_item in enumerate(all_questions):
        question = qa_item.get('question')
        gt_answer = qa_item.get('answer')
        paper_id = qa_item.get('paper_id')
        reference_file = qa_item.get('reference')
        page = qa_item.get('page')
 
        page_id = f"{paper_id}_p{page}"
        storage_path = os.path.join(storage_dir, paper_id, page_id)
        jsonl_path = os.path.join(storage_path, "docstore_data.jsonl")
 
        if not os.path.exists(jsonl_path):
            print(f"Warning: Skipping {paper_id} (page {page}) — no docstore_data.jsonl found.")
            continue
 
        page_text = ""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                txt = rec.get("text", "").strip()
                if txt:
                    page_text += txt + "\n"
 
        total_items += 1
 
        system_prompt = (
            "You are a helpful assistant. Use the information from the documents below and any provided images to answer the question."
        )
        user_prompt = f"/no_think {page_text}\nQuestion: {question}\nAnswer: "
 
        raw_response = llm.generate(
            system_message=system_prompt,
            user_message=user_prompt,
        )
 
        print(f"Question: {question}")
        print()
        print(f"Ground Truth: {gt_answer}")
        print()
        print(f"Response: {raw_response}")
        print()
 
        score, _ = judge_llm.generate(
            question=question,
            gt_answer=gt_answer,
            candidate_answer=raw_response,
        )
 
        total_l3score += score
        print(f"L3Score: {score:.4f} | Current Avg L3Score: {total_l3score/total_items:.4f}")
 
    results['average_l3score'] = total_l3score / total_items if total_items > 0 else 0.0
    return results

def eval_WikiTQ_baseline(gt_path, storage_dir, llm, metric='acc'):
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
 
        page_data = []
        with open(page_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    page_data.append(json.loads(line))
                except:
                    continue
 
        page_text = "\n".join([blk.get("text", "").strip() for blk in page_data if blk.get("text")])
 
 
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
 
    results = {}
    results['correct'] = correct
    results['total'] = total
    results['accuracy'] = round((correct / total) * 100, 3)

    return results

def save_results(result, dataset, model):
    results_dir = os.path.join("results", dataset, model)
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "eval.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Saved results] {out_path}")


def main(args):
    dataset = args.dataset
    metric = args.metric
    model = args.model
    # llm = HFLLMClient('Qwen/Qwen3-8B')
    llm = VLLMLLMClient('Qwen/Qwen3-8B', ip='localhost', port='1111')

    # llm = None
    embedder = SentenceTransformerEmbedder('Qwen/Qwen3-Embedding-8B')
    # embedder = VLLMEmbedder('Qwen/Qwen3-Embedding-8B', tensor_parallel_size=1, gpu_memory_utilization=0.6)
    # embedder = None
    if (model == 'tabrag'):
        if (dataset == 'tatdqa'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'storages/tatdqa/generation/{model}'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tatdqa_xStructureICL'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'storages/tatdqa/generation/{model}/Qwen3-VL-8B-Instruct_xStructureICL'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tatdqa_layoutonly'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/tatdqa/generation/layout_only'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tatdqa_xLayout'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'storages/tatdqa/generation/tabrag/Qwen3-VL-8B-Instruct_xLayout/'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tatdqa_noICL'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/tatdqa/generation/no_icl'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tatdqa_md'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG_icl/storages/tatdqa/generation/tabrag/md'
            result = eval_TATDQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)

        if (dataset == 'mpdocvqa'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'storages/mpdocvqa/generation/{model}'
            result = eval_MPDocVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'mpdocvqa_xStructureICL'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'storages/mpdocvqa/generation/{model}/Qwen3-VL-8B-Instruct_xStructureICL'
            result = eval_MPDocVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'mpdocvqa_noICL'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/mpdocvqa/generation/no_icl'
            result = eval_MPDocVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'mpdocvqa_xLayout'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'storages/mpdocvqa/generation/tabrag/Qwen3-VL-8B-Instruct_xLayout/'
            result = eval_MPDocVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'mpdocvqa_layoutonly'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/mpdocvqa/generation/layout_only'
            result = eval_MPDocVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)

        if (dataset == 'wikitq'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'storages/wikitablequestions/generation/{model}'
            result = eval_WikiTQ(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'wikitq_xStructureICL'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'storages/wikitablequestions/generation/{model}/Qwen3-VL-8B-Instruct_xStructureICL'
            result = eval_WikiTQ(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'wikitq_noICL'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/wikitablequestions/generation/no_icl'
            result = eval_WikiTQ(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'wikitq_xLayout'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'storages/wikitablequestions/generation/tabrag/Qwen3-VL-8B-Instruct_xLayout/'
            result = eval_WikiTQ(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'wikitq_layoutonly'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/wikitablequestions/generation/layout_only'
            result = eval_WikiTQ(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)

        if (dataset == 'tablevqa_xStructureICL'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'storages/tablevqa/generation/{model}/Qwen3-VL-8B-Instruct_xStructureICL'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_xLayout'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'storages/tablevqa/generation/tabrag/Qwen3-VL-8B-Instruct_xLayout/'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_noICL'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/tablevqa_bench_processed/generation/no_icl'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_layoutonly'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG/storages/tablevqa_bench_processed/generation/layout_only'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_tabrag1'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/mml324/TabRAG/storages/tablevqa/generation/tabrag_1/Qwen3-VL-8B-Instruct'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_tabrag5'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/mml324/TabRAG/storages/tablevqa/generation/tabrag_5/Qwen3-VL-8B-Instruct'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_tabrag7'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/mml324/TabRAG/storages/tablevqa/generation/tabrag_7/Qwen3-VL-8B-Instruct'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)
        if (dataset == 'tablevqa_md'):
            gt_path = f'datasets/tablevqa/qa.json'
            storage_dir =  f'/vol/bitbucket/js2723/PROJECTS/TabRAG_icl/storages/tablevqa_bench_processed/generation/tabrag/md'
            result = eval_TableVQA(gt_path, storage_dir, llm, embedder, metric)
            save_results(result, dataset, model)

        if (dataset == 'spiqa'):
            gt_path = f'datasets/spiqa/test-A/SPIQA_testA_wpage.json'
            storage_dir =  f'storages/spiqa/generation/{model}'
            judge_llm = L3ScoreVLLMLLMClient('Qwen/Qwen3-8B', ip='146.169.1.214', port='6000')
            result = eval_SPIQA(gt_path, storage_dir, llm, embedder, judge_llm)
            save_results(result, dataset, model)

        # if (dataset == 'spiqa_xStructureICL'):
        #     gt_path = f'datasets/spiqa/test-A/SPIQA_testA_wpage.json'
        #     storage_dir =  f'storages/spiqa/generation/{model}/Qwen3-VL-8B-Instruct_xStructureICL'
        #     judge_llm = L3ScoreVLLMLLMClient('Qwen/Qwen3-8B', ip='146.169.1.214', port='6000')
        #     result = eval_SPIQA(gt_path, storage_dir, llm, embedder, judge_llm)
            # save_results(result, storage_dir)



    elif (model == 'baseline'):
        if (dataset == 'mpdocvqa'):
            gt_path = f'datasets/mpdocvqa/val.json'
            storage_dir =  f'storages/mpdocvqa/generation/{model}'
            result = eval_MPDocVQA_baseline(gt_path, storage_dir, llm, metric)
            save_results(result, dataset, model)

        if (dataset == 'tatdqa'):
            gt_path = f'datasets/tatdqa/tatdqa_dataset_test_gold.json'
            storage_dir =  f'storages/tatdqa/generation/{model}'
            result = eval_TATDQA_baseline(gt_path, storage_dir, llm, metric)
            save_results(result, dataset, model)

        if (dataset == 'wikitq'):
            gt_path = f'datasets/wikitablequestions/qa.json'
            storage_dir =  f'storages/wikitablequestions/generation/{model}'
            result = eval_WikiTQ_baseline(gt_path, storage_dir, llm, metric)
            save_results(result, dataset, model)

        if (dataset == 'spiqa'):
            gt_path = f'datasets/spiqa/test-A/SPIQA_testA_wpage.json'
            storage_dir =  f'storages/spiqa/generation/{model}'
            judge_llm = L3ScoreVLLMLLMClient('Qwen/Qwen3-8B', ip='146.169.1.214', port='6000')
            result = eval_SPIQA_baseline(gt_path, storage_dir, llm, judge_llm)
            save_results(result, dataset, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='tatdqa_xStructureICL', help='dataset to evaluate on')
    parser.add_argument('--model', type=str, default='tabrag', help='model to use')
    parser.add_argument('--metric', type=str, default='acc', help='metric to eval on: acc, mrr10, ndcg10')
    args = parser.parse_args()
    main(args)
