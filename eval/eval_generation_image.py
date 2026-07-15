import os
import sys
import json
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import HFVLMClient, VLLMVLMClient
from src.utils import normalize_answer, normalize_answer_tablevqa
import re

#####################################################################
# Same evaluation loops/metrics as eval_generation.py, except the
# model is shown the raw page image directly instead of a text
# representation (TabRAG overview, PyMuPDF text, etc.).
#####################################################################

_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


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


def find_page_image(image_dir):
    """Return the path to the page image inside image_dir, or None if missing."""
    if not os.path.isdir(image_dir):
        return None
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(_IMAGE_EXTS):
            return os.path.join(image_dir, fname)
    return None


def eval_tatdqa_image(gt_path, image_root, vlm, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []
    for doc in tqdm(gt, desc="Evaluating TATDQA (image)"):
        doc_doc = doc['doc']
        doc_id = doc['doc']['uid']
        page = doc_doc.get("page")

        folder_name = f"{doc_id}_p{page}" if page is not None else doc_id
        image_path = find_page_image(os.path.join(image_root, folder_name))
        if image_path is None:
            continue

        doc_qs = doc['questions']
        for q in doc_qs:
            q_q = q['question']
            q_a = q['answer']

            if len(q.get('block_mapping', [])) == 0:
                continue

            total += 1

            if metric == 'acc' or metric is None:
                message = (
                    "You are a helpful assistant. Use the document image below to answer the question.\n"
                    f"Question: {q_q}\nAnswer:"
                )

                raw_response = vlm.generate(message, image_path)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                raw_response = normalize_answer(raw_response)

                if not isinstance(q_a, list):
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

    results = accuracy_with_std(correct, total)
    results['mistakes'] = mistakes
    return results


def eval_mpdocvqa_image(gt_path, image_root, vlm, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for qa_item in tqdm(gt['data'], desc="Evaluating MPDocVQA (image)"):
        answer_page_idx = qa_item['answer_page_idx']

        if not qa_item.get('page_ids') or answer_page_idx >= len(qa_item['page_ids']):
            continue

        page_id = qa_item['page_ids'][answer_page_idx]

        image_path = None
        for ext in _IMAGE_EXTS:
            candidate = os.path.join(image_root, f"{page_id}{ext}")
            if os.path.isfile(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"Missing image for page_id: {page_id}")
            continue

        q_q = qa_item['question']
        q_a = qa_item['answers']
        total += 1

        if metric == 'acc' or metric is None:
            message = (
                "You are a helpful assistant. Use the document image below to answer the question.\n"
                f"Question: {q_q}\nAnswer:"
            )

            raw_response = vlm.generate(message, image_path)
            print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

            normalized_response = normalize_answer(raw_response)
            q_a_norm = [normalize_answer(str(ans)) for ans in q_a]

            if any(ans in normalized_response for ans in q_a_norm):
                correct += 1
                print("CORRECT!")
            else:
                mistakes.append((q_q, q_a, raw_response))
                print("INCORRECT!")

            print(f'Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}')

    results = accuracy_with_std(correct, total)
    results['mistakes'] = mistakes
    return results


def eval_wikitq_image(gt_path, image_root, vlm, metric='acc'):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for qa_item in tqdm(gt, desc="Evaluating WikiTableQuestions (image)"):
        questions = qa_item.get('utterance', '')
        context = qa_item.get('context', '')
        answers = qa_item.get('targetValue', [])

        parts = context.split('/')
        if len(parts) < 3:
            continue

        doc_id = parts[-2]
        csv_name = os.path.splitext(parts[-1])[0]
        page_id = f"{doc_id}_p{csv_name}"

        image_path = find_page_image(os.path.join(image_root, doc_id, page_id))
        if image_path is None:
            print(f"[Skip] Missing image for {doc_id}/{page_id}")
            continue

        total += 1

        message = (
            "You are a helpful assistant. Use the document image below to answer the question.\n"
            f"Question: {questions}\nAnswer:"
        )

        raw_response = vlm.generate(message, image_path)

        print(f"Response: {raw_response}\nGround Truth: {answers}")

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

    results = accuracy_with_std(correct, total)
    results['mistakes'] = mistakes
    return results


def eval_tablevqabench_image(gt_path, image_root, vlm, metric='acc'):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for qa_item in tqdm(gt, desc="Evaluating tablevqa_bench (image)"):
        questions = qa_item.get('question', '')
        context = qa_item.get('qa_id', '')

        answers = qa_item.get('gt', [])
        if not isinstance(answers, list):
            answers = [answers]

        page_id = context.split('.')[0]

        image_path = find_page_image(os.path.join(image_root, page_id))
        if image_path is None:
            print(f"[Skip] Missing image for {page_id}")
            continue

        total += 1

        message = (
            "You are a helpful assistant. Use the document image below to answer the question.\n"
            f"Question: {questions}\nAnswer:"
        )

        raw_response = vlm.generate(message, image_path)

        print(f"Response: {raw_response}\nGround Truth: {answers}")

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


def eval_comtqa_image(gt_path, image_root, vlm, metric):
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    if not os.path.isdir(image_root):
        print(
            f"[Error] {image_root} does not exist. "
            "Extract comtqa_generation.zip into datasets/comtqa/generation first."
        )
        return accuracy_with_std(0, 0)

    correct = 0
    total = 0
    mistakes = []
    for q in tqdm(gt, desc="Evaluating ComTQA (image)"):
        if 'image_name' in q:
            folder_name = q['image_name'].replace('.jpg', '')
        else:
            folder_name = str(q['table_id'])

        image_path = find_page_image(os.path.join(image_root, folder_name))
        if image_path is None:
            print(f"[Skip] Missing image for {folder_name}")
            continue

        q_q = q['question']
        q_a = q['answer']

        total += 1

        try:
            if metric == 'acc' or metric is None:
                message = (
                    "You are a helpful assistant. Use the document image below to answer the question.\n"
                    f"Question: {q_q}\nAnswer:"
                )

                raw_response = vlm.generate(message, image_path)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                raw_response = normalize_answer(raw_response)

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
    results['mistakes'] = mistakes
    return results


_DECREASE_WORDS = ['decrease', 'decline', 'drop', 'fell', 'reduced',
                    'reduction', 'lower', 'loss', 'percentage decrease']


def _finqa_numeric_match(gt_str, response_text, tol=0.02):
    """Return True if any number in response_text matches gt_str within tol.
    Handles: % vs decimal, negative currency (-$X), rounding, sign+context."""
    try:
        gt_val = float(gt_str)
    except ValueError:
        return False
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
        if gt_val < 0 and has_decrease and abs(-abs(rv) - gt_val) / denom <= tol:
            return True
    return False


def _finqa_answer_matched(gt_val, response_text):
    gt_str = str(gt_val)
    if _finqa_numeric_match(gt_str, response_text):
        return True
    norm_resp = normalize_answer(response_text)
    norm_gt = normalize_answer(gt_str)
    if norm_gt in norm_resp:
        return True
    if norm_gt.endswith('.0') and norm_gt[:-2] in norm_resp:
        return True
    return False


def eval_finqa_image(gt_path, image_root, vlm, metric):
    with open(gt_path, 'r') as f:
        if gt_path.endswith('.jsonl'):
            gt = [json.loads(line) for line in f]
        else:
            gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for doc in tqdm(gt, desc="Evaluating FinQA (image)"):
        q_q = doc.get('question', '')
        doc_name = doc.get('file_name', '')
        if doc_name.endswith('.pdf'):
            doc_name = doc_name.replace('.pdf', '')
        doc_name = doc_name.replace('\\', '')
        if doc_name.startswith('pdf/'):
            doc_name = doc_name[4:]
        q_a = doc.get('program_answer', [])
        if not isinstance(q_a, list):
            q_a = [q_a]

        image_path = find_page_image(os.path.join(image_root, doc_name))
        if image_path is None:
            print(f"[Skip] Missing image for {doc_name}")
            continue

        total += 1

        try:
            if metric == 'acc' or metric is None:
                message = (
                    "You are a helpful assistant. Use the document image below to answer the question.\n"
                    f"Question: {q_q}\nAnswer:"
                )

                raw_response = vlm.generate(message, image_path)

                print(f'\nResponse: {raw_response} \nGround Truth: {q_a}')

                if all(_finqa_answer_matched(a, raw_response) for a in q_a):
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

    results = accuracy_with_std(correct, total)
    results['mistakes'] = mistakes
    return results


def main(args):
    dataset = args.dataset
    metric = args.metric
    vlm_model = args.vlm_model
    vlm = HFVLMClient(f'Qwen/{vlm_model}')
    # vlm = VLLMVLMClient(f'Qwen/{vlm_model}', ip=args.vllm_ip, port=args.vllm_port)

    if dataset == 'mpdocvqa':
        gt_path = 'datasets/mpdocvqa/val.json'
        image_root = 'datasets/mpdocvqa/images'
        result = eval_mpdocvqa_image(gt_path, image_root, vlm, metric)

    if dataset == 'tatdqa':
        gt_path = 'datasets/tatdqa/tatdqa_dataset_test_gold.json'
        image_root = 'datasets/tatdqa/generation'
        result = eval_tatdqa_image(gt_path, image_root, vlm, metric)

    if dataset == 'wikitq':
        gt_path = 'datasets/wikitablequestions/qa.json'
        image_root = 'datasets/wikitablequestions/generation'
        result = eval_wikitq_image(gt_path, image_root, vlm, metric)

    if dataset == 'tablevqa':
        gt_path = 'datasets/tablevqa/qa.json'
        image_root = 'datasets/tablevqa/generation'
        result = eval_tablevqabench_image(gt_path, image_root, vlm, metric)

    if dataset == 'comtqa':
        gt_path = 'datasets/comtqa/qa.json'
        image_root = 'datasets/comtqa/generation'
        result = eval_comtqa_image(gt_path, image_root, vlm, metric)

    if dataset == 'finqa':
        gt_path = 'datasets/finqa/metadata.jsonl'
        image_root = 'datasets/finqa/generation'
        result = eval_finqa_image(gt_path, image_root, vlm, metric)

    if 'image_root' not in locals() or 'result' not in locals():
        print(f"[Error] No matching dataset branch for dataset='{dataset}'")
        return

    print(image_root, result)
    record = {
        "dataset": dataset,
        "method": "image",
        "vlm_model": vlm_model,
        "image_root": image_root,
        "result": result,
    }

    results_path = os.path.join(
        "results",
        dataset,
        "generation",
        "image",
        vlm_model,
        "results.json",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation Pipeline (image-based)')
    parser.add_argument('--dataset', type=str, default='wikitq', help='dataset to evaluate on')
    parser.add_argument('--metric', type=str, default='acc', help='metric to eval on: acc')
    parser.add_argument('--vlm_model', type=str, default='Qwen3.6-27B',
                         help='VLM model to use directly on the page image, e.g. Qwen3.6-27B, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct')
    parser.add_argument('--vllm_ip', type=str, default='localhost')
    parser.add_argument('--vllm_port', type=str, default='2222')
    args = parser.parse_args()
    main(args)
