
import argparse
import json
from operator import index
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.embedder import SentenceTransformerEmbedder
from src.llm import HFLLMClient
from src.utils import normalize_answer, normalize_answer_tablevqa

from src.vector_store import VectorStore
import re


DEFAULT_GT = {
    "mpdocvqa": "datasets/mpdocvqa/val.json",
    "tatdqa": "datasets/tatdqa/tatdqa_dataset_test_gold.json",
    "wikitq": "datasets/wikitablequestions/qa.json",
    "tablevqa": "datasets/tablevqa/qa.json",
    "comtqa": "datasets/comtqa/qa3.json",
    "convfinqa": "datasets/finqa/metadata.jsonl",
    "finqa": "datasets/finqa/metadata.jsonl",
}


def accuracy_with_std(correct, total):
    if total == 0:
        return {"accuracy": 0.0, "std": 0.0, "n": 0}
    acc = correct / total
    std = (acc * (1 - acc) / total) ** 0.5
    return {"accuracy": acc, "std": std, "n": total}


_DECREASE_WORDS = [
    "decrease", "decline", "drop", "fell", "reduced",
    "reduction", "lower", "loss", "percentage decrease",
]


def numeric_match_finqa(gt_value, response_text, tol=0.02):
    """Return whether a number in the response matches a FinQA answer."""
    try:
        gt_value = float(gt_value)
    except (TypeError, ValueError):
        return False

    prepped = re.sub(r"-\$\s*", "-", response_text)
    prepped = re.sub(r"\+\$\s*", "", prepped)
    normalized_response = normalize_answer(prepped)

    response_values = []
    for number in re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", normalized_response):
        try:
            response_values.append(float(number))
        except ValueError:
            pass

    # FinQA stores ratios as decimals, while models often answer with percentages.
    for match in re.finditer(r"(-?\d[\d,]*\.?\d*)\s*%", response_text):
        try:
            response_values.append(float(match.group(1).replace(",", "")) / 100)
        except ValueError:
            pass

    has_decrease = any(word in response_text.lower() for word in _DECREASE_WORDS)
    denominator = abs(gt_value) if abs(gt_value) > 1e-9 else 1.0
    for response_value in response_values:
        if abs(response_value - gt_value) / denominator <= tol:
            return True
        if (
            gt_value < 0
            and has_decrease
            and abs(-abs(response_value) - gt_value) / denominator <= tol
        ):
            return True
    return False


def answer_matched_finqa(gt_value, response_text):
    """Match FinQA numeric answers while retaining a text fallback."""
    if numeric_match_finqa(gt_value, response_text):
        return True

    normalized_gt = normalize_answer(str(gt_value))
    normalized_response = normalize_answer(response_text)
    if normalized_gt in normalized_response:
        return True
    if normalized_gt.endswith(".0") and normalized_gt[:-2] in normalized_response:
        return True
    return False


def shard_contains_doc(shard_path, doc_id):
    jsonl_path = os.path.join(shard_path, "docstore_data.jsonl")
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            fname = obj["meta"]["file"]
            if fname.startswith(doc_id + "_p"):
                return True
    return False


def eval_tatdqa(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    # find folder names
    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    # for each q of each doc
    for item in tqdm(gt, desc="tatdqa full pipeline"):
        doc_id = item["doc"]["uid"]
        questions = item["questions"]

        shard_path = None
        # find which shard contains the doc
        for s in shards:
            full_path = os.path.join(store_path, s)
            if shard_contains_doc(full_path, doc_id):
                shard_path = full_path
                break

        if shard_path is None:
            print(f"WARNING: doc {doc_id} not found in any folder")
            continue

        #load index for the shard
        index = VectorStore.load(os.path.join(shard_path, "docstore"))

        for q in questions:
            q_text = q.get("question", "")
            q_a = q.get("answer", [])
            if not q_text:
                continue

            q_embed = embedder.encode([q_text]).astype("float32")
            retrieved = index.search(q_embed, k=top_k)

            context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
            if not context_text:
                continue

            system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
            user_prompt = f"/no_think {context_text}\nQuestion: {q_text}\nAnswer: "
            raw_response = llm.generate(system_prompt, user_prompt)
            raw_response = normalize_answer(raw_response)

            if not isinstance(q_a, list):
                q_a = [q_a]
            q_a = [normalize_answer(str(ans)) for ans in q_a]
            print(f"Question: {q_text}")
            print(f"Ground Truth Answers: {q_a}")
            print(f"Raw Response: {raw_response}")

            total += 1

            if all(ans in raw_response for ans in q_a):
                correct += 1
                print("correct!")
            else:
                mistakes.append((q_text, q_a, raw_response))
                print("incorrect!")

            print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats

# use q instead of q_q
def eval_tatdqa_n(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    # find folder names
    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    # for each q of each doc
    for item in tqdm(gt, desc="tatdqa full pipeline"):
        doc_id = item["doc"]["uid"]
        questions = item["questions"]

        shard_path = None
        # find which shard contains the doc
        for s in shards:
            full_path = os.path.join(store_path, s)
            if shard_contains_doc(full_path, doc_id):
                shard_path = full_path
                break

        if shard_path is None:
            print(f"WARNING: doc {doc_id} not found in any folder")
            continue

        #load index for the shard
        index = VectorStore.load(os.path.join(shard_path, "docstore"))

        for q in questions:
            q_text = q.get("question", "")
            q_a = q.get("answer", [])
            if not q_text:
                continue

            q_embed = embedder.encode([q_text]).astype("float32")
            retrieved = index.search(q_embed, k=top_k)

            context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
            if not context_text:
                continue

            system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
            user_prompt = f"/no_think {context_text}\nQuestion: {q}\nAnswer: "
            raw_response = llm.generate(system_prompt, user_prompt)
            raw_response = normalize_answer(raw_response)

            if not isinstance(q_a, list):
                q_a = [q_a]
            q_a = [normalize_answer(str(ans)) for ans in q_a]
            print(f"Question: {q_text}")
            print(f"Ground Truth Answers: {q_a}")
            print(f"Raw Response: {raw_response}")

            total += 1

            if all(ans in raw_response for ans in q_a):
                correct += 1
                print("correct!")
            else:
                mistakes.append((q_text, q_a, raw_response))
                print("incorrect!")

            print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats


def eval_mpdocvqa(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
            gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for question in tqdm(gt['data']):
        doc_id = question['doc_id']
        page_ids = question['page_ids']
        storage_path = os.path.join(store_path, doc_id)

        answer_page_idx = question['answer_page_idx']
        if not page_ids or answer_page_idx >= len(page_ids):
            continue
        answer_page = page_ids[answer_page_idx]

        if not os.path.exists(storage_path):
            continue

        # Check if answer page is actually indexed
        indexed_pages = set()
        with open(os.path.join(storage_path, 'docstore_data.jsonl')) as f:
            for line in f:
                obj = json.loads(line)
                page_id = obj['meta']['file'].replace('.jpg', '').replace('.png', '')
                indexed_pages.add(page_id)
        if answer_page not in indexed_pages:
            continue

        print(f"Processing: {doc_id}/{answer_page}")
        index = VectorStore.load(os.path.join(storage_path, 'docstore'))
        doc_q = question['question']
        q_a = question['answers']
        q_embed = embedder.encode([doc_q]).astype("float32")

        retrieved_pages = index.search(q_embed, k=top_k)
        context_text = "\n".join(r["text"] for r in retrieved_pages if r.get("text"))
        if not context_text:
            continue

        system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
        user_prompt = f"/no_think {context_text}\nQuestion: {doc_q}\nAnswer: "
        raw_response = llm.generate(system_prompt, user_prompt)
        raw_response = normalize_answer(raw_response)

        if not isinstance(q_a, list):
            q_a = [q_a]
        q_a = [normalize_answer(str(ans)) for ans in q_a]
        print(f"Question: {doc_q}")
        print(f"Ground Truth Answers: {q_a}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if any(ans in raw_response for ans in q_a):
            correct += 1
            print("correct!")
        else:
            mistakes.append((doc_q, q_a, raw_response))
            print("incorrect!")

        print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats


def eval_wikitq(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        if json_path.endswith(".jsonl"):
            gt = [json.loads(line) for line in f if line.strip()]
        else:
            gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="wikitq full pipeline"):
        q_text = item.get("utterance", "")
        q_a = item.get("targetValue", "")
        context_csv = item.get("context", "")

        if not q_text or not context_csv:
            continue

        parts = context_csv.split("/")
        if len(parts) < 3:
            continue

        doc_id = parts[-2]  # e.g. "201-csv"

        storage_path = os.path.join(store_path, doc_id)
        if not os.path.exists(storage_path):
            print(f"WARNING: storage path {storage_path} not found")
            continue

        index = VectorStore.load(os.path.join(storage_path, "docstore"))

        q_embed = embedder.encode([q_text]).astype("float32")
        retrieved = index.search(q_embed, k=top_k)

        context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
        if not context_text:
            continue

        system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
        user_prompt = f"/no_think {context_text}\nQuestion: {q_text}\nAnswer: "
        raw_response = llm.generate(system_prompt, user_prompt)
        raw_response = normalize_answer(raw_response)

        if not isinstance(q_a, list):
            q_a = [q_a]
        q_a = [normalize_answer(str(ans)) for ans in q_a]
        print(f"Question: {q_text}")
        print(f"Ground Truth Answers: {q_a}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(ans in raw_response for ans in q_a):
            correct += 1
            print("correct!")
        else:
            mistakes.append((q_text, q_a, raw_response))
            print("incorrect!")

        print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats


def shard_contains_doc_comtqa(shard_path, doc_id):
    jsonl_path = os.path.join(shard_path, "docstore_data.jsonl")
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            fname = obj["meta"]["file"]
            if fname == doc_id:
                return True
    return False


def eval_comtqa(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="comtqa full pipeline"):
        if "image_name" in item:
            # doc_id = item["image_name"].replace(".jpg", "")
            doc_id= item["image_name"]
        else:
            doc_id = str(item["table_id"])+".png"

        q_text = item.get("question", "")
        q_a = item.get("answer", "")
        if not q_text:
            continue

        shard_path = None
        for s in shards:
            full_path = os.path.join(store_path, s)
            if shard_contains_doc_comtqa(full_path, doc_id):
                shard_path = full_path
                break

        if shard_path is None:
            print(f"WARNING: doc {doc_id} not found in any shard")
            continue

        index = VectorStore.load(os.path.join(shard_path, "docstore"))

        q_embed = embedder.encode([q_text]).astype("float32")
        retrieved = index.search(q_embed, k=top_k)

        context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
        if not context_text:
            continue

        system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
        user_prompt = f"/no_think {context_text}\nQuestion: {q_text}\nAnswer: "
        raw_response = llm.generate(system_prompt, user_prompt)
        raw_response = normalize_answer(raw_response)

        if isinstance(q_a, str) and "\n" in q_a:
            q_a = [ans.strip() for ans in q_a.split("\n") if ans.strip()]
        elif not isinstance(q_a, list):
            q_a = [q_a]
        q_a = [normalize_answer(str(ans)) for ans in q_a]
        q_a = [ans for ans in q_a if ans]  # skip empty answers
        if not q_a:
            continue
        print(f"Question: {q_text}")
        print(f"Ground Truth Answers: {q_a}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(ans in raw_response for ans in q_a):
            correct += 1
            print("correct!")
        else:
            mistakes.append((q_text, q_a, raw_response))
            print("incorrect!")

        print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats


def shard_contains_doc_tablevqa(shard_path, page_id):
    jsonl_path = os.path.join(shard_path, "docstore_data.jsonl")
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            fname = obj["meta"]["file"]
            if page_id in fname or fname.startswith(page_id):
                return True
    return False


def eval_tablevqa(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="tablevqa full pipeline"):
        qa_id = item.get("qa_id", "")
        q_text = item.get("question", "")
        q_a = item.get("gt", item.get("answer", ""))
        if not q_text or not qa_id:
            continue

        page_id = qa_id.split(".")[0]

        shard_path = None
        for s in shards:
            full_path = os.path.join(store_path, s)
            if shard_contains_doc_tablevqa(full_path, page_id):
                shard_path = full_path
                break

        if shard_path is None:
            print(f"WARNING: doc {page_id} not found in any shard")
            continue

        index = VectorStore.load(os.path.join(shard_path, "docstore"))

        q_embed = embedder.encode([q_text]).astype("float32")
        retrieved = index.search(q_embed, k=top_k)

        context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
        if not context_text:
            continue

        system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
        user_prompt = f"/no_think {context_text}\nQuestion: {q_text}\nAnswer: "
        raw_response = llm.generate(system_prompt, user_prompt)

        if not isinstance(q_a, list):
            q_a = [q_a]
        q_a = [ans for ans in q_a if str(ans).strip()]  # skip empty answers
        if not q_a:
            continue
        print(f"Question: {q_text}")
        print(f"Ground Truth Answers: {q_a}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(normalize_answer_tablevqa(ans, raw_response) for ans in q_a):
            correct += 1
            print("correct!")
        else:
            mistakes.append((q_text, q_a, raw_response))
            print("incorrect!")

        print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    return stats

def eval_finqa(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        if json_path.endswith(".jsonl"):
            gt = [json.loads(line) for line in f if line.strip()]
        else:
            gt = json.load(f)

    correct = 0
    total = 0
    skipped = 0
    mistakes = []

    index = VectorStore.load(os.path.join(store_path, "docstore"))
    _raw_pages = set(m['file'].split('.')[0] for m in index.metadata)
    pages_in_index = _raw_pages | {p.replace("/", "_") for p in _raw_pages}
    print(f'Pages in index: {len(_raw_pages)}')

    for item in tqdm(gt, desc="finqa full pipeline"):
        # Normalise dataset file_name: "pdf/AAP/2006/page_85.pdf" -> "AAP/2006/page_85"
        doc_id = item["file_name"].replace("pdf/", "").replace(".pdf", "").replace("\\", "")
        doc_id_slash = doc_id
        doc_id = doc_id.replace("/", "_")  # e.g. "AAP_2006_page_85"

        if doc_id not in pages_in_index and doc_id_slash not in pages_in_index:
            skipped += 1
            continue
        question = item["question"]
        answer = item["program_answer"]

        q_embed = embedder.encode([question]).astype("float32")
        retrieved = index.search(q_embed, k=top_k)

        context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
        if not context_text:
            continue

        system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
        user_prompt = f"/no_think {context_text}\nQuestion: {question}\nAnswer: "
        raw_response = llm.generate(system_prompt, user_prompt)

        if not isinstance(answer, list):
            answer = [answer]
        print(f"Question: {question}")
        print(f"Ground Truth Answers: {answer}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(answer_matched_finqa(ans, raw_response) for ans in answer):
            correct += 1
            print("correct!")
        else:
            mistakes.append((question, answer, raw_response))
            print("incorrect!")

        print(f"Current Accuracy: {correct/total:.4f} | # correct: {correct}, # total: {total}")

    print(f"Skipped {skipped} / {len(gt)} questions (doc not in store)")
    stats = accuracy_with_std(correct, total)
    stats["mistakes"] = mistakes
    stats["skipped"] = skipped

    return stats

def build_parser():
    parser = argparse.ArgumentParser(description="Run full retrieval and generation evaluation")
    parser.add_argument("--dataset", type=str, default="tatdqa", choices=["tatdqa", "mpdocvqa", "wikitq", "comtqa", "tablevqa", "finqa"], help="dataset to evaluate on")
    parser.add_argument("--method", type=str, default="tabrag", help="method/baseline, e.g. tabrag, vlm, pymupdf")
    parser.add_argument("--vlm_model", type=str, default="Qwen3-VL-8B-Instruct", help="VLM used to build the retrieval store")
    parser.add_argument("--json_path", type=str, default=None, help="override the default ground-truth path")
    parser.add_argument("--store_path", type=str, default=None, help="override storages/{dataset}/retrieval/{method}/{vlm_model}")
    parser.add_argument("--top_k", type=int, default=3)
    return parser


def main(args):
    dataset_dir = args.dataset
    args.json_path = args.json_path or DEFAULT_GT.get(dataset_dir)
    args.store_path = args.store_path or os.path.join(
        "storages", dataset_dir, "retrieval", args.method, args.vlm_model
    )

    if args.json_path is None:
        raise ValueError(f"No default ground-truth path for dataset: {args.dataset}")
    if not os.path.isdir(args.store_path):
        raise FileNotFoundError(f"Store path does not exist: {args.store_path}")

    llm = HFLLMClient("Qwen/Qwen3.6-27B")
    embedder = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-8B")
    # if args.dataset == "tatdqa_tabrag":
    #     result = eval_tatdqa_n(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
    #     print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    if args.dataset in {"tatdqa"}:
        # result = eval_tatdqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        result = eval_tatdqa_n(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "mpdocvqa":
        result = eval_mpdocvqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "wikitq":
        result = eval_wikitq(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "comtqa":
        result = eval_comtqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "tablevqa":
        result = eval_tablevqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "finqa":
        result = eval_finqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")

    record = {
        "dataset": args.dataset,
        "method": args.method,
        "vlm_model": args.vlm_model,
        "llm_model": "Qwen/Qwen3.6-27B",
        "embedder_model": "Qwen/Qwen3-Embedding-8B",
        "json_path": args.json_path,
        "store_path": args.store_path,
        "top_k": args.top_k,
        "result": result,
    }

    results_path = os.path.join(
        "results",
        args.dataset,
        "full",
        args.method,
        args.vlm_model,
        "results.json",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())