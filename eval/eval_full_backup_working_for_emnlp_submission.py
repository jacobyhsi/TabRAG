
import argparse
import json
from operator import index
import os

from tqdm import tqdm

from src.embedder import SentenceTransformerEmbedder
from src.llm import HFLLMClient
from src.utils import normalize_answer, normalize_answer_finqa
from src.vector_store import VectorStore
import re


def accuracy_with_std(correct, total):
    if total == 0:
        return {"accuracy": 0.0, "std": 0.0, "n": 0}
    acc = correct / total
    std = (acc * (1 - acc) / total) ** 0.5
    return {"accuracy": acc, "std": std, "n": total}


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
    for item in tqdm(gt, desc="TATDQA full pipeline"):
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
    for item in tqdm(gt, desc="TATDQA full pipeline"):
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


def eval_MPDocVQA(json_path, store_path, llm, embedder, top_k=3):
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


def eval_WikiTQ(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        if json_path.endswith(".jsonl"):
            gt = [json.loads(line) for line in f if line.strip()]
        else:
            gt = json.load(f)

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="WikiTQ full pipeline"):
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


def eval_ComTQA(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="ComTQA full pipeline"):
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


def eval_TableVQA(json_path, store_path, llm, embedder, top_k=3):
    with open(json_path, "r") as f:
        gt = json.load(f)

    shards = [
        d for d in os.listdir(store_path)
        if d.startswith("folder") and os.path.isdir(os.path.join(store_path, d))
    ]

    correct = 0
    total = 0
    mistakes = []

    for item in tqdm(gt, desc="TableVQA full pipeline"):
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
        raw_response = normalize_answer(raw_response)

        if not isinstance(q_a, list):
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

def eval_convfinqa(json_path, store_path, llm, embedder, top_k=3):
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
    pages_in_index = set(m['file'].split('.')[0] for m in index.metadata)
    print(f'Pages in index: {len(pages_in_index)}')

    for item in tqdm(gt, desc="ConvFinQA full pipeline"):
        # Normalise dataset file_name: "pdf/AAP/2006/page_85.pdf" -> "AAP/2006/page_85"
        doc_id = item["file_name"].replace("pdf/", "").replace(".pdf", "").replace("\\", "")
        # doc_id = doc_id.replace("/", "_")  # e.g. "AAP_2006_page_85"

        if doc_id not in pages_in_index:
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
        raw_response = normalize_answer(raw_response)

        if not isinstance(answer, list):
            answer = [answer]
        answer = [normalize_answer(str(ans)) for ans in answer]
        print(f"Question: {question}")
        print(f"Ground Truth Answers: {answer}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(ans in raw_response for ans in answer):
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

    for item in tqdm(gt, desc="ConvFinQA full pipeline"):
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
        raw_response = normalize_answer_finqa(raw_response)

        if not isinstance(answer, list):
            answer = [answer]
        answer = [normalize_answer_finqa(str(ans)) for ans in answer]
        print(f"Question: {question}")
        print(f"Ground Truth Answers: {answer}")
        print(f"Raw Response: {raw_response}")

        total += 1

        if all(ans in raw_response for ans in answer):
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

# def eval_finqa(json_path, store_path, llm, embedder, top_k=3):
#     with open(json_path, "r") as f:
#         if json_path.endswith(".jsonl"):
#             gt = [json.loads(line) for line in f if line.strip()]
#         else:
#             gt = json.load(f)

#     correct = 0
#     total = 0
#     skipped = 0
#     mistakes = []

#     index = VectorStore.load(os.path.join(store_path, "docstore"))
#     pages_in_index = set(m['file'].split('.')[0] for m in index.metadata)
#     print(f'Pages in index: {len(pages_in_index)}')

#     for item in tqdm(gt, desc="finqa full pipeline"):
#         # Normalise dataset file_name: "pdf/AAP/2006/page_85.pdf" -> "AAP/2006/page_85"
#         doc_id = item["file_name"].replace("pdf/", "").replace(".pdf", "").replace("\\", "")
#         # doc_id = doc_id.replace("/", "_")  # e.g. "AAP_2006_page_85"

#         if doc_id not in pages_in_index:
#             skipped += 1
#             continue

#         question = item["question"]
#         answer = item["program_answer"]

#         q_embed = embedder.encode([question]).astype("float32")
#         retrieved = index.search(q_embed, k=top_k)

#         context_text = "\n".join(r["text"] for r in retrieved if r.get("text"))
#         if not context_text:
#             continue

#         total += 1
#         system_prompt = "You are a helpful assistant. Use the information from the documents below to answer the question."
#         user_prompt = f"/no_think {context_text}\nQuestion: {question}\nAnswer: "
#         raw_response = llm.generate(system_prompt, user_prompt)
#         raw_response = normalize_answer_finqa(raw_response)

#         if not isinstance(answer, list):
#             answer = [answer]
#         answer = [normalize_answer_finqa(str(ans)) for ans in answer]

#         print(f"Question: {question}")
#         print(f"Ground Truth Answers: {answer}")
#         print(f"Raw Response: {raw_response}")

#         if all(ans in raw_response for ans in answer):
#             correct += 1
#             print("Correct")
#         else:
#             mistakes.append((question, answer, raw_response))
#             print("Incorrect")

#         print(f'Current Accuracy: {correct/total} | # correct: {correct}, # total: {total}')

#     print(f"Skipped {skipped} / {len(gt)} questions (doc not in store)")
#     stats = accuracy_with_std(correct, total)
#     stats["mistakes"] = mistakes
#     stats["skipped"] = skipped
#     return stats

def build_parser():
    parser = argparse.ArgumentParser(description="Full pipeline TATDQA evaluation")
    parser.add_argument("--dataset", type=str, default="tatdqa", choices=["tatdqa", "mpdocvqa", "wikitq", "comtqa", "tablevqa", "tatdqa_b", "tatdqa_tabrag", "convfinqa","finqa"], help="Dataset to evaluate on")
    parser.add_argument("--json_path", type=str, default="datasets/tatdqa/tatdqa_dataset_test_gold.json")
    parser.add_argument("--store_path", type=str, default="storages/tatdqa/retrieval/tabrag_3/Qwen3-VL-8B-Instruct")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--embedder_model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--top_k", type=int, default=3)
    return parser


def main(args):
    llm = HFLLMClient(args.llm_model)
    embedder = SentenceTransformerEmbedder(args.embedder_model)
    if args.dataset == "tatdqa_tabrag":
        result = eval_tatdqa_n(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "tatdqa_b":
        result = eval_tatdqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "mpdocvqa":
        result = eval_MPDocVQA(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "wikitq":
        result = eval_WikiTQ(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "comtqa":
        result = eval_ComTQA(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "tablevqa":
        result = eval_TableVQA(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "convfinqa":
        result = eval_convfinqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")
    elif args.dataset == "finqa":
        result = eval_finqa(args.json_path, args.store_path, llm, embedder, top_k=args.top_k)
        print(f"Final Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f} (n={result['n']})")

if __name__ == "__main__":
    print("lol")
    parser = build_parser()
    main(parser.parse_args())