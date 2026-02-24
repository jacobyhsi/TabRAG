#!/usr/bin/env python3
import json
import os
import shutil
from collections import Counter
import fitz
from PIL import Image

# --------------------------------------------------
# Config
# --------------------------------------------------
TOP_K = 250

QA_JSON = "datasets/comtqa/annotated.json"

# FinTabNet
FINTABNET_JSONL = "datasets/fintabnet/FinTabNet_1.0.0_cell_test.jsonl"
PDF_DIR = "datasets/fintabnet/pdf"

# PubTables-1M
PUBTABLES_IMG_DIR = "datasets/pubtables-1m"

# Output
OUT_PNG = "datasets/comtqa/generation"
os.makedirs(OUT_PNG, exist_ok=True)

# --------------------------------------------------
# Load QA annotations
# --------------------------------------------------
with open(QA_JSON, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# --------------------------------------------------
# Count frequencies
# --------------------------------------------------
table_id_counter = Counter()
image_name_counter = Counter()

for q in qa_data:
    if q.get("table_id") is not None:
        table_id_counter[str(q["table_id"])] += 1
    if q.get("image_name") is not None:
        image_name_counter[q["image_name"]] += 1

# --------------------------------------------------
# Select TOP-K
# --------------------------------------------------
top_table_ids = set(tid for tid, _ in table_id_counter.most_common(TOP_K))
top_image_names = set(img for img, _ in image_name_counter.most_common(TOP_K))

print(f"[INFO] Selected {len(top_table_ids)} FinTabNet table_id")
print(f"[INFO] Selected {len(top_image_names)} PubTables image_name")

# --------------------------------------------------
# === PubTables-1M ===
# --------------------------------------------------
print("[INFO] Processing PubTables-1M images...")

for img_name in top_image_names:
    src = os.path.join(PUBTABLES_IMG_DIR, img_name)
    if not os.path.exists(src):
        continue

    img_id = os.path.splitext(img_name)[0]
    out_dir = os.path.join(OUT_PNG, img_id)
    os.makedirs(out_dir, exist_ok=True)

    dst = os.path.join(out_dir, img_name)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

# --------------------------------------------------
# === FinTabNet ===
# --------------------------------------------------
print("[INFO] Processing FinTabNet PDFs...")

# Map table_id → pdf filename
tableid_to_file = {}
with open(FINTABNET_JSONL, "r") as f:
    for line in f:
        s = json.loads(line)
        tid = str(s.get("table_id"))
        if tid in top_table_ids:
            tableid_to_file[tid] = s["filename"]

for table_id in top_table_ids:
    pdf_file = tableid_to_file.get(table_id)
    if pdf_file is None:
        continue

    pdf_path = os.path.join(PDF_DIR, pdf_file)
    if not os.path.exists(pdf_path):
        continue

    out_dir = os.path.join(OUT_PNG, table_id)
    os.makedirs(out_dir, exist_ok=True)

    out_png = os.path.join(out_dir, f"{table_id}.png")
    if os.path.exists(out_png):
        continue

    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(
        matrix=fitz.Matrix(144 / 72, 144 / 72),
        alpha=False
    )

    Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    ).save(out_png)

    doc.close()

print("[DONE] COMTQA filtered dataset built successfully")
