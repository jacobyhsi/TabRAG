#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

import weasyprint
from weasyprint import HTML
import re
import json


# ============================================================
# PART 1 — HTML → PDF
# ============================================================
def html_to_pdf(html_path: Path, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_text = f.read()
    HTML(string=html_text, base_url=str(html_path.parent)).write_pdf(str(pdf_path))


# ============================================================
# PART 2 — PDF → PNG (first page)
# ============================================================
def pdf_to_png_first_page(pdf_path: Path, png_path: Path, dpi=288):
    import fitz  # PyMuPDF
    png_path.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(pdf_path)) as doc:
        if doc.page_count == 0:
            raise RuntimeError(f"No pages in PDF: {pdf_path}")

        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(png_path))


# ============================================================
# PART 3 — Convert a single HTML into PDF + PNG
# ============================================================
def process_one(html_path: Path, in_root: Path, out_pdf: Path, out_png: Path):
    group_dir = html_path.parent.name  # e.g. 200-csv
    stem = html_path.stem              # e.g. 0

    if not group_dir.endswith("-csv"):
        return

    out_leaf = f"{group_dir}_p{stem}"
    pdf_path = out_pdf / group_dir / out_leaf / f"{out_leaf}.pdf"
    png_path = out_png / group_dir / out_leaf / f"{out_leaf}.png"

    # Always rebuild to ensure consistency
    html_to_pdf(html_path, pdf_path)
    print(f"[PDF] {pdf_path}")

    pdf_to_png_first_page(pdf_path, png_path)
    print(f"[PNG] {png_path}")


# ============================================================
# PART 4 — MAIN HTML→PDF→PNG PIPELINE
# ============================================================
def run_conversion_pipeline():
    parser = argparse.ArgumentParser(description="Convert WikiTableQuestions HTMLs to PDF & PNG.")
    parser.add_argument("--in-root", type=Path, default="wikitablequestions/csv")
    parser.add_argument("--out-png", type=Path, default="wikitablequestions/png")
    parser.add_argument("--out-pdf", type=Path, default="wikitablequestions/pdf")
    args = parser.parse_args()

    if not args.in_root.exists():
        print(f"Input root not found: {args.in_root}", file=sys.stderr)
        sys.exit(1)

    html_files = []
    for group in sorted(args.in_root.glob("*-csv")):
        if group.is_dir():
            html_files.extend(sorted(group.glob("*.html")))

    if not html_files:
        print("No HTML files found.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(html_files)} HTML files.")

    for html_path in html_files:
        try:
            process_one(html_path, args.in_root, args.out_pdf, args.out_png)
        except Exception as e:
            print(f"[ERROR] {html_path}: {e}", file=sys.stderr)

    return args.out_png, args.out_pdf


# ============================================================
# PART 5 — Build retrieval folders for PNG + PDF
# ============================================================
def build_retrieval_folders(base_dir="wikitablequestions"):
    print("\n=== Building retrieval folders ===\n")

    modes = ["png", "pdf"]

    for mode in modes:
        src_base = os.path.join(base_dir, mode)
        dst_base = os.path.join(base_dir, f"{mode}_retrieval")
        os.makedirs(dst_base, exist_ok=True)

        for folder in tqdm(os.listdir(src_base), desc=f"Processing {mode.upper()} folders"):
            folder_path = os.path.join(src_base, folder)
            if not os.path.isdir(folder_path):
                continue

            dst_folder = os.path.join(dst_base, folder)
            os.makedirs(dst_folder, exist_ok=True)

            # example: png/200-csv/200-csv_p0/200-csv_p0.png
            for subdir in os.listdir(folder_path):
                subdir_path = os.path.join(folder_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                for file in os.listdir(subdir_path):
                    if not (file.endswith(".png") or file.endswith(".pdf")):
                        continue

                    src = os.path.join(subdir_path, file)
                    dst = os.path.join(dst_folder, file)

                    if os.path.exists(dst):
                        continue

                    shutil.copy2(src, dst)

    print("✔ Retrieval folders built.\n")


# ============================================================
# PART 6 — Parse WTQ training.examples → qa.json
# ============================================================
def build_wtq_qa_json(
    input_file="wikitablequestions/data/training.examples",
    output_file="wikitablequestions/qa.json"
):

    print("\n=== Parsing training.examples → qa.json ===\n")

    def parse_examples(text):
        examples = []
        blocks = re.findall(r"\(example\b.*?(?=\(example\b|$)", text, re.DOTALL)

        for block in blocks:
            ex = {}

            id_match = re.search(r"\(id\s+([^\)]+)\)", block)
            utt_match = re.search(r'\(utterance\s+"(.*?)"\)', block)
            context_match = re.search(
                r'\(context\s+\(graph\s+tables\.TableKnowledgeGraph\s+([^\)]+)\)\)',
                block
            )
            target_vals = re.findall(r'\(description\s+"(.*?)"\)', block)
            formula_match = re.search(r'\(targetFormula\s+([^\)]+)\)', block)
            error_match = re.search(r'\(error\s+"(.*?)"\)', block)

            if context_match:
                context_path = context_match.group(1).strip()

                # Exclusion rule
                if "203-csv" in context_path or "204-csv" in context_path:
                    continue

                ex["context"] = context_path

            if id_match:
                ex["id"] = id_match.group(1).strip()

            if utt_match:
                ex["utterance"] = utt_match.group(1).strip()

            if target_vals:
                ex["targetValue"] = target_vals

            if formula_match:
                ex["targetFormula"] = formula_match.group(1).strip()

            if error_match:
                ex["error"] = error_match.group(1).strip()

            examples.append(ex)

        return examples

    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read()

    examples = parse_examples(raw)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"✔ Saved {len(examples)} examples to {output_file}\n")


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    # Step 1–4: Convert HTML → PDF → PNG
    out_png, out_pdf = run_conversion_pipeline()

    # Step 5: Build retrieval folders
    build_retrieval_folders("wikitablequestions")

    # Step 6: Convert training.examples → qa.json
    build_wtq_qa_json()

    print("\n=== All WikiTQ tasks completed successfully. ===\n")
