import os
import sys
import re
import json
import arxiv
import pymupdf
import unicodedata
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download

def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def build_caption_pattern(label: str) -> re.Pattern:
    label = normalize_unicode(label).strip()
    m = re.match(r'(?i)\s*(figure|fig\.?|table)\s*([0-9]+)', label)
    if not m:
        raise ValueError(f"Invalid caption format: {label}")
    word, num = m.groups()
    if word.lower().startswith("fig"):
        prefix_pattern = r"(?:fig(?:ure)?\.?)"
    else:
        prefix_pattern = r"(?:table\.?)"
    pattern = rf"\b{prefix_pattern}\s*{num}\s*[:\.]?\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def separate_letters_numbers(s: str) -> str:
    return re.sub(r'([A-Za-z])(\d)', r'\1 \2', s)

def split_pdf_to_pages(pdf_path: str, png_root: str, pdf_root: str,
                       png_flat_root: str, pdf_flat_root: str,
                       ref_root: str, dpi: int = 108):
    doc = pymupdf.open(pdf_path)
    base = Path(pdf_path).stem
    if base not in paper_ids:
        return

    # ---- parent dirs ----
    parent_png = Path(png_root) / base
    parent_pdf = Path(pdf_root) / base
    parent_png_flat = Path(png_flat_root) / base
    parent_pdf_flat = Path(pdf_flat_root) / base
    parent_png.mkdir(parents=True, exist_ok=True)
    parent_pdf.mkdir(parents=True, exist_ok=True)
    parent_png_flat.mkdir(parents=True, exist_ok=True)
    parent_pdf_flat.mkdir(parents=True, exist_ok=True)

    refs_dir = Path(ref_root)
    refs_dir.mkdir(exist_ok=True)
    out_json = refs_dir / f"{base}.json"

    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    meta = doc.metadata or {}
    figure_to_page = {}
    figures_map = json_data[base]['all_figures']

    for page_idx in range(doc.page_count):
        page_num = page_idx + 1

        # === Nested layout ===
        page_png_dir = parent_png / f"{base}_p{page_num}"
        page_png_dir.mkdir(parents=True, exist_ok=True)
        png_path_nested = page_png_dir / f"{base}_p{page_num}.png"

        page_pdf_dir = parent_pdf / f"{base}_p{page_num}"
        page_pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path_nested = page_pdf_dir / f"{base}_p{page_num}.pdf"

        # === Flat layout ===
        png_path_flat = parent_png_flat / f"{base}_p{page_num}.png"
        pdf_path_flat = parent_pdf_flat / f"{base}_p{page_num}.pdf"

        # --- Caption assignment ---
        for search_idx in range(len(doc)):
            page = doc[search_idx]
            page_text = page.get_text("text")
            for figure in figures_map.keys():
                if figure in figure_to_page.keys():
                    continue
                fig_label = separate_letters_numbers(figure.split('-')[1]) + ':'
                fig_pattern = build_caption_pattern(fig_label)
                if fig_pattern.search(page_text):
                    figure_to_page[fig_label] = search_idx

        # --- Export images & PDFs ---
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # save both nested + flat PNGs
        pix.save(png_path_nested.as_posix())
        pix.save(png_path_flat.as_posix())

        # save both nested + flat PDFs
        for out_pdf in [pdf_path_nested, pdf_path_flat]:
            if not out_pdf.exists():
                one = pymupdf.open()
                one.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                if meta:
                    one.set_metadata(meta)
                one.save(out_pdf.as_posix())
                one.close()

    doc.close()
    print(f'{base}: Successfully assigned pages to {len(figure_to_page)} of {len(figures_map)} pages')
    if len(figures_map) != len(figure_to_page):
        print(figures_map.keys())
        print(figure_to_page)
        exit()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(figure_to_page, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if not os.path.exists(os.path.join(ROOT, 'datasets/spiqa/test')):
        snapshot_download(repo_id="google/spiqa", repo_type="dataset", local_dir=os.path.join(ROOT, 'datasets/spiqa')) 

    SPIQA_PATH = os.path.join(ROOT, 'datasets/spiqa/test-A')
    DATA_PATH = os.path.join(ROOT, 'datasets/spiqa/test')
    PNG_PATH = os.path.join(ROOT, 'datasets/spiqa/generation') # png
    PDF_PATH = os.path.join(ROOT, 'datasets/spiqa/pdf_generation')
    PNG_FLAT_PATH = os.path.join(ROOT, 'datasets/spiqa/retrieval') # png
    PDF_FLAT_PATH = os.path.join(ROOT, 'datasets/spiqa/pdf_retrieval')
    REFS_PATH = os.path.join(ROOT, 'datasets/spiqa/refs')

    for p in [PNG_PATH, PDF_PATH, PNG_FLAT_PATH, PDF_FLAT_PATH, REFS_PATH]:
        os.makedirs(p, exist_ok=True)

    with open(os.path.join(SPIQA_PATH, 'SPIQA_testA.json'), 'r') as file:
        json_data = json.load(file)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    client = arxiv.Client()
    docs = json_data.keys()
    print("- Downloading arxiv pdfs ...")
    paper_ids = ['1805.06431v4', '1805.04687v2', '1805.01216v3', '1906.06589v3', '1704.07121v2', '1707.08608v3', '1901.00398v2', '1811.02721v3', '1803.01128v3', '1707.01922v5', '1805.06447v3', '1706.00827v2', '1703.02507v3', '1812.00281v3', '1811.08481v2', '1803.04572v2', '1708.00160v2', '1901.00056v2', '1809.03550v3', '1805.00912v4', '1802.07351v2', '1706.00633v4', '1705.09296v2', '1611.04684v1', '1906.10843v1', '1811.09393v4', '1811.08257v1', '1809.04276v2', '1809.02731v3', '1804.05936v2', '1803.03467v4', '1709.08294v3', '1707.06320v2', '1707.01917v2', '1705.10667v4', '1705.02798v6', '1704.05426v4', '1611.03780v2', '1611.02654v2', '1812.10735v2']

    for doc in tqdm(docs):
        paper_id = json_data[doc]['paper_id']
        if paper_id not in paper_ids:
            continue
        if os.path.exists(os.path.join(DATA_PATH, f'{paper_id}.pdf')):
            continue
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))
        paper.download_pdf(dirpath=DATA_PATH, filename=f'{paper_id}.pdf')

    print(f"- Splitting pdfs in {DATA_PATH} ...")
    pdf_files = sorted([f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")])
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"[WARN] Skipping missing file: {pdf_path}")
            continue
        try:
            split_pdf_to_pages(pdf_path, PNG_PATH, PDF_PATH, PNG_FLAT_PATH, PDF_FLAT_PATH, REFS_PATH, dpi=288)
            os.remove(pdf_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")
            continue
