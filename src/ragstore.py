import os
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm

from src.layout import LayoutProcessor
from src.embedder import BaseEmbedder
from src.vector_store import VectorStore
from src.llm import BaseVLMEngine, BaseLLMEngine
from src.prompts import LLMPrompts, VLMPrompts
from tqdm import tqdm
class Ragstore:
    def __init__(self, 
                lp: LayoutProcessor,
                embedder: BaseEmbedder,
                vlm: BaseLLMEngine, 
                llm: BaseVLMEngine,
                vlm_prompts: VLMPrompts,
                llm_prompts: LLMPrompts,
                data_dir,
                save_dir):
        self.lp = lp
        self.embedder = embedder
        self.vlm = vlm
        self.llm = llm
        self.vlm_prompts = vlm_prompts
        self.llm_prompts = llm_prompts
        self.data_dir = data_dir
        self.save_dir = save_dir

    def process_one_file(self, file_path, file_name):
        # file_path = os.path.join(self.data_dir, file_name)
        comps = self.lp.preprocess(file_path)

        # Sort components top-to-bottom by bbox midpoint
        comps_sorted = [
            sorted(
                range(len(comps[page])),
                key=lambda i: (comps[page][i]['bbox'][1] + comps[page][i]['bbox'][3]) / 2
            )
            for page in comps
        ]

        output_emb = []
        output_data = []
        output_meta = []

        for page_idx, component_indices in enumerate(comps_sorted):
            page_texts = []
            page_components = []

            for local_idx, comp_idx in enumerate(component_indices):
                component = comps[page_idx][comp_idx]
                component_class = component['class']
                vlm_image = component['path']
                vlm_prompt = self.vlm_prompts.prompt_map[component_class]

                # Extract text with VLM
                output = self.vlm.generate(vlm_prompt, vlm_image)

                # Special handling for tables (class 3)
                if component_class == 3:
                    llm_prompt = self.llm_prompts.prompt_map[component_class]
                    prev_context = ""
                    if local_idx > 0:
                        prev_idx = component_indices[local_idx - 1]
                        prev_context = comps[page_idx][prev_idx].get('details', '')
                    output = self.llm.generate(llm_prompt, prev_context + '\n\n' + output)

                # Collect component-level info
                page_texts.append(output)
                page_components.append({
                    "text": output,
                    "bbox": component['bbox'],
                    "mode": "component"
                })

            # Concatenate all components' text for the page
            page_text = "\n".join(page_texts)

            # Page-level metadata with sub-components
            page_meta = {
                "file": file_name,
                "page": page_idx,
                "mode": "page",
                "components": page_components
            }

            # Store
            output_data.append(page_text)
            output_meta.append(page_meta)
            output_emb.append(self.embedder.encode([page_text]))

        return output_data, output_meta, output_emb
    
    def process_one_file_components(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        comps = self.lp.preprocess(file_path)

        comps_sorted = [
            sorted(
                range(len(comps[page])),
                key=lambda i: (comps[page][i]['bbox'][1] + comps[page][i]['bbox'][3]) / 2
            )
            for page in comps
        ]

        output_emb = []
        output_data = []
        output_meta = []

        for page_idx, component_indices in enumerate(comps_sorted):
            for local_idx, comp_idx in enumerate(component_indices):
                component = comps[page_idx][comp_idx]
                component_class = component['class']
                vlm_image = component['path']
                vlm_prompt = self.vlm_prompts.prompt_map[component_class]

                output = self.vlm.generate(vlm_prompt, vlm_image)

                if component_class == 3:
                    llm_prompt = self.llm_prompts.prompt_map[component_class]
                    prev_context = ""
                    if local_idx > 0:
                        prev_idx = component_indices[local_idx - 1]
                        prev_context = comps[page_idx][prev_idx].get('details', '')
                    output = self.llm.generate(llm_prompt, prev_context + '\n\n' + output)

                metadata = {
                    "file": file_name,
                    "bbox": component['bbox'],
                }

                output_data.append(output)
                output_meta.append(metadata)
                output_emb.append(self.embedder.encode([output]))

        return output_data, output_meta, output_emb

    # def aggregate_to_pages(self, output_data, output_meta, output_emb):
    #     pages = {}
    #     for text, meta, emb in zip(output_data, output_meta, output_emb):
    #         file_name = meta["file"]
    #         if file_name not in pages:
    #             pages[file_name] = {
    #                 "texts": []
    #             }
    #         pages[file_name]["texts"].append(text)

    #     page_data, page_meta, page_emb = [], [], []
    #     for file_name in sorted(pages.keys()):
    #         texts = pages[file_name]["texts"]
    #         full_text = "\n".join(texts)

    #         meta = {
    #             "file": file_name,
    #             "mode": "file"   # clarify this is file-level now
    #         }

    #         page_data.append(full_text)
    #         page_meta.append(meta)
    #         page_emb.append(self.embedder.encode([full_text]))

    #     return page_data, page_meta, page_emb

    def build_index_per_file(self):

        # Step 1: Preprocess pdfs into pagesa
        staging_dir = os.path.join(self.data_dir, "test")
        os.makedirs(staging_dir, exist_ok=True)

        files = [
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.data_dir, f))
        ]

        for f in tqdm(files, desc="Preprocessing PDFs"):
            src_path = os.path.join(self.data_dir, f)
            dst_path = os.path.join(staging_dir, f)

            with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
                dst.write(src.read())

        # Step 2: Build index from preprocessed PDFs
        files = [
            f for f in os.listdir(staging_dir)
            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(staging_dir, f))
        ]

        for f in tqdm(files, desc="Building indices"):
            raw_name = f[:-4]
            emb_dim = self.embedder.get_dims()
            index = VectorStore(emb_dim)
            output_data, output_meta, output_emb = self.process_one_file(file_path=os.path.join(staging_dir, f), file_name=f)
            for d, m, e in zip(output_data, output_meta, output_emb):
                index.add(e, d, m)

            save_path = os.path.join(self.save_dir, raw_name)
            os.makedirs(save_path, exist_ok=True)
            index.save(os.path.join(save_path, "docstore"))


    # def build_index_per_file(self):
    #     files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.data_dir, f))]
    #     for f in tqdm(files):
    #         raw_name = f[:-4]
    #         emb_dim = self.embedder.get_dims()
    #         index = VectorStore(emb_dim)
    #         output_data, output_meta, output_emb = self.process_one_file(f)
    #         for d, m, e in zip(output_data, output_meta, output_emb):
    #             index.add(e, d, m)
    #         if not os.path.exists(os.path.join(self.save_dir, raw_name)):
    #             os.makedirs(os.path.join(self.save_dir, raw_name))
    #         index.save(os.path.join(self.save_dir, raw_name, 'docstore'))

    # def build_index_for_folders(self):
    #     exts = ('.pdf', '.jpg', '.jpeg', '.png')
    #     files = sorted(
    #         f for f in os.listdir(self.data_dir)
    #         if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.data_dir, f))
    #     )

    #     emb_dim = self.embedder.get_dims()
    #     os.makedirs(self.save_dir, exist_ok=True)

    #     # One page-level index per document
    #     page_index = VectorStore(emb_dim)

    #     # Sort pages numerically by _p<number>
    #     files = sorted(
    #         files,
    #         key=lambda x: int(os.path.splitext(x)[0].split('_p')[-1]) if '_p' in x else 0
    #     )

    #     for f in tqdm(files[3:]):
    #         comp_data, comp_meta, comp_emb = self.process_one_file_components(f)

    #         if not comp_data or not comp_emb:
    #             print(f"[skip] no components for {f}")
    #             continue

    #         page_components_store = VectorStore(emb_dim)
    #         for d, m, e in zip(comp_data, comp_meta, comp_emb):
    #             if getattr(e, "ndim", 2) == 1:
    #                 e = e.reshape(1, -1)
    #             page_components_store.add(e.astype("float32"), d, m)

    #         page_stem, _ = os.path.splitext(f)
    #         page_components_store.save(os.path.join(self.save_dir, f"{page_stem}_components"))

    #         page_text = "\n".join(t for t in comp_data if isinstance(t, str) and t.strip())

    #         p_emb = self.embedder.encode([page_text])

    #         page_meta = {"file": f, "mode": "file"}
    #         page_index.add(p_emb.astype("float32"), page_text, page_meta)

    #         doc_id = os.path.basename(os.path.normpath(self.data_dir))
    #         page_index.save(os.path.join(self.save_dir, f"{doc_id}_pages"))

    # def build_index_per_file(self):
    #     os.makedirs(self.save_dir, exist_ok=True)

    #     files = [
    #         f for f in os.listdir(self.data_dir)
    #         if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.data_dir, f))
    #     ]

    #     for f in tqdm(files):
    #         pdf_path = os.path.join(self.data_dir, f)
    #         reader = PdfReader(pdf_path)
    #         base_name = os.path.splitext(f)[0]

    #         # If multi-page: split and save; else: use original
    #         if len(reader.pages) > 1:
    #             page_dir = os.path.join(self.save_dir, base_name)
    #             os.makedirs(page_dir, exist_ok=True)

    #             pdfs_to_process = []
    #             for i, page in enumerate(reader.pages, start=1):
    #                 writer = PdfWriter()
    #                 writer.add_page(page)
    #                 out_name = f"{base_name}_p{i}.pdf"
    #                 out_path = os.path.join(page_dir, out_name)
    #                 with open(out_path, "wb") as out_f:
    #                     writer.write(out_f)
    #                 pdfs_to_process.append(out_path)
    #         else:
    #             pdfs_to_process = [pdf_path]

    #         # Build index for each (saved split or original)
    #         emb_dim = self.embedder.get_dims()
    #         for pdf in pdfs_to_process:
    #             print(pdf)
    #             raw_name = os.path.splitext(os.path.basename(pdf))[0]
    #             index = VectorStore(emb_dim)

    #             # Use the saved file path (for split) or original path (single-page)
    #             output_data, output_meta, output_emb = self.process_one_file(pdf)
    #             for d, m, e in zip(output_data, output_meta, output_emb):
    #                 index.add(e, d, m)

    #             save_dir = os.path.join(self.save_dir, raw_name)
    #             os.makedirs(save_dir, exist_ok=True)
    #             index.save(os.path.join(save_dir, "docstore"))

    # def build_index_per_folder(self):
    #     exts = ('.pdf', '.jpg', '.jpeg', '.png')

    #     # List only unprocessed folders
    #     all_folders = [
    #         f for f in os.listdir(self.data_dir)
    #         if os.path.isdir(os.path.join(self.data_dir, f))
    #     ]
    #     unprocessed_folders = [
    #         f for f in all_folders
    #         if not os.path.exists(os.path.join(self.save_dir, f))
    #     ]

    #     print(f"[Info] Found {len(unprocessed_folders)} folders to process "
    #         f"({len(all_folders) - len(unprocessed_folders)} already done).")

    #     # Process only unprocessed folders
    #     for folder_name in tqdm(unprocessed_folders):
    #         folder_path = os.path.join(self.data_dir, folder_name)

    #         # Collect all valid page files in sorted order (relative to self.data_dir)
    #         page_files = [
    #             os.path.join(folder_name, f)  # relative path
    #             for f in sorted(os.listdir(folder_path))
    #             if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder_path, f))
    #         ]
    #         if not page_files:
    #             continue  # skip empty folders

    #         # Create a fresh index for this document
    #         emb_dim = self.embedder.get_dims()
    #         index = VectorStore(emb_dim)

    #         # Process each page in the folder
    #         for file_name in page_files:
    #             output_data, output_meta, output_emb = self.process_one_file(file_name)
    #             for d, m, e in zip(output_data, output_meta, output_emb):
    #                 index.add(e, d, m)

    #         # Save the index under a folder named after the document
    #         save_path = os.path.join(self.save_dir, folder_name)
    #         os.makedirs(save_path, exist_ok=True)
    #         index.save(os.path.join(save_path, 'docstore'))
    #         print(f"[Done] Processed and saved index for '{folder_name}'.")

    # def build_index_all_files(self):
    #     exts = ('.pdf', '.jpg', '.jpeg', '.png')
    #     files = sorted([
    #         f for f in os.listdir(self.data_dir)
    #         if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.data_dir, f))
    #     ])

    #     emb_dim = self.embedder.get_dims()
    #     index = VectorStore(emb_dim)

    #     if not os.path.exists(self.save_dir):
    #         os.makedirs(self.save_dir)

    #     save_path = os.path.join(self.save_dir, 'docstore')

    #     for f in tqdm(files):
    #         print(f"Processing File {f}")
    #         output_data, output_meta, output_emb = self.process_one_file(f)
    #         for d, m, e in zip(output_data, output_meta, output_emb):
    #             index.add(e, d, m)

    #         index.save(save_path)

