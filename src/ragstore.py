import os
from tqdm import tqdm

from src.layout import LayoutProcessor
from src.embedder import BaseEmbedder
from src.vector_store import VectorStore
from src.llm import BaseVLMEngine, BaseLLMEngine
from src.prompts import VLMPrompts
from tqdm import tqdm
import pymupdf
import re
import heapq
from PIL import Image
# import pytesseract

# MAX_EMBED_TOKENS = 32000 # < 40960 for Qwen3-Embedding-8B
MAX_PROMPT_TOKENS = 32000 # it sometimes outputs ..... etc.

class Ragstore:
    def __init__(self,
                lp: LayoutProcessor,
                embedder: BaseEmbedder,
                vlm: BaseVLMEngine, 
                # llm: BaseLLMEngine,
                vlm_prompts: VLMPrompts,
                # llm_prompts: LLMPrompts,
                icl,
                model,
                data_dir,
                save_dir):
        self.lp = lp
        self.embedder = embedder
        self.vlm = vlm
        # self.llm = llm
        self.vlm_prompts = vlm_prompts
        # self.llm_prompts = llm_prompts
        self.icl = icl
        self.model = model
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.num_examples = 3
        self.fallback_ct = 0

    def process_one_file_pymupdf(self, file_path, file_name):
        doc = pymupdf.open(file_path)
        output_data, output_meta, output_emb = [], [], []

        # Extract logical page number from filename
        match = re.search(r'_p(\d+)', file_name)
        logical_page = int(match.group(1)) if match else 0

        for page_idx, page in enumerate(doc):
            page_text = page.get_text("text")

            page_meta = {
                "file": file_name,
                "page": logical_page,
                "mode": "page_pymupdf"
            }

            output_data.append(page_text)
            output_meta.append(page_meta)
            output_emb.append(self.embedder.encode([page_text]))

        return output_data, output_meta, output_emb
    
    def process_one_file_pytesseract(self, file_path, file_name):
        output_data, output_meta, output_emb = [], [], []

        match = re.search(r'_p(\d+)', file_name)
        logical_page = int(match.group(1)) if match else 0

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        page_meta = {
            "file": file_name,
            "page": logical_page,  
            "mode": "page_pytesseract"
        }

        output_data.append(text)
        output_meta.append(page_meta)
        output_emb.append(self.embedder.encode([text]))

        return output_data, output_meta, output_emb
    
    def process_one_file_vlm(self, file_path, file_name):
        output_data, output_meta, output_emb = [], [], []

        match = re.search(r'_p(\d+)', file_name)
        logical_page = int(match.group(1)) if match else 0

        vlm_prompt = """/no_think
        Please parse everything in the attached image and output the parsed contents only without anything else.
        """

        text = self.vlm.generate(vlm_prompt, file_path)

        page_meta = {
            "file": file_name,
            "page": logical_page,  
            "mode": "page_vlm",
        }

        output_data.append(text)
        output_meta.append(page_meta)
        output_emb.append(self.embedder.encode([text]))

        return output_data, output_meta, output_emb

    def process_one_file_xLayout(self, file_path, file_name):
        output_data, output_meta, output_emb = [], [], []

        match = re.search(r'_p(\d+)', file_name)
        logical_page = int(match.group(1)) if match else 0

        title_prompt = self.vlm_prompts.vlm_title_prompt
        figure_prompt = self.vlm_prompts.vlm_figure_prompt
        text_prompt = self.vlm_prompts.vlm_text_prompt
        table_prompt = self.vlm_prompts.build_vlm_table_prompt(self.icl)

        vlm_prompt = "\n".join([
            "Documents can contain title, figure, text, and table components. Please parse the components according to the following instructions:\n",
            "Title Component:",
            title_prompt,
            "Figure Component:",
            figure_prompt,
            "Text Component:",
            text_prompt,
            "Table Component:",
            table_prompt
        ])

        text = self.vlm.generate(vlm_prompt, file_path)

        # Add page-level overview using VLM
        vlm_page_prompt = self.vlm_prompts.prompt_map[5]
        page_overview = self.vlm.generate(vlm_page_prompt, file_path)
        page_text = page_overview + "\n" + "\n".join(text)

        # Page-level metadata with sub-components
        page_meta = {
            "file": file_name,
            "page": logical_page,
            "mode": "page_xLayout",
            "overview": page_overview,
        }

        # Store
        output_data.append(page_text)
        output_meta.append(page_meta)

        MAX_CHARS = 40960 # output too long for embedding
        if len(page_text) > MAX_CHARS:
            print(f"Page text length {len(page_text)} exceeds {MAX_CHARS}, splitting into chunks.")

        chunk_embs = []
        for i in range(0, len(page_text), MAX_CHARS):
            chunk = page_text[i : i + MAX_CHARS]
            emb = self.embedder.encode([chunk])[0]
            chunk_embs.append(emb)

        page_emb = sum(chunk_embs) / len(chunk_embs) # mean pooling embeddings
        output_emb.append(page_emb[None, :])

        return output_data, output_meta, output_emb
    
    def process_one_file(self, file_path, file_name):
        # file_path = os.path.join(self.data_dir, file_name)
        comps = self.lp.preprocess(file_path)

        match = re.search(r'_p(\d+)', file_name)
        logical_page = int(match.group(1)) if match else 0

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
                # if component_class != 3 or len(self.icl_examples) == 0:
                if component_class != 3:
                    vlm_prompt = self.vlm_prompts.prompt_map[component_class]
                else:
                    vlm_prompt = self.vlm_prompts.build_vlm_table_prompt(self.icl)
                    # vlm_prompt = self.vlm_prompts.vlm_table_prompt_xStructureICL
                    # vlm_prompt = self.vlm_prompts.vlm_table_prompt_xICL

                # Fallback for exceedingly long prompts
                if len(vlm_prompt) > MAX_PROMPT_TOKENS:
                    vlm_prompt = self.vlm_prompts.prompt_map[component_class]
                    print(f"Prompt too long, using default prompt. ({self.fallback_ct})")
                    self.fallback_ct += 1

                # Extract text with VLM
                output = self.vlm.generate(vlm_prompt, vlm_image)

                # Collect component-level info
                page_texts.append(output)
                page_components.append({
                    "text": output,
                    "bbox": component['bbox'],
                    "mode": "component"
                })

            # Add page-level overview using VLM
            vlm_page_prompt = self.vlm_prompts.prompt_map[5]
            page_overview = self.vlm.generate(vlm_page_prompt, file_path)
            # page_overview = ""
            page_text = page_overview + "\n" + "\n".join(page_texts)

            # Page-level metadata with sub-components
            page_meta = {
                "file": file_name,
                "page": logical_page,
                "mode": "page",
                "overview": page_overview,
                "components": page_components
            }

            # Store
            output_data.append(page_text)
            output_meta.append(page_meta)

            MAX_CHARS = 40960 # output too long for embedding
            if len(page_text) > MAX_CHARS:
                print(f"Page text length {len(page_text)} exceeds {MAX_CHARS}, splitting into chunks.")

            chunk_embs = []
            for i in range(0, len(page_text), MAX_CHARS):
                chunk = page_text[i : i + MAX_CHARS]
                emb = self.embedder.encode([chunk])[0]
                chunk_embs.append(emb)

            page_emb = sum(chunk_embs) / len(chunk_embs) # mean pooling embeddings
            output_emb.append(page_emb[None, :])
        
        return output_data, output_meta, output_emb
    
    def build_index_per_folder(self):
        # Collect all image files in the folder
        files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')) and os.path.isfile(os.path.join(self.data_dir, f))
        ])

        # Create a single VectorStore for the entire folder
        emb_dim = self.embedder.get_dims()
        index = VectorStore(emb_dim)

        for f in tqdm(files, desc=f"Building Folder {os.path.basename(self.data_dir)}"):
            file_path = os.path.join(self.data_dir, f)

            if self.model == "tabrag":
                output_data, output_meta, output_emb = self.process_one_file(file_path=file_path, file_name=f)
                # output_data, output_meta, output_emb = self.process_one_file_xLayout(file_path=file_path, file_name=f)
            elif self.model == "pymupdf":
                output_data, output_meta, output_emb = self.process_one_file_pymupdf(file_path=file_path, file_name=f)
            elif self.model == "pytesseract":
                output_data, output_meta, output_emb = self.process_one_file_pytesseract(file_path=file_path, file_name=f)
            elif self.model == "vlm":
                output_data, output_meta, output_emb = self.process_one_file_vlm(file_path=file_path, file_name=f)
            
            for d, m, e in zip(output_data, output_meta, output_emb):
                index.add(e, d, m)

        # Save once per folder
        os.makedirs(self.save_dir, exist_ok=True)
        index.save(os.path.join(self.save_dir, "docstore"))
