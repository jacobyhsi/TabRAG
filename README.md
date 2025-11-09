# TabRAG: Tabular Document Retrieval via Structured Language Representations
<div align="center">
<div>
    <a href="https://jacobyhsi.github.io/" target="_blank">Jacob Si</a><sup>*</sup> | 
    <a href="https://mikequ1.github.io/" target="_blank">Mike Qu</a><sup>*</sup> | 
    <a href="https://www.linkedin.com/in/michelle-lee-9a796712b/" target="_blank">Michelle Lee</a> | 
    <a href="http://yingzhenli.net/home/en/" target="_blank">Yingzhen Li</a>
</div>
<br>
<div>
    Imperial College London
</div>
<br>
</div>

<p align="center">
<a href=""><img src="https://img.shields.io/badge/arXiv-2505.18495-b31b1b.svg?logo=arxiv&logoColor=red" alt="TabRAG on arXiv"/></a>
<a href="https://github.com/jacobyhsi/VUD/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

## Installation

The following delineates the installation instructions. Clone this repository and navigate to it in your terminal. Create an environment using a preferred package manager.

Note: can replace with `uv`.

```
conda create --name tabrag python=3.10
conda activate tabrag
```
Installing Dependencies
```
pip install torch
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
pip install pymupdf
pip install transformers
pip install openai
pip install faiss-gpu
pip install timm
pip install shapely
pip install qwen_vl_utils
pip install scipy
pip install sentence-transformers
pip install gdown
pip install opencv-python
pip install numpy==1.26.4
pip install pypdf
pip install vllm
pip install arxiv
```

### Layout model checkpoint
Microsoft's DIT model (Document Image Transformer) is used for layout extraction: https://github.com/microsoft/unilm/tree/master/dit

Download this checkpoint: 
https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ESKnk2I_O09Em52V1xb2Ux0BrO_Z-7cuzL3H1KQRxipb7Q?e=iqTfGc

Move it to the project directory

### Datasets
Create a datasets/ folder

```
mkdir datasets
cd datasets
```

**TatDQA**:

Download the TAT-DQA Dataset from Google Drive

Make a tatdqa/ folder and download the following:

Dataset: gdown https://drive.google.com/uc?id=1iqe5r-qgQZLhGtM4G6LkNp9S6OCwOF2L (unzip this after downloading)

QA Answer Pairs: gdown https://drive.google.com/uc?id=1ZQjjIC0BB14l6t9b1Ryq0t-CNAP6iC2J

Make sure Dataset and Answer Pairs are in datasets/tatdqa/test and datasets/tatdqa/

**MP-DocVQA**:
```
wget https://datasets.cvc.uab.es/rrc/DocVQA/Task4/images.tar.gz --no-check-certificate
tar -xvf images.tar.gz
python process_mpdocvqa.py # get documents with tables
python filter_mpdocvqa.py # select 500 pages based on qa:pages ratio
python indent_mpdocvqa.py # visibility of val.json
```

**SPIQA**:
```
# mkdir/cd into datasets/SPIQA
pip install arxiv

# open python shell: python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/spiqa", repo_type="dataset", local_dir='.') ### Mention the local directory path
```

**FinTabNet**:
```
wget https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/fintabnet.tar.gz
tar -xvf fintabnet.tar.gz
```

### Run
```
python make_ragstore.py
```
