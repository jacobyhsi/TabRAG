# Setup Instructions

### Setting up environments
```
conda create --name searag python=3.10
conda activate searag
```
Installing Dependencies
```
pip install torch
pip install torchvision
pip install transformers
pip install accelerate
pip install openai
pip install faiss-gpu
pip install opencv-python
pip install pymupdf
pip install timm
pip install shapely
pip install qwen_vl_utils
uv pip install scipy
uv pip install sentence-transformers
pip install numpy==1.26.4
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

```
uv venv --python 3.10
source .venv/bin/activate
```
Installing Dependencies
```
uv pip install torch
uv pip install torchvision
uv pip install transformers
uv pip install accelerate
uv pip install openai
uv pip install faiss-gpu
uv pip install opencv-python
uv pip install pymupdf
uv pip install timm
uv pip install shapely
uv pip install qwen_vl_utils
uv pip install scipy
uv pip install sentence-transformers
uv pip install numpy==1.26.4
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
uv pip install sentencepiece
uv pip install PyPDF2
uv pip install gdown
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

TatDQA:

Download the TAT-DQA Dataset from Google Drive

Make a tatdqa/ folder and download the following:

Dataset: gdown https://drive.google.com/uc?id=1iqe5r-qgQZLhGtM4G6LkNp9S6OCwOF2L (unzip this after downloading)

QA Answer Pairs: gdown https://drive.google.com/uc?id=1ZQjjIC0BB14l6t9b1Ryq0t-CNAP6iC2J

Make sure Dataset and Answer Pairs are in datasets/tatdqa/test and datasets/tatdqa/

FintabNet:
```
wget https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/fintabnet.tar.gz
tar -xvf fintabnet.tar.gz
```

MP-DocVQA:
```
wget https://datasets.cvc.uab.es/rrc/DocVQA/Task4/images.tar.gz --no-check-certificate
tar -xvf fintabnet.tar.gz
```

### Run
python make_ragstore.py