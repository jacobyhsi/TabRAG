# Install UV
## Download UV to appropriate directory
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/voyager/projects/jacobyhsi/uv" sh

## Ensure you set UV cache directory in bashrc
vim ~/.bashrc
export UV_CACHE_DIR="/voyager/projects/jacobyhsi/.cache/uv"

# Create UV environment within your working repo
uv venv venv --python 3.10 --seed
source venv/bin/activate

# Install UV packages
uv pip install vllm
uv pip install pandas
uv pip install numpy
uv pip install deprecated
uv pip install filetype
uv pip install dataclasses_json
uv pip install nltk
uv pip install sqlalchemy
uv pip install tenacity
uv pip install pymupdf
uv pip install qwen_vl_utils
uv pip install sentencepiece
uv pip install flashinfer
uv pip install PyPDF2
uv pip install gdown

# # Set CUDA if needed before installing flash-attn
# echo 'export CUDA_HOME=/pkgs/cuda-12.4' >> ~/.bashrc
# echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# uv pip install flash-attn --no-build-isolation
