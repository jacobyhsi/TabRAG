retrain:
- tabrag, tablevqa, 8b

source /home/js2723/PROJECTS/TabRAG/.venv/bin/activate

python eval/eval_generation.py \
  --dataset tablevqa \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tablevqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tablevqa \
  --method vlm \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset wikitq \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset wikitq \
  --method vlm \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset wikitq \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset wikitq \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset mpdocvqa \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset mpdocvqa \
  --method vlm \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset mpdocvqa \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset comtqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc


python eval/eval_generation.py \
  --dataset comtqa \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc
  
python eval/eval_generation.py \
  --dataset comtqa \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset finqa \
  --method vlm \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset finqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset finqa \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset comtqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method gpt-5.2 \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset finqa \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset finqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python retrieval_to_gen.py

python eval/eval_generation.py \
  --dataset mpdocvqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset wikitq \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset comtqa \
  --method vlm \
  --vlm_model gpt-5.2 \
  --metric acc

python eval/eval_generation.py \
  --dataset mpdocvqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tablevqa \
  --method vlm \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tatdqa \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

# ===== mpdocvqa =====
python eval/eval_full.py --dataset mpdocvqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset mpdocvqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset mpdocvqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset mpdocvqa --method vlm --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset mpdocvqa --method vlm --vlm_model gpt-5.2

# ===== tatdqa =====
python eval/eval_full.py --dataset tatdqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset tatdqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset tatdqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset tatdqa --method vlm --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset tatdqa --method vlm --vlm_model gpt-5.2

# ===== comtqa =====
python eval/eval_full.py --dataset comtqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset comtqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset comtqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset comtqa --method vlm --vlm_model Qwen3-VL-8B-Instruct

python eval/eval_full.py --dataset comtqa --method vlm --vlm_model gpt-5.2

# ===== tablevqa =====
python eval/eval_full.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset tablevqa --method vlm --vlm_model gpt-5.2

# ===== finqa =====
python eval/eval_full.py --dataset finqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset finqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset finqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset finqa --method vlm --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset finqa --method vlm --vlm_model gpt-5.2

# ===== wikitq =====
python eval/eval_full.py --dataset wikitq --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset wikitq --method tabrag --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset wikitq --method vlm --vlm_model Qwen3-VL-32B-Instruct
python eval/eval_full.py --dataset wikitq --method vlm --vlm_model Qwen3-VL-8B-Instruct
python eval/eval_full.py --dataset wikitq --method vlm --vlm_model gpt-5.2

# ComTQA
python retrieval_to_gen.py \
  --retrieval_dir storages/comtqa/retrieval/tabrag/Qwen3-VL-32B-Instruct \
  --generation_dir storages/comtqa/generation/tabrag/Qwen3-VL-32B-Instruct_2 \
  --flatten_groups

# TableVQA
python retrieval_to_gen.py \
  --retrieval_dir storages/tablevqa/retrieval/tabrag/Qwen3-VL-32B-Instruct \
  --generation_dir storages/tablevqa/generation/tabrag/Qwen3-VL-32B-Instruct_2 \
  --flatten_groups

python eval/eval_generation.py \
  --dataset comtqa2 \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python eval/eval_generation.py \
  --dataset tablevqa2 \
  --method tabrag \
  --vlm_model Qwen3-VL-32B-Instruct \
  --metric acc

python retrieval_to_gen.py \
  --retrieval_dir storages/tablevqa/retrieval/vlm/Qwen3-VL-8B-Instruct \
  --generation_dir storages/tablevqa/generation/vlm/Qwen3-VL-8B-Instruct \
  --flatten_groups

python eval/eval_generation.py \
  --dataset tablevqa \
  --method vlm \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python retrieval_to_gen.py \
  --retrieval_dir storages/tablevqa/retrieval/tabrag/Qwen3-VL-8B-Instruct \
  --generation_dir storages/tablevqa/generation/tabrag/Qwen3-VL-8B-Instruct \
  --flatten_groups

python eval/eval_generation.py \
  --dataset tablevqa \
  --method tabrag \
  --vlm_model Qwen3-VL-8B-Instruct \
  --metric acc

python3 eval/eval_generation_image.py --dataset tatdqa
python3 eval/eval_generation_image.py --dataset mpdocvqa
python3 eval/eval_generation_image.py --dataset wikitq
python3 eval/eval_generation_image.py --dataset tablevqa
python3 eval/eval_generation_image.py --dataset comtqa
python3 eval/eval_generation_image.py --dataset finqa

python3 eval/eval_generation_image.py --dataset tatdqa --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset mpdocvqa --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset wikitq --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset comtqa --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset finqa --vlm_model Qwen3.5-9B

python3 eval/eval_generation_image.py --dataset tatdqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset mpdocvqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset wikitq --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset comtqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset finqa --vlm_model Qwen3-VL-8B-Instruct

python3 eval/eval_generation_image.py --dataset tatdqa --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset mpdocvqa --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset wikitq --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset comtqa --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset finqa --vlm_mowhat about del Qwen3-VL-32B-Instruct

#

python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct #done
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-32B-Instruct #done
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model gpt-5.2 #done

python3 eval/eval_generation.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct #done
python3 eval/eval_generation.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct #done
python3 eval/eval_full.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct #done 

python3 eval/eval_full.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_full.py --dataset tablevqa --method vlm --vlm_model gpt-5.2
python3 eval/eval_full.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-8B-Instruct

python3 eval/eval_full.py --dataset tablevqa --method tabrag --vlm_model Qwen3-VL-32B-Instruct
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-32B-Instruct

python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3.5-9B
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3.6-27B
python3 eval/eval_generation_image.py --dataset finqa --vlm_model Qwen3-VL-32B-Instruct

python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-8B-Instruct
python3 eval/eval_generation_image.py --dataset tablevqa --vlm_model Qwen3-VL-32B-Instruct


# 
CUDA_VISIBLE_DEVICES=0 python main.py --model vlm --mode generation --dataset comtqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_comtqa --use_hf
CUDA_VISIBLE_DEVICES=1 python main.py --model vlm --mode generation --dataset finqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_finqa --use_hf
CUDA_VISIBLE_DEVICES=2 python main.py --model vlm --mode generation --dataset tablevqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_tablevqa --use_hf
CUDA_VISIBLE_DEVICES=3 python main.py --model vlm --mode generation --dataset tatdqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_tatdqa --use_hf
CUDA_VISIBLE_DEVICES=4 python main.py --model vlm --mode generation --dataset wikitq --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_wikitq --use_hf
CUDA_VISIBLE_DEVICES=5 python main.py --model vlm --mode generation --dataset mpdocvqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_mpdocvqa --use_hf

# not ready still processing the mpdocvqa dataset
python process_mpdocvqa.py
cd ../..
python main.py --model vlm --mode generation --dataset mpdocvqa --vlm Qwen/Qwen3-VL-8B-Instruct --prompt_type complex_wikitq --use_hf

python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct_complex_comtqa
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct_complex_finqa
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct_complex_tablevqa
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct_complex_tatdqa
python3 eval/eval_generation.py --dataset tablevqa --method vlm --vlm_model Qwen3-VL-8B-Instruct_complex_wikitq

