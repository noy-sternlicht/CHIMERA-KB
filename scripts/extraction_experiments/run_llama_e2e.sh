#!/bin/bash -x
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1,vmem:48G
#SBATCH --mem-per-cpu=20g
#SBATCH --mail-user=noy.sternlicht@mail.huji.ac.il
#SBATCH --mail-type=ALL
#SBATCH --job-name=hf-finetune

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME

  LLAMA_MODEL_DIR=$PWD/.llama_models
  export LLAMA_MODEL_DIR
}

activate
set_env_vars

module load cuda
module load nvidia

python3 src/experiments/recombination_extraction/llama_e2e.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --train_path "./data/recombination_extraction_data/train.csv" \
  --eval_path "./data/recombination_extraction_data/eval.csv" \
  --output_dir "simple_hf_finetune" \
  --temperature 0.0 \
  --max_seq_length 4096 \
  --nr_steps 500 \
  --batch_size 1 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --hf_key_path huggingface_api_key \
  --checkpoint "simple_hf_finetune/meta-llama/Llama-3.1-8B-Instruct-2025-03-27-14-04/best_checkpoint_meta-llama/Llama-3.1-8B-Instruct-2025-03-27-14-04"