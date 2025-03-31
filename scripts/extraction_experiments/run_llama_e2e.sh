#!/bin/bash -x
#SBATCH --time=30:00:00
#SBATCH --gres gg:g4
#SBATCH --mem-per-cpu=20g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_HOME=$PWD/.hf_home
  export HF_HOME
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
  --checkpoint "models/extraction_models/checkpoints/llama-8b"