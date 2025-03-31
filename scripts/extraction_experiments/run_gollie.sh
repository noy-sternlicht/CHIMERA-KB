#!/bin/bash -x
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem-per-cpu=10g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/GoLLIE
  export PYTHONPATH

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars

module load cuda
module load nvidia

python3 src/experiments/recombination_extraction/run_gollie.py \
  --output_dir gollie_out \
  --eval_path data/recombination_extraction_data/eval.csv \
  --model_name HiTZ/GoLLIE-13B

python3 src/automatic_annotation/test_gollie.py \
  --output_dir gollie_eval_out \
  --eval_path data/recombination_extraction_data/eval.csv \
  --predictions_path ./gollie_out/predictions.json
