#!/bin/bash -x
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem-per-cpu=50g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH
}

activate
set_env_vars

python3 src/experiments/recombination_prediction/user_study/prep_user_study_data.py \
  --output_path "compare_baselines_out" \
  --baselines_results_path "user_study/baselines.json" \
  --test_path "data/recombination_prediction_data/test.csv" \
  --baselines_to_compare random mpnet_zero sciIE gpt-4o ours \
  --top_k 1 \
  --inspiration_examples_only
