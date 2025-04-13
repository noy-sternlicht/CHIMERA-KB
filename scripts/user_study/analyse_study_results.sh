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

python3 src/experiments/recombination_prediction/user_study/results_analysis.py \
  --responses_path "data/user_study_results.csv" \
  --output_path "study_results"