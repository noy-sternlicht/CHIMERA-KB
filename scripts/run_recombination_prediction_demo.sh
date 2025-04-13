#!/bin/bash -x
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --mem-per-cpu=20g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  PYTHONPATH=$PYTHONPATH:$PWD/RankGPT
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

python3 src/demo/recombination_prediction.py \
  --input_path "src/demo/prediction_demo_example.json" \
  --entities_path "data/CHIMERA/entities_text.csv" \
  --output_path "sentence_transformers_link_prediction_res" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt" \
  --weights_precision 32 \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --checkpoint 'models/pred_models/all-mpnet-base'
