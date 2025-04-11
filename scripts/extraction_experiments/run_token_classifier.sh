#!/bin/bash -x
#SBATCH --time=10:00:00
#SBATCH --gres gg:g4
#SBATCH --mem-per-cpu=30g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate # Replace with virtual env activate script path
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME

  TOKENIZERS_PARALLELISM=false
  export TOKENIZERS_PARALLELISM
}

activate
set_env_vars

module load cuda
module load nvidia

python3 src/experiments/recombination_extraction/general_token_classifier.py \
--output_dir token_classifier_out \
--data_dir 'data/recombination_extraction_data' \
--model_name 'allenai/scibert_scivocab_uncased' \
--checkpoint 'models/extraction_models/checkpoints/token_classifier'
