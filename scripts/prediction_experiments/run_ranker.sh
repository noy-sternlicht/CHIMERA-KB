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

python3 src/experiments/recombination_prediction/finetune_sent_transformer_biencoder.py \
  --train_path "data/recombination_prediction_data/train.csv" \
  --test_path "data/recombination_prediction_data/test.csv" \
  --valid_path "data/recombination_prediction_data/valid.csv" \
  --entities_path "data/CHIMERA/entities_text.csv" \
  --output_path "sentence_transformers_link_prediction_res" \
  --nr_negatives 30 \
  --all_edges_path "data/recombination_prediction_data/all.csv" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt" \
  --valid_candidates_path "data/recombination_prediction_data/entities_before_cutoff.txt" \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --num_train_epochs 3 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --encode_batch_size 1024 \
  --eval_candidates_cutoff_year 2024 \
  --weights_precision 32 \
  --checkpoint 'models/pred_models/all-mpnet-base' \
  --zero_shot
