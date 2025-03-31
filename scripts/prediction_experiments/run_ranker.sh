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

python3 src/link_prediction/finetune_sent_transformer_biencoder.py \
  --train_path "cross_domain_link_prediction_06_03_all_with_other/train.csv" \
  --test_path "cross_domain_link_prediction_06_03_all_with_other/test.csv" \
  --valid_path "cross_domain_link_prediction_06_03_all_with_other/valid.csv" \
  --entities_path "kb_output_31_01_with_other_nodes/entities_text.csv" \
  --output_path "sentence_transformers_link_prediction_res" \
  --nr_negatives 30 \
  --all_edges_path "cross_domain_link_prediction_06_03_all_with_other/all.csv" \
  --test_candidates_path "cross_domain_link_prediction_06_03_all_with_other/entities_after_cutoff.txt" \
  --valid_candidates_path "cross_domain_link_prediction_06_03_all_with_other/entities_before_cutoff.txt" \
  --model_name "BAAI/bge-large-en-v1.5" \
  --num_train_epochs 3 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --encode_batch_size 1024 \
  --eval_candidates_cutoff_year 2024 \
  --weights_precision 32 \
  --checkpoint '' \
  --zero_shot \
