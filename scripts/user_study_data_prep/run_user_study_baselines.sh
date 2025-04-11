#!/bin/bash -x
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem-per-cpu=10g
#SBATCH --mail-user=noy.sternlicht@mail.huji.ac.il
#SBATCH --mail-type=ALL
#SBATCH --job-name=finetune_doc_classifier

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
}

activate
set_env_vars

module load cuda
module load nvidia



#-------------------------------ZS-CHIMERA-----------------------------------
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
  --checkpoint '' \
  --zero_shot


#-------------------------------ZS-SCIERC-----------------------------------
python3 src/experiments/recombination_prediction/user_study/scierc/extract_sciie_entities.py \
  --test_path "data/recombination_prediction_data/test.csv" \
  --output_path "sciERC_baseline_res"


python3 src/experiments/recombination_prediction/user_study/scierc/refine_sciie_entities.py\
  --data_path "sciERC_baseline_res/entities_by_rel_type.json" \
  --output_path "sciERC_baseline_res" \
  --encoder 'sentence-transformers/all-mpnet-base-v2' \
  --clustering_thershold 0.05 \
  --selected_relations 'Conjunction' 'Hyponym-of'

python3 src/experiments/recombination_prediction/user_study/scierc/sciie_baseline.py \
  --test_path "data/recombination_prediction_data/test.csv" \
  --entities_path "data/CHIMERA/entities_text.csv" \
  --sciie_entities "sciERC_baseline_res/reduced_entities.csv" \
  --output_path "sciERC_baseline_out" \
  --all_edges_path "data/recombination_prediction_data/all.csv" \
  --model_name "sentence-transformers/all-mpnet-base-v2" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt"


#-------------------------------GPT-4o-----------------------------------
python3 src/experiments/recombination_prediction/user_study/llm_baseline.py \
  --test_path "data/recombination_prediction_data/test.csv" \
  --entities_path "data/CHIMERA/entities_text.csv" \
  --output_path "llm_baseline_out" \
  --all_edges_path "data/recombination_prediction_data/all.csv" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt" \
  --openai_engine "gpt-4o"


#-------------------------------Random-----------------------------------
python3 src/link_prediction/random_baseline.py \
  --test_path "data/recombination_prediction_data/test.csv" \
  --entities_path "data/CHIMERA/entities_text.csv" \
  --output_path "random_baseline" \
  --all_edges_path "data/recombination_prediction_data/all.csv" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt"