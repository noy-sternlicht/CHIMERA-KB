#!/bin/bash -x
#SBATCH --time=100:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=50g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  PYTHONPATH=$PYTHONPATH:$PWD/RankGPT
  export PYTHONPATH
}

activate
set_env_vars


python3 src/experiments/recombination_prediction/reranker.py \
  --biencoder_results "sentence_transformers_link_prediction_res/bge-large-en-v1.5_zero_shot_2025-04-11_12-02-54/results.json" \
  --output_dir "reranker_out" \
  --openai_engine "gpt-4o" \
  --rank_gpt_window_size 10 \
  --rank_gpt_step_size 5 \
  --top_k 20 \
  --perform_checkpoint_at 500