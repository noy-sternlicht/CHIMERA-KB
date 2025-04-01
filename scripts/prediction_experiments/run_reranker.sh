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


python3 src/link_prediction/reranker.py \
  --biencoder_results "" \
  --output_dir "reranker_out" \
  --openai_engine "gpt-4o" \
  --rank_gpt_window_size 10 \
  --rank_gpt_step_size 5 \
  --top_k 20 \
  --perform_checkpoint_at 500