#!/bin/bash -x
#SBATCH --time=01:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=30g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate # Replace with virtual env activate script path
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH
}

activate
set_env_vars

python3 src/experiments/analyse_kb.py \
  --data-path "data/CHIMERA" \
  --output-path "cross_domain_sunkey_analysis" \
  --min-year 2019 \
  --max-year 2025 \
  --pair_count_percentile 0.9 \
  --entity_count_percentile 0.5