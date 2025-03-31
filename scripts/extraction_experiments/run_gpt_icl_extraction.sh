#!/bin/bash -x
#SBATCH --time=30:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=10g

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

#-------------------------------ICL Entity GPT-----------------------------------

python3 src/experiments/recombination_extraction/icl_entity_experiment.py \
  --icl_examples_path data/recombination_extraction_data/train.csv \
  --eval_path data/recombination_extraction_data/eval.csv \
  --nr_samples_per_class 45 \
  --nr_repeats 5 \
  --output_dir './icl_ner_gpt_out' \
  --openai_engine 'gpt-4o'

#-------------------------------ICL E2E GPT-----------------------------------

python3 src/experiments/recombination_extraction/icl_e2e_experiment.py \
  --icl_examples_path data/recombination_extraction_data/train.csv \
  --eval_path data/recombination_extraction_data/eval.csv \
  --nr_samples_per_class 45 \
  --nr_repeats 5 \
  --output_dir './icl_e2e_gpt_out' \
  --openai_engine 'gpt-4o'
