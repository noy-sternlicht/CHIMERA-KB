#!/bin/bash -x
#SBATCH --time=30:00:00
#SBATCH --gres gg:g4
#SBATCH --mem-per-cpu=10g

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


#-------------------------------Mistral CoT Abstract Classifier-----------------------------------

python3 src/experiments/recombination_extraction/eval_mistral_classifier.py \
  --eval_path data/recombination_extraction_data/mistral_data/cot_abstract_classification/eval.jsonl \
  --tokenizer_path models/extraction_models/checkpoints/mistral_abstract_cot_classifier/tokenizer.model.v3 \
  --model_path mistral_base_model \
  --lora_path models/extraction_models/checkpoints/mistral_abstract_cot_classifier/lora.safetensors \
  --output_dir mistral_7B_cot_classifier/eval_out \
  --is-cot

#-------------------------------Mistral Abstract Classifier-----------------------------------

python3 src/experiments/recombination_extraction/eval_mistral_classifier.py \
  --eval_path data/recombination_extraction_data/mistral_data/abstract_classification/eval.jsonl \
  --tokenizer_path models/extraction_models/checkpoints/mistral_abstract_classifier/tokenizer.model.v3 \
  --model_path mistral_base_model \
  --lora_path models/extraction_models/checkpoints/mistral_abstract_classifier/lora.safetensors \
  --output_dir mistral_7B_classifier/eval_out


#-------------------------------Mistral E2E Extractor-----------------------------------

python3 src/experiments/recombination_extraction/eval_mistral_e2e.py \
  --eval_path data/recombination_extraction_data/eval.csv \
  --trained_model_path models/extraction_models/checkpoints/mistral_e2e \
  --base_model_path mistral_base_model \
  --output_dir mistral_7B_e2e/eval_out