#!/bin/bash -x
#SBATCH --time=30:00:00
#SBATCH --gres gg:g4
#SBATCH --mem-per-cpu=20g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars

module load cuda
module load nvidia

python3 src/demo/recombination_extraction.py \
  --input_path src/demo/extraction_input_example.txt \
  --output_dir extraction_demo_out \
  --base_model_path mistral_base_model \
  --extraction_model_path models/extraction_models/checkpoints/mistral_e2e
