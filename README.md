
[![DOI](https://img.shields.io/badge/DOI-10.XXXX/XXXXX-blue.svg)](https://doi.org/10.XXXX/XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

[//]: # (<h1 align="center">CHIMERA: A Knowledge Base of Idea Recombination in Scientific Literature</h1>)

<p align="center">
  <img src="kb_logo.svg" alt="Centered Image" width="450" />
</p>

## The CHIMERA knowledge base
CHIMERA is knowledge base of over 28K _real_ scientific recombination examples. 
Recombination is the process of creating original ideas by integrating elements of existing mechanisms and concepts. For example, taking inspiration from nature to design new technologies:

<p align="center">
      <img src="recombination_example.svg" alt="Description" width="550" />
</p>

We build CHIMERA by automatically extracting examples of "recombination in action" from the scientific literature. You are welcome to use CHIMERA to study recombination in science, develop new algorithms, or for any other purpose! 
Make sure to cite our paper as described [here](#Citation). 

### Data
**TODO**: update
The `data/` contains three zip files with the following contents:
```aiignore
├── CHIMERA.zip                             # The CHIMERA knowledge base
├── recombination_extraction_data.zip       # train, test sets for recombination extraction
├── recombination_prediction_data.zip       # train, dev, test sets for recombination prediction       
```
## Getting Started
**TODO**: Add prerequisites, run pip freeze when I'm done with the repo and add a requirement file here

### Prerequisites
* Python 3.11.2 or higher
* Note that:
  * Some code requires a GPU for training or evaluation.
  * Some code requires an OpenAI API key.
  * Some code requires an HuggingFace API key.

### Installation

```bash
# Clone this repository
git clone https://github.cs.huji.ac.il/tomhope-lab/CHIMERA.git

# Recommended: Create and activate a virtual environment
python3 -m venv myenv
source ./myenv/bin/activate

# Clone external baselines
git clone https://github.com/mistralai/mistral-finetune.git
git clone https://github.com/hitz-zentroa/GoLLIE.git

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch==2.5.1
pip install --no-cache-dir -r requirements.txt
```

### Setting up the OpenAI API
Some experiments require an OpenAI API key. You can set it up by following the instructions [here](https://beta.openai.com/docs/developer-quickstart/).
After you have the API key, create a simple text file `openai_api_key` in the root directory of the project and paste the key there. The code will automatically read the key from this file.

### Setting up the HuggingFace API
Some experiments require an HuggingFace API key. Set it up by creating a similar text file `huggingface_api_key` in the root directory of the project and paste the key there. The code will automatically read the key from this file.

## Reproducing Results
This part describe how to reproduce the results presented in our the paper.

### Recombination extraction
```bash
# Unzip the data
unzip data/recombination_extraction_data.zip -d data/

# Unzip checkpoints
unzip models/extraction_models.zip -d models/

# Now, run the relevant script from scripts/extraction_experiments. For example:
chmod +x ./scripts/extraction_experiments/run_gpt_icl_extraction.sh
./scripts/extraction_experiments/run_gpt_icl_extraction.sh
```
##### PURE Extraction
We use [PURE](https://github.com/princeton-nlp/PURE) as one of our extractive baselines. Reproducing its results requires a few more steps, since the repository code isn't compatible with python>3.7 

### Knowledge base analysis
Run the following to generate the tables and csv files used to create the analysis figures in the paper.
```bash
# Unzip the data
unzip data/CHIMERA.zip -d data/

chmod +x scripts/analyse_kb.sh
./scripts/analyse_kb.sh
````

### Prediction experiments
**TODO**: add an unzip models step

```bash
# Unzip the data
unzip data/recombination_prediction_data.zip -d data/

```
Run the following script to reproduce ranking results:

```bash
#!/bin/bash -x
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
  --entities_path " data/CHIMERA/entities_text.csv" \
  --output_path "sentence_transformers_link_prediction_res" \
  --nr_negatives 30 \
  --all_edges_path "data/recombination_prediction_data/all.csv" \
  --test_candidates_path "data/recombination_prediction_data/entities_after_cutoff.txt" \
  --valid_candidates_path "data/recombination_prediction_data/entities_before_cutoff.txt" \
  --model_name "BAAI/bge-large-en-v1.5" \ # Either "BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2" or "sentence-transformers/all-mpnet-base-v2"
  --num_train_epochs 3 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --encode_batch_size 1024 \
  --weights_precision 32 \
  --checkpoint '' \  # path to a checkpoint to load, remove if training from scratch
  --zero_shot          # remove if training from scratch
```
**TODO**: mention rankgpt prompt changes
## Citation
If you use this code or data in your research, please cite our paper:

```bibtex
@article{author2025paper,
  title={Paper Title},
  author={Last, First and Coauthor, Another},
  journal={Journal Name},
  volume={X},
  number={Y},
  pages={ZZ--ZZ},
  year={2025},
  publisher={Publisher}
}
```

## Authors

- **First Last** - [GitHub Profile](https://github.com/username)
- **Another Coauthor** - [GitHub Profile](https://github.com/coauthor)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
