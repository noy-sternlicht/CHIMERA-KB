
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
* Some code requires a GPU for training or evaluation.
* Some code requires an OpenAI API key.

### Installation

```bash
# Clone this repository
git clone https://github.cs.huji.ac.il/tomhope-lab/CHIMERA.git

# Recommended: Create and activate a virtual environment
python3 -m venv myenv
source ./myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setting up the OpenAI API
Some experiments require an OpenAI API key. You can set it up by following the instructions [here](https://beta.openai.com/docs/developer-quickstart/).
After you have the API key, create a simple text file `openai_api_key` in the root directory of the project and paste the key there. The code will automatically read the key from this file.

### Reproducing Results
This part describe how to reproduce the results presented in our the paper.

#### Recombination Extraction
```bash
# Unzip the data
unzip data/recombination_extraction_data.zip -d data/

# Unzip checkpoints
unzip models/extraction_models.zip -d models/

# Run all extraction baselines. You can comment out the ones you don't want to run (for example, ones that require an OpenAI API key)
chmod +x ./scripts/01_run_extraction_experiments.sh
./scripts/01_run_extraction_experiments.sh

```



##  Citation

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
