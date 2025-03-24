
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
`data/` contains contains a zip file with two directories:
```aiignore
├── human_annotated_data/  # Training and evaluation data of our extraction model  
├── CHIMERA/               
│   ├── raw_graph/         # The CHIMERA knowledge base
│   └── train.csv/         # Recombination prediction training data
|   └── valid.csv/         # Recombination prediction validation data
|   └── test.csv/          # Recombination prediction test data (papers > 2024)
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

# Install dependencies
pip install -r requirements.txt

# Unzip the data
unzip data/chimera_data.zip -d data
```

### Reproducing Results
```bash
# Navigate to the code directory
cd code

# Run the analysis
python main.py
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
