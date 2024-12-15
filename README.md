# LLMiniST

This repository contains a collection of tools and scripts for identifying niches in spatial transcriptomics data, comparing zero-shot and fine-tuned approaches.The project supports multiple spatial transcriptomics platforms including STARmap, MERFISH, and Visium.

## Overview

（see you soon）

## Project Structure

```
.
├── zeroshot_on_starmap.py    # Zero-shot analysis for STARmap data
├── zeroshot_on_merfish.py    # Zero-shot analysis for MERFISH data
├── zeroshot_on_visium.py     # Zero-shot analysis for Visium data
├── finetune_on_starmap.py    # Fine-tuning analysis for STARmap data
├── finetune_on_merfish.py    # Fine-tuning analysis for MERFISH data
├── finetune_on_visium.py     # Fine-tuning analysis for Visium data
├── utils.py                  # Utility functions
├── prompt.py                 # Prompt engineering templates
└── submit_end2end.py         # End-to-end submission script
```



## Dependencies

- Python 3.7+
- pandas
- numpy
- scanpy
- scipy
- scikit-learn
- openai
- yaml

### Install with conda in Linux

    ```
    conda create -n stgpt python=3.12
    conda activate stgpt
    conda install -c conda-forge scikit-learn pandas scipy jupyter notebook jupyterlab
    pip install -q -U google-generativeai openai kmodes
    conda install -c conda-forge scanpy python-igraph leidenalg
    pip install tiktoken

    # if use jupyterlab-lsp
    conda install -c conda-forge 'jupyterlab>=4.1.0,<5.0.0a0' jupyterlab-lsp
    conda install -c conda-forge python-lsp-server
    ```

## Configuration

The project uses YAML configuration files located in `model_config/` directory. Key configuration parameters include:

- `data_name`: Name of the dataset
- `r`: Neighborhood radius
- `name_truth`: Ground truth label column
- `tissue_region`: Tissue region information
- `model_type`: Analysis method (zero-shot/fine-tune)
- `minimal_f`: Minimum frequency threshold
- `top_n`: Number of top features to consider

## Usage

### Zero-shot Analysis

```python
# For STARmap data
python zeroshot_on_starmap.py

# For MERFISH data
python zeroshot_on_merfish.py

# For Visium data
python zeroshot_on_visium.py
```

### Fine-tuning Analysis

```python
# For STARmap data
python finetune_on_starmap.py

# For MERFISH data
python finetune_on_merfish.py

# For Visium data
python finetune_on_visium.py
```


## Data Processing Pipeline

1. **Data Loading**: Load spatial transcriptomics data using scanpy
2. **Preprocessing**: 
   - Filter genes
   - Normalize expression data
   - Scale data
3. **Neighbor Analysis**:
   - Compute adjacency matrices
   - Calculate cell type frequencies
   - Analyze gene expression patterns
4. **Model Application**:
   - Generate prompts
   - Apply LLM analysis
   - Process results

## Output

The analysis generates several types of outputs:

- JSON files with model predictions
- Batch processing results
- Cell type annotations
- Microenvironment classifications

Results are stored in:
- `batch_json/`: Raw model outputs
- `batch_results/`: Processed results
- `finetune_json/`: Fine-tuning data

