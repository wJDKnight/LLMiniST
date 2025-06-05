# LLMiniST

This repository contains a collection of tools and scripts for identifying spatial microenvironments (niches) in spatial transcriptomics data using Large Language Models (LLMs). The project supports zero-shot and fine-tuned approaches for multiple spatial transcriptomics platforms including STARmap, MERFISH, and Visium.

## Overview

LLMiniST leverages the power of Large Language Models to analyze spatial transcriptomics data and identify cellular microenvironments. The framework processes neighborhood information around each cell, including cell type compositions and gene expression patterns, to predict spatial niches using natural language prompts.

## Usage

### Quick Start

The easiest way to get started with LLMiniST is through the interactive Jupyter notebook:

1. **Open the notebook**: Launch `zeroshot_notebook.ipynb` in your Jupyter environment
2. **Configure your analysis**: Set up your data paths and model parameters in the configuration
3. **Load and process data**: Run the data loading and preprocessing cells
4. **Submit batch job**: Send your analysis request to the OpenAI batch API
5. **Wait for results**: Since we use OpenAI's batch API, processing may take some time (typically 30 minutes to several hours)
6. **Retrieve and analyze results**: Download the results and run post-processing analysis

### Step-by-Step Workflow

#### 1. Environment Setup
```bash
# Activate your conda environment
conda activate stgpt

# Set up API keys
export OPENAI_API_KEY="your_openai_api_key"
export API_KEY="your_gemini_api_key"  # If using Gemini
```

#### 2. Data Preparation
Ensure your data directory contains the required files:
- `data.csv` (gene expression data)
- `celltype.csv` (cell type annotations)
- `pos.csv` (spatial coordinates)
- `domain.csv` (optional, ground truth labels)

#### 3. Configuration
Edit the YAML configuration file in `model_config/` to match your dataset and analysis requirements.

#### 4. Run Analysis
Open and execute `zeroshot_notebook.ipynb` cell by cell

### Alternative: Gemini API
For faster results (but potentially higher cost), you can use the Gemini API directly through the notebook, which provides immediate responses without batch processing delays.

## Dependencies

- Python 3.7+
- pandas
- numpy
- scanpy
- scipy
- scikit-learn
- openai
- google-generativeai
- yaml
- tiktoken
- kmodes

### Install with conda in Linux

```bash
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

## Data Format

The project expects the following input files in the data directory:

### Required Files
- **`data.csv`**: Main expression data file containing gene expression values (cells × genes)
- **`celltype.csv`**: Cell type annotations for each cell
- **`pos.csv`**: Spatial coordinates (x,y) for each cell

### Optional Files
- **`domain.csv`**: Ground truth domain/niche labels for each cell (for evaluation)

## Configuration

The project uses YAML configuration files located in `model_config/` directory. Here's a detailed breakdown of configuration parameters:

### Data and Input Parameters
- **`data_name`**: String identifier for the dataset (e.g., "BZ5")
- **`r`**: Float value representing the spatial radius threshold for defining cell neighborhoods (e.g., 700)
- **`tissue_region`**: String describing the tissue/region being analyzed (e.g., "mPFC region of mice")
- **`celltype_name`**: Column name in the data containing cell type annotations (e.g., "cell_type")
- **`pos_name`**: List of column names for spatial coordinates (e.g., ["x", "y"])
- **`name_truth`**: Optional column name for ground truth labels (e.g., "niche_truth")

### Model Configuration
- **`model_type`**: String specifying the analysis approach (e.g., "zeroshot_end2end")
- **`gpt_model`**: OpenAI model to use (e.g., "gpt-4o-mini")
- **`openai_url`**: API endpoint for OpenAI requests (default: "/v1/chat/completions")

### Graph and Feature Parameters
- **`Graph_type`**: Type of features to include in analysis:
  - `"count"`: Only cell type counts
  - `"countPlusGenes"`: Cell type counts plus gene expression
  - `"GeneOnly"`: Only gene expression data (not recommended)
- **`minimal_f`**: Minimal frequency threshold for cell types in neighborhoods (float, e.g., 0)
- **`top_n`**: Number of top features/genes to consider (int, e.g., 10)
- **`minimal_gene_threshold`**: Minimum gene expression threshold (float, default: 0.0)

### Prompt Engineering Parameters
- **`with_negatives`**: Boolean indicating whether to include negative examples in prompts
- **`with_numbers`**: Boolean indicating whether to include quantitative information (frequencies/expression levels)
- **`with_CoT`**: Boolean for Chain-of-Thought prompting (default: false)
- **`use_full_name`**: Boolean for using full cell type names vs. abbreviations
- **`with_self_type`**: Boolean for including the cell's own type in neighborhood description
- **`with_region_name`**: Boolean for including tissue region information in prompts
- **`with_domain_name`**: Boolean for including domain-specific terminology
- **`oneshot_prompt`**: String containing few-shot example prompts (empty by default)
- **`system_prompt`**: String containing system-level instructions for the model (empty by default)

### Output and Execution Parameters
- **`output_type`**: Type of output expected (e.g., "niche")
- **`confident_output`**: Boolean indicating whether to extract confidence scores from outputs
- **`replicate`**: String suffix for distinguishing multiple runs (e.g., "_testgeneonly")
- **`prototype_p`**: Float parameter for prototype-based analysis (default: 0.0)




## Example Usage

See `zeroshot_notebook.ipynb` for a complete walkthrough of the analysis pipeline, including:
- Configuration setup
- Data loading and preprocessing
- Neighborhood analysis
- Batch API submission
- Result processing and visualization
- Performance evaluation

