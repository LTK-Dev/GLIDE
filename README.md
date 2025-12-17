# GLIDE: Graph-based Lateral Movement Detection with LLM-Enhanced Analysis

[![Paper](https://img.shields.io/badge/Paper-Insider%20Threat%20Detection-blue.svg)](https://github.com/LTK-Dev/INT3220E_1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**📄 Research Paper:** _Leveraging Graph-based Reasoning and Large Language Models for Insider Threat Detection_

## Project Overview

GLIDE (Graph-based Lateral movement Intrusion Detection and Evaluation) is a comprehensive framework for detecting lateral movement in enterprise networks using graph-based learning and LLM-enhanced analysis. The project combines advanced graph neural networks (GraphSAGE/Metapath2Vec) with ensemble methods and LLM reasoning to achieve accurate threat detection.

## How to Run the Code

This project was originally designed to run on Kaggle with T4 GPU. Below is a detailed guide to reproduce the entire pipeline.

## System Requirements

- Python 3.x
- GPU (recommended: T4 or equivalent)
- Jupyter Notebook or Kaggle environment

## Data Preparation

### 1. Download LANL Dataset

- Original dataset: [LANL Tracker Data](https://www.kaggle.com/datasets/kietle277/lmtrackerdata/data)
- This dataset will be used as input for the first step (extract_redteam_auth)

### 2. Download Pre-trained Embeddings for Metapath2Vec

- Dataset: [Word to Vec - Ben Ngoai](https://www.kaggle.com/datasets/kietle277/wordtovec-benngoai/data)
- This is the output data from the metapath2vec step

## Execution Order

Run the notebooks in the following order:

### Step 1: Extract Red Team Authentication

```bash
extract_redteam_auth.ipynb
```

- **Input**: LANL dataset from Kaggle
- **Output**: `data/extract_redteam_auth/red_team_auth.txt`
- **Purpose**: Extract red team authentication information including auth type, logon type, and logon orientation

### Step 2: Graph Construction

```bash
graph_construct.ipynb
```

- **Input**: Output from Step 1 (`data/extract_redteam_auth/`)
- **Output**: `data/graph_construct/graph_data_<timestamp>/`
  - `graph_data_torch.pt`
  - `user2nodeid.pt`, `computer2nodeid.pt`, `process2nodeid.pt`
  - Metapath files: `*_path_CUC.pt`, `*_path_UCAC.pt`, `*_path_UCC.pt`, `*_path_UCCA.pt`
  - Train/test split files
- **Purpose**: Construct graph from authentication data

### Step 3: Metapath2Vec

```bash
metapath2vec.ipynb
```

- **Input**: Output from Step 2 (`data/graph_construct/`)
- **Output**: [Word to Vec embeddings](https://www.kaggle.com/datasets/kietle277/wordtovec-benngoai/data)
- **Purpose**: Learn node embeddings in the graph through metapath

#### Alternative: GraphSAGE (Appendix)

As an alternative to Metapath2Vec, you can use GraphSAGE for learning node embeddings:

**Step 3a: GraphSAGE**

```bash
graphsage/graphsage.ipynb
```

- **Input**: Output from Step 2 - Graph Construction (`data/graph_construct/graph_data_<timestamp>/`)
- **Output**: GraphSAGE node embeddings (saved in `graphsage/` directory)
- **Purpose**: Learn node embeddings using GraphSAGE, a scalable graph neural network approach that samples and aggregates features from a node's local neighborhood

**Step 3b: Autoencoder with GraphSAGE** (Optional enhancement)

```bash
graphsage/autoencoder_with_graphsage.ipynb
```

- **Input**: Output from Step 3a - GraphSAGE embeddings (`graphsage/` directory)
- **Output**: Enhanced embeddings through autoencoder
- **Purpose**: Further refine GraphSAGE embeddings using autoencoder for better feature representation

### Step 4: Hybrid Ensemble

```bash
hybrid_ensemble.ipynb
```

- **Input**:
  - Output from Step 2 (graph data)
  - Output from Step 3 (embeddings)
- **Output**: Lateral movement detection results
- **Purpose**: Combine ensemble methods for lateral movement detection

### Step 5: ReAct Agent

Go in the `react_agent` folder then follow the instruction there

### Step 6: Data Visualization and Analysis

```bash
analysis.ipynb
```

- **Input**: Output from Step 5 (ReAct Agent predictions - CSV file with predictions)
- **Output**:
  - Comprehensive visualizations (confusion matrices, ROC curves, score distributions)
  - Performance metrics comparison tables
  - Exported CSV files with analysis results
- **Purpose**: Visualize and analyze the prediction results to facilitate easier interpretation and insights:
  - Compare model performance metrics (Accuracy, Precision, Recall, F1-Score, AUC)
  - Visualize confusion matrices and ROC curves
  - Analyze score distributions and threshold effects
  - Track performance across different seeds
  - Identify improved/degraded samples after LLM reasoning
  - Generate exportable reports for presentation

## Important Notes

- The code is designed to run in Kaggle environment; you may need to adjust paths if running locally
- Each step creates a separate output folder in the `data/` directory
- Ensure sufficient storage space for intermediate files
- T4 GPU is recommended for faster training

## Directory Structure After Execution

```
INT3220E_1/
├── extract_redteam_auth.ipynb
├── graph_construct.ipynb
├── metapath2vec.ipynb
├── hybrid_ensemble.ipynb
├── README.md
└── data/
    ├── extract_redteam_auth/
    │   └── red_team_auth.txt
    └── graph_construct/
        └── graph_data_<timestamp>/
            ├── graph_data_torch.pt
            ├── user2nodeid.pt
            ├── computer2nodeid.pt
            ├── process2nodeid.pt
            └── ... (metapath files)
```

## Dependencies

Main libraries used:

- `torch`
- `torch_geometric`
- `dgl`
- `pandas`
- `numpy`
- `matplotlib`

See each notebook for specific dependency details.
