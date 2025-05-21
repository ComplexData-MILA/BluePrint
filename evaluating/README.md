# Model Evaluation for Social Media Persona Simulation

This directory contains scripts and utilities for evaluating the performance of fine-tuned Large Language Models (LLMs) in the context of social media persona simulation. The evaluation process typically involves generating responses from the models on test datasets and then computing various metrics to assess their quality, and persona consistency.

## Overview

The evaluation pipeline generally consists of two main stages:
1.  **Data Generation**: Using fine-tuned models to generate responses or continue conversations based on evaluation prompts or datasets.
2.  **Metrics Computation**: Calculating quantitative metrics on the generated data.

Scripts are provided to automate these stages, with support for SLURM cluster execution for larger-scale evaluations.

## User Guide

### Prerequisites

1.  **Project Setup**:
    *   Ensure the `bluesky_blueprint` repository is cloned, preferably in your home directory.
    *   Fine-tuned models (LoRA adapters and tokenizer configurations) should be available, typically from the `bluesky_blueprint/finetuning/scratch/` directory.
    *   Evaluation datasets should be prepared. These might be held-out sets from the clustering/instruction generation phase or specifically crafted evaluation prompts.
    *   The scripts might rely on `prompts.toml` from the `../shared/` directory for system prompts during generation.

2.  **Python Environment**:
    *   Navigate to the `bluesky_blueprint/evaluating` directory.
    *   It's recommended to use a virtual environment similar to the one used for fine-tuning, or create a new one and install dependencies:
        ```bash
        cd ~/bluesky_blueprint/evaluating
        virtualenv ENV-evaluating
        source ENV-evaluating/bin/activate
        pip install -r requirements.txt
        ```

3.  **Input Data Paths**:
    *   Paths to fine-tuned model adapters.
    *   Paths to evaluation datasets (e.g., `~/bluesky_blueprint/scratch/evaluation_dataset/`).

### Running the Evaluation Scripts

#### 1. Generating Data for Evaluation

See `generate_data` folder.

#### 2. Computing Metrics (`compute_metrics.py`)

This script takes the generated data and reference data to compute evaluation metrics.

**Command-line usage (example)**:
```bash
python compute_metrics.py \
    --generated_file <path_to_generated_data.jsonl> \
    --reference_file <path_to_reference_ground_truth.jsonl> \
    --metrics <list_of_metrics_to_compute> \
    --output_file <path_to_save_metrics_results.json> \
    # ... other relevant arguments
```

**Using `job_script.sh` on a SLURM Cluster**:
This script automates the execution of `compute_metrics.py`.

*   **Modify** the SLURM directives and script parameters within `job_script.sh`.
*   **Submit the job**:
    ```bash
    sbatch job_script.sh
    ```

### Key Scripts and Components

*   `compute_metrics.py`: Calculates various performance metrics on the generated outputs.
*   `utils.py`: Contains helper functions and utility classes used by the evaluation scripts (e.g., for data loading, metric calculation helpers).
*   `job_script.sh`: SLURM job script, likely for `compute_metrics.py` or other evaluation tasks.
*   `requirements.txt`: Lists Python dependencies for the evaluation environment.
*   `compare_vs_gpt/`: Subdirectory potentially containing scripts or resources for comparative analysis against baseline models like GPT.

### Input Data

*   **Fine-tuned Models**: LoRA adapter weights and configurations from the fine-tuning stage.
*   **Base Model**: The original pre-trained model used for fine-tuning.
*   **Evaluation Datasets**: JSONL files or other formats containing prompts, conversation histories, or reference texts for the tasks being evaluated.

### Output

*   **Generated Data**: Files (e.g., JSONL) containing the outputs produced by the fine-tuned models on the evaluation datasets.
*   **Metrics Reports**: Files (e.g., JSON, CSV, text) summarizing the computed performance metrics for each model or experiment.

### Computed Metrics

The `compute_metrics.py` script calculates several metrics to evaluate model performance. These include:

*   **Semantic Similarity using Embeddings:** These metrics assess how semantically similar the generated text is to the reference (real) text.
    *   **Average Embedding Cosine Similarity**: Measures the cosine similarity between the mean embeddings of the entire set of real texts and the generated texts. A higher score indicates greater overall semantic similarity.
    *   **Maximum Cosine Similarity**: For each generated text, this finds the highest cosine similarity to any text in the real dataset. The maximum of these scores is reported, indicating how well the best-generated samples match the real data.
    *   **Embedding JS Divergence**: Calculates the Jensen-Shannon divergence between the distributions of text embeddings from the real and generated datasets. A lower score suggests that the two distributions are more similar.

*   **Topic Distribution Analysis:**
    *   **TF-IDF Jaccard Similarity**: This metric compares the most important terms (identified by TF-IDF) in the real and generated texts. It computes the Jaccard similarity between the sets of top N terms, where a higher score means more overlap in topics.

*   **Action Statistics:**
    *   **F1 Score**: When evaluating social media interactions, models might also predict actions (e.g., like, reply, repost). This F1 score measures the accuracy of these predicted actions against the ground truth actions in the evaluation dataset.

*   **Diagonality Scores**: This set of scores evaluates a model's ability to capture the specific characteristics of the persona (cluster) it was trained on. For each of the following metrics, a diagonality score is computed. A higher diagonality score indicates that a model performs better on data from its own persona/cluster compared to data from other personas/clusters.
    *   Jaccard Similarity
    *   Average Embedding Cosine Similarity
    *   Maximum Cosine Similarity
    *   F1 Scores (for actions)
    *   Inverted Embedding JS Divergence (since lower JS Divergence is better, the value is inverted for the diagonality calculation, so higher is still better).

The script outputs these metrics in a JSON file, allowing for detailed analysis of model performance across different aspects and personas.

### Hardware Requirements

*   **CPU/Memory**: `compute_metrics.py` might not be GPU-intensive but could require significant CPU and RAM depending on the dataset size.

### Dependencies

Key Python libraries are listed in `requirements.txt`. These typically include:
*   `transformers`
*   `peft` (if loading LoRA models)
*   `datasets`
*   `torch`

Refer to `requirements.txt` for the complete list.