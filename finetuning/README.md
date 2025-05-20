# LLM Fine-tuning for Social Media Persona Simulation

This directory contains scripts and configurations for fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) on social media data, potentially derived from the clustering process. The goal is to adapt pre-trained models to better simulate specific user personas or conversational styles found in the dataset.

## Overview

The fine-tuning pipeline involves:
1.  Loading a pre-trained Hugging Face transformer model and tokenizer (e.g., Qwen models).
2.  Preparing a dataset consisting of conversation chains and instruction-following examples.
3.  Applying LoRA for efficient fine-tuning, targeting specific modules of the model.
4.  Training the model using the Hugging Face `Trainer` API, with options for custom loss functions like Focal Loss.
5.  Saving the trained LoRA adapters and tokenizer for later use in inference or evaluation.
6.  Support for running fine-tuning jobs on a SLURM cluster using array tasks to process different data segments (e.g., clusters).

## User Guide

### Prerequisites

1.  **Project Setup**:
    *   Ensure the `bluesky_blueprint` repository is cloned, preferably in your home directory, as some default paths are constructed relative to `~`.
    *   The fine-tuning script expects a `prompts.toml` file in the `../shared/` directory relative to `finetuning.py`.

2.  **Python Environment**:
    *   Navigate to the `bluesky_blueprint/finetuning` directory.
    *   Create a virtual environment and activate it:
        ```bash
        cd ~/bluesky_blueprint/finetuning
        virtualenv ENV-finetuning 
        source ENV-finetuning/bin/activate
        ```
    *   Install the required Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Input Data**:
    *   **Main Dataset**: Clustered conversation data, typically located at a path like `~/bluesky_blueprint/scratch/pii_removed/processed_25_clusters_hashed/cluster_{id}.jsonl`. Each file should contain JSON objects with a "chains" key.
    *   **Instruction-Following Dataset**: Data formatted for instruction tuning, typically at a path like `~/bluesky_blueprint/scratch/instruction_following/cluster_{id}.jsonl`.
    *   The script uses `args.slurm_array_task_id` (or a default value) to select the specific cluster file.

4.  **Hugging Face Cache**:
    *   The script sets `HF_HOME` to `~/bluesky_blueprint/scratch/HF-cache`. Ensure this directory is writable or modify the path in `finetuning.py` if needed.

### Running the Fine-tuning

#### Directly using `finetuning.py`

You can run the fine-tuning script directly from the terminal:

```bash
python finetuning.py [ARGUMENTS]
```

**Arguments for `finetuning.py`**:

*   `--slurm_array_task_id` (int, default: `0`): Task ID, typically used to select a specific cluster data file (e.g., `cluster_0.jsonl`).
*   `--model_name` (str, default: `"Qwen/Qwen2.5-3B-Instruct"`): The Hugging Face model identifier for the base model to be fine-tuned.
*   `--train_batch_size` (int, default: `6`): Batch size for training.
*   `--eval_batch_size` (int, default: `4`): Batch size for evaluation.
*   `--grad_accumulation_steps` (int, default: `8`): Number of steps for gradient accumulation.
*   `--focal_loss` (action, default: `False`): If specified, uses Focal Loss instead of the default cross-entropy loss.

#### Using `job_script.sh` on a SLURM Cluster

For running on a SLURM-managed cluster, a `job_script.sh` is provided. This script handles environment setup and runs `finetuning.py` as an array job, allowing parallel fine-tuning on different data clusters.

**Before running `job_script.sh`**:

1.  **Modify Account**: Change the `#SBATCH --account=def-rrabba` line to your SLURM account.
2.  **Verify Virtual Environment Path**: Ensure the `source ENV-finetuning/bin/activate` line correctly points to your virtual environment.
3.  **Adjust SLURM Array**: Modify `#SBATCH --array=0,1,6,18,21` to specify the task IDs (and thus cluster files) you want to process.

**Submit the job**:

```bash
sbatch job_script.sh
```
The `job_script.sh` passes the `$SLURM_ARRAY_TASK_ID` to `finetuning.py`.

### Key Features

*   **Model Support**: Uses Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`, compatible with a wide range of causal language models.
*   **LoRA Fine-tuning**: Implements LoRA using the `peft` library for parameter-efficient fine-tuning. Configuration targets `q_proj`, `k_proj`, `v_proj`, `o_proj` modules by default.
*   **Data Processing**:
    *   Loads JSONL datasets.
    *   Formats conversation chains and user histories.
    *   Applies model-specific chat templates (example provided for Qwen models).
    *   Tokenizes data for training.
*   **Training**:
    *   Utilizes Hugging Face `Trainer` and `TrainingArguments`.
    *   Supports mixed-precision training (`torch.bfloat16`).
    *   Option to use Focal Loss for potentially better handling of imbalanced data.
*   **Distributed Training**: While `device_map="auto"` is used, the `job_script.sh` requests a single GPU. For multi-GPU or other distributed setups, `TrainingArguments` and the SLURM script might need adjustments.

### Output

The fine-tuning script generates the following outputs:

*   **Fine-tuned Model Adapters**: Saved in a directory structure like:
    `~/bluesky_blueprint/scratch/{MODEL_NAME}-lora-finetuned-{task_id}{'-no-focal' or ''}`.
    This directory will contain the LoRA adapter weights (`adapter_model.safetensors`), configuration (`adapter_config.json`), and the tokenizer.
*   **Evaluation Dataset Samples**: A subset of the validation dataset (`NUM_SAVED_VALIDATION_SAMPLES`, default 1000) is saved to:
    `~/bluesky_blueprint/scratch/evaluation_dataset/validation_samples_cluster_{task_id}.jsonl`
    and
    `~/bluesky_blueprint/scratch/evaluation_dataset/instruction_validation_samples_cluster_{task_id}.jsonl`.

### Hardware Requirements

*   **GPU**: A CUDA-enabled GPU is highly recommended for feasible training times. The script checks for CUDA availability. The `job_script.sh` requests one GPU.
*   **Memory (RAM & VRAM)**: Depends significantly on the base model size (`--model_name`) and batch size. Larger models require more VRAM. The `job_script.sh` requests 48G of CPU memory per CPU, but VRAM is the more critical factor for the model itself.

### Dependencies

Key Python libraries used:
*   `transformers`
*   `peft`
*   `datasets`
*   `torch`
*   `accelerate`
*   `toml` (for reading `prompts.toml`)

Refer to `requirements.txt` for a full list of dependencies and their versions.
