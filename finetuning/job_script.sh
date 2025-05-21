#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=48G
#SBATCH --gpus-per-node=1
#SBATCH --array=0,1,6,18,21

# This script is used to run the finetuning job on a distributed compute cluster.
# It is specifically designed for the Compute Canada environment.
# Documentation on how to use the cluster available at 
# https://docs.alliancecan.ca/wiki/Technical_documentation/en

# You will most likely need to change the account name in the #SBATCH --account line.

echo "Qwen/Qwen2.5-7B-Instruct finetuning job started"

module load python/3.11
module load scipy-stack

source ENV-finetuning/bin/activate ## replace with your environment name ##
python finetuning.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name "Qwen/Qwen2.5-7B-Instruct" --train_batch_size 2 --eval_batch_size 2 --grad_accumulation_steps 16 --focal_loss