#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python generate_data.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_path "/scratch/s4yor1/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-$SLURM_ARRAY_TASK_ID" --model_name "Qwen/Qwen2.5-7B-Instruct" --output "model_7B_$SLURM_ARRAY_TASK_ID"
# pip freeze --local
