#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2

echo "meta-llama/Llama-3.2-3B-Instruct finetuning job started"

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python finetuning.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name "meta-llama/Llama-3.2-3B-Instruct" --train_batch_size 1 --eval_batch_size 1 --grad_accumulation_steps 32 --focal_loss

# pip install --no-index -r ../evaluating/requirements.txt
# python ../evaluating/generate_data.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_path "/scratch/s4yor1/meta-llama/Llama-3.2-3B-Instruct-lora-finetuned-$SLURM_ARRAY_TASK_ID" --model_name "meta-llama/Llama-3.2-3B-Instruct" --output "model_3B_$SLURM_ARRAY_TASK_ID"
# pip freeze --local
