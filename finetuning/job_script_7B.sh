#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=48G
#SBATCH --gpus-per-node=1
#SBATCH --array=0,1,6,18,21

echo "Qwen/Qwen2.5-7B-Instruct finetuning job started"

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python finetuning.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name "Qwen/Qwen2.5-7B-Instruct" --train_batch_size 2 --eval_batch_size 2 --grad_accumulation_steps 16 --focal_loss

# pip install --no-index -r ../evaluating/requirements.txt
# python ../evaluating/generate_data.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_path "/scratch/s4yor1/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-$SLURM_ARRAY_TASK_ID" --model_name "Qwen/Qwen2.5-7B-Instruct" --output "model_7B_$SLURM_ARRAY_TASK_ID"
# pip freeze --local
