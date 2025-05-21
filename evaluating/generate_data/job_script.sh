#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --array=0,1,6,18,21

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python generate_data_finetuned.py --slurm_array_task_id $SLURM_ARRAY_TASK_ID --use_lora --no_focal
# pip freeze --local
