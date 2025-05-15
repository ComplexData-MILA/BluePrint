#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1

module load python/3.11
module load scipy-stack
# source ENV-evaluating/env/bin/activate
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python structured_outputs.py
# pip freeze --local
