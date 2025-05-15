#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python compute_metrics.py --force_recompute
# pip freeze --local
