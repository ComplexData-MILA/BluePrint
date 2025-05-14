#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=160G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

module load python/3.11
module load scipy-stack
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip

source ENV-data-processing/bin/activate

# pip install --no-index -r requirements.txt
# pip install k-means-constrained==0.7.3
# python dataset_filter.py
python src/main.py --n-clusters 1000 --start-date 1 --end-date 31 --n-workers 10 --similarity-threshold 0.9 --add-ignored-messages
# pip freeze --local

# 4h, 160g, 1gpu, 5cpus, 10 workers