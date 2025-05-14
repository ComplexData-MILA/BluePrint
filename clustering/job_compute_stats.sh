#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=80G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module load python/3.11
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
# python dataset_filter.py
python get_cluster_stats.py --folder_name "processed_25_clusters"
# pip freeze --local
