#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=0:01:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module load python/3.11
module load scipy-stack
ls -a
pip freeze --local
source ENV-data-processing/bin/activate
pip freeze --local