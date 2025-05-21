#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10

# This script is used to run the clustering job on a distributed compute cluster.
# It is specifically designed for the Compute Canada environment.
# Documentation on how to use the cluster available at 
# https://docs.alliancecan.ca/wiki/Technical_documentation/en

# You will most likely need to change the account name in the #SBATCH --account line.

module load python/3.11
module load scipy-stack

source ENV-data-processing/bin/activate #### Replace with your virtual environment path ####
python src/main.py --n-clusters 25 --start-date 1 --end-date 31 --n-workers 10