#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

# This script is used to run the metrics computation job on a distributed compute cluster.
# It is specifically designed for the Compute Canada environment.
# Documentation on how to use the cluster available at 
# https://docs.alliancecan.ca/wiki/Technical_documentation/en

# You will most likely need to change the account name in the #SBATCH --account line.

module load python/3.11
module load scipy-stack

source ENV-evaluating/bin/activate ## replace with your environment name ##
python compute_metrics.py --force_recompute
# pip freeze --local
