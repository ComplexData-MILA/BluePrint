# Social Media Persona Data Processing

This directory contains scripts for processing and clustering social media data from Bluesky.

## Overview

The data processing pipeline:
1. Parses raw Bluesky data files
2. Extracts user posts, replies, likes, reposts, and follows
3. Filter out users without enough english posts
4. Computes embeddings for text content using a multilingual transformer model
5. Clusters users based on content similarity
6. Outputs structured data for each cluster

## User guide

### Clustering
In order to run this pipeline, it is important to note that an initial dataset
is required. A dummy placeholder dataset is provided in `bluesky_blueprint/scratch` to
demonstrate what kind of file structure is expected, but
it should be replaced with your own data. For safety and ethics reasons, we do
not provide our own raw unanonymized data. Some info about the expected input formats
is available in `expected_input_format.md` and `expected_input_format_examples.json`.

After cloning the `bluesky_blueprint` in your **home directory** (important since some paths
in code assume this is in your home directory), cd into it and start by creating a virtual
environment in it using VENV and install the required dependencies

```bash
cd bluesky_blueprint/clustering
virtualenv --no-download ENV-data-processing
source ENV-data-processing/bin/activate
pip install -r requirements.txt
```

You may then run the clustering using

```bash
python src/main.py --n-clusters 25 --start-date 1 --end-date 31 --n-workers 1
```

#### Arguments

- `--auto-cluster`: Use automatic clustering to determine optimal number of clusters
- `--n-clusters`: Number of clusters if not using auto-clustering (default: 10)
- `--similarity-threshold`: Similarity threshold for ignored content (default: 0.7)
- `--start-date`: Start date to process (day of month)
- `--end-date`: End date to process (day of month)
- `--force-parse`: Force parsing of files even if cache exists
- `--n-workers`: Number of workers to use for multiprocessing (default: 5)
- `--add-ignored-messages`: Estimates which messages were seen and then ignored. Adds these to the dataset
- `--cap-ignored-messages`: Cap the number of ignored messages per user (default: 1000)

#### Output

The script generates the following outputs in the specified output directory:
- `cluster_[id].jsonl`: Data files for each cluster containing:
    - Conversation chains
    - Like actions
    - Repost actions
    - Follow actions
    - Ignored content
- `user_clusters.json`: Mapping of users to their assigned clusters

#### Cache Files

To speed up processing, the script maintains cache files:
- `embedding_cache.pkl`: Cache of text embeddings
- `user_data_cache.pkl`: Cache of processed user data

#### Hardware Requirements

If you use the placeholder dataset, this should run on more or less anything, but if
you replace it with your own data, depending on it's size, it could be really
compute-intensive. We would recommend having CUDA enabled, and depending on the
size of your dataset, an appropriate amount of memory. Processing raw data requires
significant RAM. Consider the size of your dataset when choosing it. You may increase --n-workers
in order to speedup the process as multi-core processing is supported for faster
data parsing. For reference, on a 29G dataset, using 10 workers with CUDA enabled
and 160G of memory, this should take between 2-10 hours. If
you wish to run this on a cluster that runs using SLURM, a `job_script.sh` file is
provided, you may simply have to change the account name in the `#SBATCH --account` line.

### Augmenting data for instruction-following

Information available in `instruct_dataset/README.md`.
