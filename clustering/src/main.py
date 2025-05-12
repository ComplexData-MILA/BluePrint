import argparse
import torch
import multiprocessing
import numpy as np
from utils import logger
from processor import SocialMediaProcessor

def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if __name__ == "__main__":
        multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Process Bluesky data and cluster users')
    parser.add_argument('--auto-cluster', action='store_true', help='Use automatic clustering')
    parser.add_argument('--n-clusters', type=int, default=10, help='Number of clusters if not auto-clustering')
    parser.add_argument('--similarity-threshold', type=float, default=0.7, help='Similarity threshold for ignored content')
    parser.add_argument('--start-date', type=int, default=2, help='Start date to process (day of month)')
    parser.add_argument('--end-date', type=int, default=28, help='End date to process (day of month)')
    parser.add_argument('--force-parse', action='store_true', help='Force parsing of files even if cache exists (Make sure to allocate a gpu and multiple CPUs)')
    parser.add_argument('--n-workers', type=int, default=5, help='Number of workers to use for multiprocessing')
    parser.add_argument('--cap-ignored-messages', type=int, default=1000, help='Cap the number of ignored messages per user')
    parser.add_argument('--add-ignored-messages', action='store_true', help='Add ignored messages to the output')

    args = parser.parse_args()
    
    # Check if CUDA is available for faster processing
    if torch.cuda.is_available():
        logger.info("CUDA is available! Using GPU for faster processing.")
    else:
        logger.info("CUDA not available. Using CPU for processing.")
    
    # Create processor instance with command line arguments
    processor = SocialMediaProcessor(
        auto_cluster=args.auto_cluster,
        n_clusters=args.n_clusters,
        similarity_threshold=args.similarity_threshold,
        n_workers=args.n_workers,
        cap_ignored_messages=args.cap_ignored_messages
    )
    
    # Run the main processing pipeline
    processor.process_and_cluster(
        start_date=args.start_date, 
        end_date=args.end_date, 
        force_parse=args.force_parse
    )

if __name__ == "__main__":
    main()