import os
import json
import pickle
from collections import defaultdict

from utils import logger, USER_CACHE_FILE, USERNAME # Import USERNAME
from embedding import EmbeddingManager
from data_parsing import DataParser
from clustering import ClusteringManager
from content_analysis import ContentAnalyzer
from output_formatter import OutputFormatter

class SocialMediaProcessor:
    def __init__(self, auto_cluster=True, n_clusters=None, similarity_threshold=0.7, n_workers=5, cap_ignored_messages=1000):
        self.auto_cluster = auto_cluster
        self.n_clusters_param = n_clusters # Store the parameter
        self.similarity_threshold = similarity_threshold
        self.n_workers = n_workers
        self.cap_ignored_messages = cap_ignored_messages
        self.output_dir = None # Initialize output_dir
        
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.data_parser = DataParser(self.embedding_manager)
        # Pass n_clusters param to ClusteringManager
        self.clustering_manager = ClusteringManager(auto_cluster=auto_cluster, n_clusters=self.n_clusters_param, n_workers=n_workers)
        self.content_analyzer = ContentAnalyzer(similarity_threshold=similarity_threshold)
        # OutputFormatter will need the output_dir later
        self.output_formatter = OutputFormatter(cap_ignored_messages=cap_ignored_messages)
        
        # User data will be loaded from data parser
        self.user_data = self.data_parser.user_data
        
    
    def load_user_cache(self):
        """Load user data cache if it exists"""
        try:
            cache_file_path = os.path.expanduser(USER_CACHE_FILE)
            if os.path.exists(cache_file_path):
                with open(cache_file_path, 'rb') as f:
                    saved_user_data = pickle.load(f)
                    # Update the data parser's user data with the cached data
                    for user_id, data in saved_user_data.items():
                        self.data_parser.user_data[user_id] = data
                    # Reference the data parser's user data
                    self.user_data = self.data_parser.user_data
                logger.info(f"Loaded cached data for {len(self.user_data)} users")
        except Exception as e:
            logger.error(f"Error loading user cache: {str(e)}")
    
    def save_user_cache(self):
        """Save user data cache"""
        try:
            cache_file_path = os.path.expanduser(USER_CACHE_FILE)
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            # Convert defaultdict to regular dict for saving
            with open(cache_file_path, 'wb') as f:
                pickle.dump(dict(self.user_data), f)
            logger.info("Saved user cache successfully")
        except Exception as e:
            logger.error(f"Error saving user cache: {str(e)}")
    
    def process_and_cluster(self, start_date=1, end_date=31, force_parse=False, add_ignored_messages=False):
        """Main process function that handles the entire pipeline"""
        logger.info("Starting data processing and clustering")

        if not force_parse:
            # Load user data cache if it exists
            self.load_user_cache()

        # Step 1: Parse all Bluesky files only if needed
        if force_parse or not self.user_data or len(self.user_data) < 10:
            logger.info("Parsing Bluesky files (cache empty or force parse enabled)...")
            self.data_parser.parse_bluesky_files(start_date, end_date, self.n_workers)
            self.user_data = self.data_parser.user_data
            self.save_user_cache()
        else:
            logger.info(f"Using cached data for {len(self.user_data)} users - skipping parse step")
        
        # Step 2: Filter users (this is done in data_parser)
        
        # Step 3: Compute embeddings for each user
        user_embeddings = self.data_parser.compute_user_embeddings()
        
        # Step 4: Cluster users
        user_clusters, cluster_users = self.clustering_manager.cluster_users(user_embeddings)
        
        # Determine the actual number of clusters found
        actual_n_clusters = len(cluster_users)
        logger.info(f"Clustering resulted in {actual_n_clusters} clusters.")
        
        # Define and create the output directory based on actual clusters
        self.output_dir = os.path.expanduser(f"~/bluesky_blueprint/scratch/processed_{actual_n_clusters}_clusters")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")

        # Pass the output directory to the formatter
        self.output_formatter.set_output_dir(self.output_dir)
        
        # Save clustering results for reference in the correct directory
        with open(os.path.join(self.output_dir, "user_clusters.json"), 'w') as f:
            json.dump(user_clusters, f)
        
        ignored_content_by_cluster = {}
        if add_ignored_messages:
            # Step 5: For each cluster, find seen content and ignored content
            seen_content_by_cluster = {}
            
            for cluster_id, users in cluster_users.items():
                # Skip clusters with too few users
                if len(users) < 5:
                    logger.info(f"Skipping cluster {cluster_id} with only {len(users)} users")
                    continue
                    
                logger.info(f"Processing cluster {cluster_id} with {len(users)} users")
                
                # Find content users in this cluster have likely seen
                seen_content = self.content_analyzer.find_seen_content(cluster_id, cluster_users, self.user_data)
                seen_content_by_cluster[cluster_id] = seen_content
                
                # Extract embeddings from seen content
                content_embeddings = [c['embedding'] for c in seen_content if 'embedding' in c and c['embedding'] is not None]
                
                # Cluster the seen content
                content_centers, content_clusters = self.clustering_manager.cluster_content(content_embeddings)
                
                # Compute weighted centers for seen content
                weighted_centers = self.clustering_manager.compute_weighted_centers(seen_content, content_embeddings, content_clusters)
                
                # Find ignored content
                ignored_content = self.content_analyzer.find_ignored_content(cluster_users, cluster_id, weighted_centers, self.user_data)
                ignored_content_by_cluster[cluster_id] = ignored_content
        
        # Step 6: Compile data for each cluster
        self.output_formatter.compile_cluster_data(cluster_users, ignored_content_by_cluster, self.user_data)
        
        logger.info("Data processing and clustering completed")