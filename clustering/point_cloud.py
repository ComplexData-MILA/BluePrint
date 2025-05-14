import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

# Adjust path to import from src subdirectory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from embedding import EmbeddingManager
    from data_parsing import DataParser
    from utils import logger as util_logger # Try to import logger from utils
except ImportError as e:
    print(f"Failed to import from src: {e}. Ensure src modules are accessible.")
    # Fallback logger if utils.logger cannot be imported
    util_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Define necessary classes if imports fail, for basic script structure
    # This is a fallback, ideally imports should succeed.
    if "EmbeddingManager" not in globals():
        class EmbeddingManager:
            def __init__(self): util_logger.warning("Using mock EmbeddingManager.")
    if "DataParser" not in globals():
        class DataParser:
            def __init__(self, em): util_logger.warning("Using mock DataParser.")
            def parse_bluesky_files(self, start_date, end_date, n_workers): 
                util_logger.warning("Mock parse_bluesky_files called.")
                self.user_data = {}
            def compute_user_embeddings(self): 
                util_logger.warning("Mock compute_user_embeddings called.")
                return {}

logger = util_logger

# --- Configuration ---
USER_CLUSTERS_FILE = "/home/s4yor1/scratch/processed_25_clusters/user_clusters.json"
OUTPUT_PLOT_FILE = "/home/s4yor1/SM-based-personas/data_processing/user_embedding_point_cloud.png"
OUTPUT_JSON_DATA_FILE = "/home/s4yor1/SM-based-personas/data_processing/user_embedding_point_cloud_data.json" # New output file for JSON data
# Parameters for data_parser.parse_bluesky_files
# These should ideally match the parameters used when generating the cluster file
PARSE_START_DATE = 1
PARSE_END_DATE = 31 
PARSE_N_WORKERS = 5

def main():
    logger.info("Starting user embedding point cloud generation...")

    # 1. Initialize and load data
    logger.info("Initializing EmbeddingManager and DataParser...")
    try:
        embedding_manager = EmbeddingManager() # Loads its own cache by default
        data_parser = DataParser(embedding_manager)
    except Exception as e:
        logger.error(f"Error initializing EmbeddingManager or DataParser: {e}")
        return

    logger.info(f"Parsing Bluesky files (from day {PARSE_START_DATE} to {PARSE_END_DATE}) to populate user data. This will use embedding cache if available.")
    try:
        # This step populates data_parser.user_data, which is needed for compute_user_embeddings
        data_parser.parse_bluesky_files(start_date=PARSE_START_DATE, end_date=PARSE_END_DATE, n_workers=PARSE_N_WORKERS)
    except Exception as e:
        logger.error(f"Error during parse_bluesky_files: {e}")
        return

    logger.info("Computing average user embeddings...")
    try:
        user_avg_embeddings = data_parser.compute_user_embeddings() # Dict: {user_id: avg_embedding_vector}
    except Exception as e:
        logger.error(f"Error during compute_user_embeddings: {e}")
        return

    if not user_avg_embeddings:
        logger.error("No user embeddings were computed. Exiting.")
        return
    logger.info(f"Computed average embeddings for {len(user_avg_embeddings)} users.")

    # 2. Load cluster information
    logger.info(f"Loading user cluster data from {USER_CLUSTERS_FILE}...")
    if not os.path.exists(USER_CLUSTERS_FILE):
        logger.error(f"Cluster file not found: {USER_CLUSTERS_FILE}. Please ensure this file exists and was generated with compatible data.")
        return
    try:
        with open(USER_CLUSTERS_FILE, 'r') as f:
            user_to_cluster_map = json.load(f) # Expected format: {user_id: cluster_id}
    except Exception as e:
        logger.error(f"Error loading or parsing cluster file {USER_CLUSTERS_FILE}: {e}")
        return
    logger.info(f"Loaded cluster map for {len(user_to_cluster_map)} users.")

    # 3. Prepare data for PCA
    user_ids_list = []
    embeddings_list = []
    cluster_ids_list = []

    logger.info("Aligning embeddings with cluster data for PCA...")
    for user_id, avg_embedding in user_avg_embeddings.items():
        if user_id in user_to_cluster_map:
            if isinstance(avg_embedding, np.ndarray): # Ensure embedding is a numpy array
                user_ids_list.append(user_id)
                embeddings_list.append(avg_embedding)
                cluster_ids_list.append(user_to_cluster_map[user_id])
            else:
                logger.warning(f"User {user_id} has an invalid embedding type: {type(avg_embedding)}. Skipping.")
        # else:
            # logger.debug(f"User {user_id} found in embeddings but not in cluster map. Skipping.")


    if not embeddings_list:
        logger.error("No users found that are in both embeddings (with valid format) and cluster map. Cannot perform PCA. Exiting.")
        return
    
    logger.info(f"Prepared {len(embeddings_list)} users for PCA.")
    embeddings_matrix = np.array(embeddings_list)
    
    if embeddings_matrix.shape[0] < 2:
        logger.error(f"Not enough samples ({embeddings_matrix.shape[0]}) for PCA with 2 components. Exiting.")
        return
    if embeddings_matrix.ndim != 2:
        logger.error(f"Embeddings matrix has incorrect dimensions: {embeddings_matrix.shape}. Expected 2 dimensions (n_samples, n_features). Exiting.")
        return


    # 4. Apply PCA
    logger.info("Applying PCA to project embeddings to 2D...")
    try:
        pca = PCA(n_components=2, random_state=42)
        projected_embeddings = pca.fit_transform(embeddings_matrix)
    except Exception as e:
        logger.error(f"Error during PCA transformation: {e}")
        return

    # Save the projected data to a JSON file
    logger.info(f"Saving projected point data to {OUTPUT_JSON_DATA_FILE}...")
    point_cloud_data = []
    for i in range(len(user_ids_list)):
        point_cloud_data.append({
            "user_id": user_ids_list[i],
            "x": float(projected_embeddings[i, 0]),
            "y": float(projected_embeddings[i, 1]),
            "cluster_id": int(cluster_ids_list[i])
        })
    
    try:
        with open(OUTPUT_JSON_DATA_FILE, 'w') as f_json:
            json.dump(point_cloud_data, f_json, indent=4)
        logger.info(f"Successfully saved point cloud data to {OUTPUT_JSON_DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving point cloud data to JSON: {e}")

    # 5. Visualization
    logger.info("Generating plot...")
    plt.figure(figsize=(14, 10)) # Adjusted size for legend
    
    unique_cluster_ids = sorted(list(set(cluster_ids_list)))
    num_unique_clusters = len(unique_cluster_ids)
    
    colors_for_plot = []
    if num_unique_clusters > 0:
        if num_unique_clusters <= 10: # Standard matplotlib cycle
             prop_cycle = plt.rcParams['axes.prop_cycle']
             base_colors = prop_cycle.by_key()['color']
             colors_for_plot = [base_colors[i % len(base_colors)] for i in range(num_unique_clusters)]
        elif num_unique_clusters <= 20: # tab20 offers 20 distinct colors
            try:
                cmap = plt.cm.get_cmap('tab20', num_unique_clusters)
                colors_for_plot = [cmap(i) for i in range(num_unique_clusters)]
            except ValueError: # Fallback if specific number of colors not supported
                 colors_for_plot = [plt.cm.viridis(i/num_unique_clusters) for i in range(num_unique_clusters)]
        else: # For more than 20 clusters, spread viridis
            colors_for_plot = [plt.cm.viridis(i/num_unique_clusters) for i in range(num_unique_clusters)]

    cluster_to_idx_map = {cluster_id: i for i, cluster_id in enumerate(unique_cluster_ids)}

    for cluster_id_val in unique_cluster_ids:
        # Select points belonging to the current cluster_id_val
        current_cluster_points_indices = [i for i, c_id in enumerate(cluster_ids_list) if c_id == cluster_id_val]
        if not current_cluster_points_indices:
            continue
            
        cluster_projected_coords = projected_embeddings[current_cluster_points_indices]
        
        color_idx = cluster_to_idx_map[cluster_id_val]
        
        plt.scatter(cluster_projected_coords[:, 0], cluster_projected_coords[:, 1], 
                    label=f'Cluster {cluster_id_val}', 
                    color=colors_for_plot[color_idx % len(colors_for_plot)], # Modulo for safety
                    alpha=0.7)

    plt.title('User Embeddings 2D PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    if num_unique_clusters > 0:
        if num_unique_clusters > 15:
            plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        else:
            plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.80) # Adjust layout to make space for legend
    else:
        logger.warning("No clusters to plot in legend.")

    plt.grid(True)

    try:
        plt.savefig(OUTPUT_PLOT_FILE)
        logger.info(f"Point cloud plot saved to {OUTPUT_PLOT_FILE}")
    except Exception as e:
        logger.error(f"Error saving plot to {OUTPUT_PLOT_FILE}: {e}")
    
    # plt.show() # Uncomment to display interactively if in a suitable environment

if __name__ == "__main__":
    # Setup basic logging if util_logger is not fully configured (e.g. running script directly)
    if not util_logger.hasHandlers() or util_logger.getEffectiveLevel() > logging.INFO :
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Ensure logger variable is the one configured here if it was a fallback
        if 'util_logger' in globals() and util_logger.name == __name__ : # if it's the fallback logger
             logger = logging.getLogger(__name__)
        else: # if util_logger was imported, re-get it to apply config
             logger = logging.getLogger(util_logger.name if 'util_logger' in globals() else __name__)


    main()
