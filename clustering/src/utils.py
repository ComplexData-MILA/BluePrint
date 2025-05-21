import os
import logging
import getpass
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("social_media_processor")

# Constants
USERNAME = getpass.getuser()
DATA_ROOT = "~/bluesky_blueprint/scratch/placeholder_bluesky-2025-03"
EMBEDDING_CACHE_FILE = "~/bluesky_blueprint/scratch/caches/data_processing_embedding_cache.pkl"
USER_CACHE_FILE = "~/bluesky_blueprint/scratch/caches/data_processing_user_data_cache.pkl"
MODEL_NAME = "intfloat/multilingual-e5-large"  # Embedding model

# Ensure cache directory exists
os.makedirs(os.path.expanduser("~/bluesky_blueprint/scratch/caches"), exist_ok=True)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def default_user_data():
    """Default structure for user data"""
    return {
        'posts': [],
        'replies': [],
        'likes': [],
        'unlikes': [],
        'reposts': [],
        'unreposts': [],
        'follows': [],
        'unfollows': [],
        'blocks': [],
        'unblocks': [],
        'post_updates': [],
        'post_deletes': [],
        'quotes': [],
        'profile': None,
        'post_embeddings': []
    }

def default_actions():
    """Default structure for actions"""
    return {
        "like": False,
        "unlike": False,
        "repost": False,
        "unrepost": False,
        "follow": False,
        "unfollow": False,
        "block": False,
        "unblock": False,
        "post_update": False,
        "post_delete": False,
        "profile_update": False,
        "quote": False,
        "ignore": False,
    }