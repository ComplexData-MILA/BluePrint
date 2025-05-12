import os
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils import logger, MODEL_NAME, EMBEDDING_CACHE_FILE

class EmbeddingManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.embedding_cache = {}
        self.initialize_model()
        self.load_cache()
    
    def initialize_model(self):
        """Initialize the embedding model for this process"""
        logger.info(f"Process {torch.multiprocessing.current_process().pid} loading embedding model")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Process {torch.multiprocessing.current_process().pid} using device: {self.device}")
    
    def load_cache(self):
        """Load embedding cache if it exists"""
        try:
            if os.path.exists(EMBEDDING_CACHE_FILE):
                with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Error loading embedding cache: {str(e)}")
    
    def save_cache(self):
        """Save embedding cache"""
        try:
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {str(e)}")
    
    def get_embedding(self, text):
        """Get embedding for text with caching"""
        if not text or text.strip() == "":
            # Return zero embedding with proper dimensions for the model
            return np.zeros(self.model.config.hidden_size)
        
        text_key = text[:1000]  # Limit key size for very long texts
        if text_key in self.embedding_cache:
            return self.embedding_cache[text_key]
        
        # Properly compute embedding using the transformer model
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        # Get embedding from model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding (first token of last hidden state)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        self.embedding_cache[text_key] = embedding
        return embedding

    def update_cache(self, new_cache):
        """Update the embedding cache with new entries"""
        self.embedding_cache.update(new_cache)
        
    def get_cache(self):
        """Return the current embedding cache"""
        return self.embedding_cache