import numpy as np
# from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import logger, cosine_similarity

class ClusteringManager:
    def __init__(self, auto_cluster=True, n_clusters=None, n_workers=5):
        self.auto_cluster = auto_cluster
        self.n_clusters = n_clusters
        self.n_workers = n_workers
    
    def cluster_users(self, user_embeddings):
        """Cluster users using KMeans and elbow method or fixed clusters with parallel processing"""
        if not user_embeddings:
            logger.error("No user embeddings to cluster")
            return {}, {}
        
        # Extract user IDs and embeddings
        user_ids = list(user_embeddings.keys())
        embeddings = np.array(list(user_embeddings.values()))
        
        # Determine optimal number of clusters using elbow method if auto_cluster is True
        if self.auto_cluster:
            logger.info("Determining optimal number of clusters using elbow method with multiprocessing")
            
            # Try a range of cluster numbers
            max_clusters = min(10, len(user_ids) // 10)  # Limit max clusters 
            if max_clusters < 2:
                max_clusters = 2
                
            k_range = range(2, max_clusters + 1)
            
            # Use multiprocessing to compute scores for different k values
            with Pool(processes=min(cpu_count(), len(k_range), self.n_workers)) as pool:
                results = list(tqdm(
                    pool.starmap(self.compute_scores_for_k, [(k, embeddings) for k in k_range]),
                    total=len(k_range),
                    desc="Finding optimal clusters"
                ))
            
            # Process results
            inertias = []
            silhouette_scores = []
            valid_k_values = []
            
            for k, inertia, silhouette in results:
                if inertia is not None:
                    valid_k_values.append(k)
                    inertias.append(inertia)
                    if silhouette is not None:
                        silhouette_scores.append(silhouette)
            
            # Find optimal k based on silhouette scores
            if silhouette_scores:
                optimal_k_idx = np.argmax(silhouette_scores)
                optimal_k = valid_k_values[optimal_k_idx] if optimal_k_idx < len(valid_k_values) else 3
            else:
                optimal_k = 3  # Default if no valid silhouette scores
            
            logger.info(f"Optimal number of clusters determined to be: {optimal_k}")
            n_clusters = optimal_k
        else:
            n_clusters = self.n_clusters
            logger.info(f"Using fixed number of clusters: {n_clusters}")
        
        # Perform clustering with optimal or fixed number of clusters
        kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=10, random_state=42, n_jobs=10, max_iter=300) # reduce max_iter if n_clusters large
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Map users to clusters
        user_clusters = {}
        cluster_users = {}
        
        for i, user_id in enumerate(user_ids):
            cluster_id = int(cluster_labels[i])
            user_clusters[user_id] = cluster_id
            
            # Add user to cluster
            if cluster_id not in cluster_users:
                cluster_users[cluster_id] = []
            cluster_users[cluster_id].append(user_id)
        
        return user_clusters, cluster_users

    def compute_scores_for_k(self, k, embeddings):
        """Compute inertia and silhouette scores for a specific k value"""
        logger.info(f"Computing scores for k={k}")
        if len(embeddings) < k:
            return k, None, None
            
        kmeans = KMeansConstrained(n_clusters=k, size_min=10, random_state=42, n_jobs=10, max_iter=300) # reduce max_iter if n_clusters large
        kmeans.fit(embeddings)
        inertia = kmeans.inertia_
        
        silhouette = None
        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette = silhouette_score(embeddings, kmeans.labels_)
        
        logger.info(f"Computed scores for k={k}: inertia={inertia}, silhouette={silhouette}")
        return k, inertia, silhouette
    
    def cluster_content(self, content_embeddings, max_clusters=10):
        """Cluster content using KMeans"""
        logger.info(f"Clustering {len(content_embeddings)} content items")
        
        if not content_embeddings:
            logger.warning("No content embeddings to cluster")
            return [], {}
        
        # Convert to numpy array
        embeddings_array = np.array(content_embeddings)
        
        # Determine number of clusters - simplified for processing speed
        n_clusters = min(max_clusters, len(content_embeddings) // 20 + 1)
        n_clusters = max(1, n_clusters)  # Ensure at least one cluster
        
        if n_clusters == 1:
            # Just one cluster, all content belongs to it
            cluster_centers = [np.mean(embeddings_array, axis=0)]
            content_clusters = {0: list(range(len(content_embeddings)))}
        else:
            # Perform KMeans clustering
            kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=10, random_state=42, n_jobs=10, max_iter=300)  # reduce max_iter if n_clusters large
            labels = kmeans.fit_predict(embeddings_array)
            cluster_centers = kmeans.cluster_centers_
            
            # Group content by cluster
            content_clusters = {}
            for i, label in enumerate(labels):
                if int(label) not in content_clusters:
                    content_clusters[int(label)] = []
                content_clusters[int(label)].append(i)
        
        return cluster_centers, content_clusters
    
    def compute_weighted_centers(self, content_items, content_embeddings, content_clusters):
        """Compute weighted centers for content clusters"""
        weighted_centers = []
        
        logger.info(f"Computing weighted centers for {len(content_clusters)} clusters")
        
        for cluster_id, content_indices in content_clusters.items():
            if not content_indices:
                continue
                
            # Extract embeddings and weights for this cluster
            cluster_embeddings = []
            cluster_weights = []
            
            for idx in tqdm(content_indices, desc=f"Cluster {cluster_id}"):
                if idx >= len(content_embeddings):
                    continue
                    
                embedding = content_embeddings[idx]
                content = content_items[idx]
                
                # Calculate weight based on interaction and follows
                weight = 1.0  # Base weight
                
                # Increase weight for content with interactions
                if 'interaction_type' in content:
                    weight += 2.0
                
                # Increase weight for content from followed users
                if 'followed_by' in content and content['followed_by']:
                    weight += len(content['followed_by']) * 0.5
                
                cluster_embeddings.append(embedding)
                cluster_weights.append(weight)
            
            if not cluster_embeddings:
                continue
                
            # Convert to numpy arrays
            embeddings_array = np.array(cluster_embeddings)
            weights_array = np.array(cluster_weights)
            
            # Normalize weights to sum to 1
            normalized_weights = weights_array / np.sum(weights_array)
            
            # Compute weighted average
            weighted_center = np.sum(embeddings_array * normalized_weights[:, np.newaxis], axis=0)
            
            weighted_centers.append(weighted_center)
        
        return weighted_centers