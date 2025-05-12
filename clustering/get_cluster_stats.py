import os
import json
import re
from tqdm import tqdm
import getpass
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import cdist
import pickle
import argparse

parser = argparse.ArgumentParser(description='Compute the statistics of clusters')
parser.add_argument("--folder_name", type=str, help="Name of the folder that contains the clusters", default="processed_2_clusters")

args = parser.parse_args()

USERNAME = getpass.getuser()
PATH = f"/scratch/{USERNAME}/{args.folder_name}"
EMBEDDING_CACHE_FILE = f"/scratch/{USERNAME}/caches/data_processing_embedding_cache.pkl"

clusters_stats = {}

def get_stats(data):
    stats = {"total": 0, "posts": 0, "replies": 0, "like": 0, "unlike": 0, 
             "repost": 0, "unrepost": 0, "follow": 0, "unfollow": 0, 
             "block": 0, "unblock": 0, "post_update": 0, "post_delete": 0,
             "quote": 0, "ignore": 0}
    
    for chain in data:
        if len(chain) == 1:
            stats["posts"] += 1
        elif len(chain) > 1 and "actions" in chain[-1]:
            actions = chain[-1]["actions"]
            # Increment counters for each action type
            for action_type in actions:
                if actions[action_type] and action_type in stats:
                    stats[action_type] += 1
                    break
            else:
                # If no action was found, count as a reply
                stats["replies"] += 1
        else:
            stats["replies"] += 1
        
        stats["total"] += 1

    return stats

def get_users(clusters_stats, data):
    for (key, val) in data.items():
        clusters_stats[val]["users"] = clusters_stats[val].get("users", 0) + 1
    return clusters_stats

def get_top_words_tfidf(cluster_data):
    """Compute the top 5 words for each cluster using TF-IDF."""
    # Combine all texts from each chain into one document per cluster
    cluster_texts = []
    
    for chain in cluster_data:
        # Get the text from each message in the chain
        message = chain[-1]
        # Skip messages that are actions
        if "text" in message:
            cluster_texts.append(message["text"])
    
    # Join all texts into one document
    combined_text = " ".join(cluster_texts)
    
    # Return empty list if no text
    if not combined_text.strip():
        return []
    
    return combined_text

def load_embedding_cache():
    """Load the cached embeddings."""
    if os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def compute_medoids(cluster_files, embedding_cache, top_n=3):
    """Compute the top N medoid posts for each cluster."""
    # Group posts by cluster
    cluster_posts = {}
    
    for file_name in tqdm(cluster_files, desc="Collecting posts for medoid computation"):
        cluster_number = int(re.search(r"cluster_(\d+)\.jsonl", file_name).group(1))
        cluster_posts[cluster_number] = []
        
        # Load the data
        with open(os.path.join(PATH, file_name), 'r') as f:
            for line in f:
                chain = json.loads(line)
                if len(chain) >= 1:
                    # Extract text from the last post in the chain
                    last_post = chain[-1]
                    
                    text = last_post.get('text', '')
                    if not text:
                        continue
                    
                    # Get embedding from cache
                    text_key = text[:1000]  # Same key generation as in embedding cache
                    if text_key in embedding_cache:
                        cluster_posts[cluster_number].append({
                            'text': text,
                            'embedding': embedding_cache[text_key]
                        })
    
    # Compute medoids for each cluster
    cluster_medoids = {}
    
    for cluster_id, posts in tqdm(cluster_posts.items(), desc="Computing cluster medoids"):
        if len(posts) < top_n:
            cluster_medoids[cluster_id] = posts  # Take all posts if fewer than top_n
            continue
        
        # Extract embeddings for this cluster
        embeddings = np.array([post['embedding'] for post in posts])
        
        # For very large clusters, use sampling approach to avoid memory issues
        if len(embeddings) > 1000:
            # Sample a subset of posts for medoid computation
            sample_size = min(1000, len(embeddings) // 2)
            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # Compute distances between sample and all posts
            distances = cdist(sample_embeddings, embeddings, 'cosine')
            
            # Sum distances for each candidate medoid
            sum_distances = np.sum(distances, axis=0)
            
            # Get indices of top N medoids
            medoid_indices = np.argsort(sum_distances)[:top_n]
        else:
            # For smaller clusters, use original approach
            # Compute pairwise distances between all posts
            distances = cdist(embeddings, embeddings, 'cosine')
            
            # Compute sum of distances for each post to find medoids
            sum_distances = np.sum(distances, axis=1)
            
            # Get indices of top N medoids (posts with smallest sum of distances)
            medoid_indices = np.argsort(sum_distances)[:top_n]
        # Store the medoid posts
        cluster_medoids[cluster_id] = [posts[idx] for idx in medoid_indices]
    
    return cluster_medoids

def compute_lda_topics(cluster_files, num_topics=5, num_words=10):
    """Compute LDA topics for each cluster."""
    cluster_topics = {}
    cluster_texts = {}
    
    # Collect all posts for each cluster
    for file_name in tqdm(cluster_files, desc="Collecting posts for LDA"):
        cluster_number = int(re.search(r"cluster_(\d+)\.jsonl", file_name).group(1))
        texts = []
        
        # Load the data
        with open(os.path.join(PATH, file_name), 'r') as f:
            for line in f:
                chain = json.loads(line)
                post = chain[-1]
                text = post.get('text', '')
                if text:
                    texts.append(text)
        
        cluster_texts[cluster_number] = texts
    
    # Run LDA on each cluster
    for cluster_id, texts in tqdm(cluster_texts.items(), desc="Running LDA on clusters"):
        if len(texts) < 20:  # Skip clusters with too few posts
            cluster_topics[cluster_id] = [["insufficient data"]]
            continue
        
        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        try:
            dtm = vectorizer.fit_transform(texts)
            
            # Run LDA
            lda = LatentDirichletAllocation(
                n_components=min(num_topics, len(texts) // 5),  # Adjust topics based on data size
                random_state=42,
                learning_method='online'
            )
            lda.fit(dtm)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
            
            cluster_topics[cluster_id] = topics
        except Exception as e:
            print(f"Error computing LDA for cluster {cluster_id}: {e}")
            cluster_topics[cluster_id] = [["error computing topics"]]
    
    return cluster_topics

if __name__ == "__main__":
    # Find all files in PATH
    files = os.listdir(PATH)

    # Filter for files of the form "cluster_n.jsonl"
    cluster_files = [f for f in files if re.match(r"cluster_\d+\.jsonl", f)]

    # Load embedding cache
    print("Loading embedding cache...")
    embedding_cache = load_embedding_cache()
    print(f"Loaded {len(embedding_cache)} cached embeddings")

    # Process each cluster file
    cluster_text_data = {}
    for file_name in tqdm(cluster_files):
        # Extract cluster number
        cluster_number = int(re.search(r"cluster_(\d+)\.jsonl", file_name).group(1))
        
        # Load the data
        data = []
        with open(os.path.join(PATH, file_name), 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Get stats for this cluster
        clusters_stats[cluster_number] = get_stats(data)
        
        # Store text data for TF-IDF processing
        cluster_text_data[cluster_number] = get_top_words_tfidf(data)

    with open(os.path.join(PATH, "user_clusters.json"), 'r') as f:
        user_cluster_map = json.load(f)

    clusters_stats = get_users(clusters_stats, user_cluster_map)
    
    # Filter out clusters with only one user
    original_count = len(clusters_stats)
    clusters_stats = {cluster: stats for cluster, stats in clusters_stats.items() 
                     if stats.get("users", 0) > 1}
    removed_count = original_count - len(clusters_stats)
    print(f"\nRemoved {removed_count} clusters with only one user.")

    # Compute TF-IDF for remaining clusters
    remaining_clusters = list(clusters_stats.keys())
    cluster_texts = [cluster_text_data[cluster_id] for cluster_id in remaining_clusters]
    
    # Create and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Extract top 5 words for each cluster
        top_words = {}
        for i, cluster_id in enumerate(remaining_clusters):
            if tfidf_matrix[i].nnz > 0:  # Check if there are non-zero entries
                tfidf_scores = tfidf_matrix[i].toarray().flatten()
                top_indices = tfidf_scores.argsort()[-5:][::-1]
                top_words[cluster_id] = [feature_names[idx] for idx in top_indices]
            else:
                top_words[cluster_id] = ["No distinctive words found"]
    except Exception as e:
        print(f"Error computing TF-IDF: {e}")
        top_words = {cluster_id: ["Error computing TF-IDF"] for cluster_id in remaining_clusters}

    # Recalculate total number of clusters after filtering
    total_clusters = len(clusters_stats)
    if total_clusters == 0:
        print("No clusters with more than one user found!")
        exit()

    # Compute medoids for each cluster
    print("\nComputing cluster medoids...")
    cluster_medoids = compute_medoids(cluster_files, embedding_cache, top_n=3)
    
    # Compute LDA topics for each cluster
    print("\nComputing LDA topics for each cluster...")
    cluster_topics = compute_lda_topics(cluster_files)

    # Print overall summary
    print("\nOverall statistics:")
    for cluster_num, stats in sorted(clusters_stats.items()):
        print(f"Cluster {cluster_num}: {stats}")
        if cluster_num in top_words:
            print(f"  Top words: {', '.join(top_words[cluster_num])}")
    
    # --- Start of Table Generation Logic ---
    # Create a markdown table with stats for first 3 and last 3 clusters
    sorted_clusters = sorted(clusters_stats.items())
    first_three = sorted_clusters[:3]
    last_three = sorted_clusters[-3:]

    # Define the order and labels for statistics
    stats_order = ["users", "total", "posts", "replies", "like", "unlike", "repost", 
                 "unrepost", "follow", "unfollow", "block", "unblock", 
                 "post_update", "post_delete", "quote", "ignore"]
    
    labels = {
        "users": "Users", 
        "total": "Total", 
        "posts": "Posts", 
        "replies": "Replies", 
        "like": "Likes", 
        "unlike": "Unlikes",
        "repost": "Reposts", 
        "unrepost": "Unreposts", 
        "follow": "Follows", 
        "unfollow": "Unfollows",
        "block": "Blocks",
        "unblock": "Unblocks",
        "post_update": "Post Updates",
        "post_delete": "Post Deletes",
        "quote": "Quotes",
        "ignore": "Ignore"
    }
    
    # Calculate totals and averages
    total_stats = {
        stat: sum(stats.get(stat, 0) for _, stats in clusters_stats.items()) 
        for stat in stats_order
    }
    avg_stats = {
        stat: total_stats[stat] / total_clusters if total_clusters > 0 else 0
        for stat in stats_order
    }
    
    # Print markdown table with rows and columns switched
    print("\n## Cluster Statistics")
    print("| Statistic | " + " | ".join([f"Cluster {num}" for num, _ in first_three]) + " | ... | " + 
          " | ".join([f"Cluster {num}" for num, _ in last_three]) + " | Average | Total |")
    print("|-----------|" + "|".join(["----" for _ in range(len(first_three) + len(last_three) + 3)]) + "|")
    
    # Print each statistic as a row
    for stat in stats_order:
        row = f"| {labels[stat]} | "
        # First three clusters
        for _, stats in first_three:
            row += f"{stats.get(stat, 0)} | " # Use .get() for safety
        row += "... | "
        # Last three clusters
        for _, stats in last_three:
            row += f"{stats.get(stat, 0)} | " # Use .get() for safety
        # Average
        row += f"{avg_stats[stat]:.2f} |"
        # Total
        row += f" {total_stats[stat]} |"
        print(row)
    # --- End of Table Generation Logic ---
    
    # Print the top words table
    print("\n## Top Words by Cluster")
    print("| Cluster | Top Words |")
    print("|---------|-----------|")
    for cluster_num in sorted(top_words.keys()):
        words = top_words[cluster_num]
        print(f"| {cluster_num} | {', '.join(words)} |")
    
    # Print medoid posts for each cluster
    print("\n## Cluster Medoid Posts")
    for cluster_num in sorted(cluster_medoids.keys()):
        print(f"\n### Cluster {cluster_num}")
        for i, post in enumerate(cluster_medoids[cluster_num]):
            print(f"#### Medoid {i+1}")
            # Print first 200 chars of the post text with ellipsis if longer
            post_text = post['text']
            if len(post_text) > 200:
                post_text = post_text[:200] + "..."
            print(f"```\n{post_text}\n```")
    
    # Print LDA topics for each cluster
    print("\n## Cluster LDA Topics")
    for cluster_num in sorted(cluster_topics.keys()):
        print(f"\n### Cluster {cluster_num}")
        for i, topic in enumerate(cluster_topics[cluster_num]):
            print(f"#### Topic {i+1}")
            print(", ".join(topic))
