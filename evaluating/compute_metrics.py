import os
import json
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import warnings
import argparse
import getpass
# warnings.filterwarnings("ignore")

# Make sure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

parser = argparse.ArgumentParser(description='Compute evaluation metrics for a model')
parser.add_argument("--subsample", type=int, help="Number of samples to use for evaluation", default=None)
parser.add_argument("--seed", type=int, help="Random seed for reproducibility", default=42)
parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of embeddings", default=False)

args = parser.parse_args()


# Set file paths
USERNAME = getpass.getuser()
DATASET_PATH = os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset")
RESULTS_PATH = os.path.expanduser(f"~/bluesky_blueprint/scratch/model_evaluation/model_evaluation_metrics.json")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
SEED = args.seed
SUBSAMPLE = args.subsample
FORCE_RECOMPUTE = args.force_recompute
CACHE_DIR = os.path.expanduser(f"~/bluesky_blueprint/scratch/caches")


# Set HuggingFace cache path for embeddings model
os.environ["HF_HOME"] = os.path.expanduser(f"~/bluesky_blueprint/scratch/HF-cache")

# 0. LOAD DATA
def filter_text(text: str):
    try:
        obj = json.loads(text)
        actions_taken = obj.get('actions', {})
        filtered_text_content = obj.get('text', "")
        if actions_taken is None:
            actions_taken = {}
        if filtered_text_content is None:
            filtered_text_content = ""

        try:
            filtered_text_content = filtered_text_content.encode('utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            pass

        # Strip any extra whitespace
        filtered_text_content = filtered_text_content.strip()

        # Remove text after double quote if present
        quote_idx = filtered_text_content.find('"')
        if quote_idx != -1:
            filtered_text_content = filtered_text_content[:quote_idx]
        backslash = filtered_text_content.find('\\')
        if backslash != -1:
            filtered_text_content = filtered_text_content[:backslash]

        # Filter out some errors where the model will output a single character
        if len(filtered_text_content) == 1:
            filtered_text_content = ""

        return {"actions": actions_taken, "text": filtered_text_content}
    except json.JSONDecodeError:
        # If text is not a valid JSON string, return it as is, assuming it's plain text.
        # This might happen if a model output is not correctly formatted.
        return {"actions": {}, "text": text.strip()}

def load_cluster_data():
    datasets = {}  # "cluster_n": {"validation": [], "model_0": [], "model_1": [], ...}
    actions_dataset = {}  # "cluster_n": {"validation": [], "model_0": [], "model_1": [], ...}

    # Get all cluster directories from dataset path
    print("Loading cluster data...")
    cluster_dirs = [d for d in os.listdir(DATASET_PATH) if d.startswith('cluster_') and os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    # Load data for each cluster
    for cluster_dir in tqdm(cluster_dirs, desc="Loading clusters"):
        cluster_id = int(cluster_dir.split('_')[1])
        datasets[cluster_id] = {}
        actions_dataset[cluster_id] = {}
        
        cluster_path = os.path.join(DATASET_PATH, cluster_dir)
        jsonl_files = [f for f in os.listdir(cluster_path) if f.endswith('.jsonl')]
        
        ids_to_keep = []
        
        for jsonl_file in jsonl_files:
            # Extract the model name without extension (e.g., "validation", "model_0")
            model_name = os.path.splitext(jsonl_file)[0]
            
            file_path = os.path.join(cluster_path, jsonl_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f.readlines()]
                if "custom_id" in data[0][0]:
                    data = sorted(data, key=lambda x: int(x[0]["custom_id"].split("-")[2]))
                    ids = [int(item[0]["custom_id"].split("-")[2]) for item in data]
                    ids_to_keep = [i for i in range(len(ids)) if ids[i] < 1000]
                    data = [item for item in data if int(item[0]["custom_id"].split("-")[2]) < 1000]
                
                datasets[cluster_id][model_name] = [filter_text(item[-1]["content"])["text"] for item in data]
                actions_dataset[cluster_id][model_name] = [filter_text(item[-1]["content"])["actions"] for item in data]

        datasets[cluster_id]["validation"] = [datasets[cluster_id]["validation"][i] for i in ids_to_keep]
        actions_dataset[cluster_id]["validation"] = [actions_dataset[cluster_id]["validation"][i] for i in ids_to_keep]

        # Subsample data if needed
        if SUBSAMPLE is not None:
            ids_to_keep = random.sample(range(len(datasets[cluster_id]["validation"])), SUBSAMPLE)
            for model_name in datasets[cluster_id]:
                datasets[cluster_id][model_name] = [datasets[cluster_id][model_name][i] for i in ids_to_keep]
                actions_dataset[cluster_id][model_name] = [actions_dataset[cluster_id][model_name][i] for i in ids_to_keep]

    return datasets, actions_dataset

# 1. LINGUISTIC STYLE METRICS

# 1.1 Readability and Complexity
# print("Calculating readability metrics...")

# def get_readability_metrics(tweets):
#     flesch_scores = []
#     word_counts = []
#     avg_word_lengths = []

#     for tweet in tweets:
#         # Flesch Reading Ease
#         try:
#             flesch = flesch_reading_ease(tweet)
#             flesch_scores.append(flesch)
#         except Exception as e:
#             # If calculation fails, use neutral score
#             flesch_scores.append(50)
#             print(f"Error calculating Flesch score: {e}")

#         # Word count and average word length
#         words = tweet.split()
#         word_counts.append(len(words))
#         if words:
#             avg_word_length = sum(len(word) for word in words) / len(words)
#             avg_word_lengths.append(avg_word_length)
#         else:
#             avg_word_lengths.append(0)

#     return {
#         "flesch_reading_ease": np.mean(flesch_scores),
#         "avg_word_count": np.mean(word_counts),
#         "avg_word_length": np.mean(avg_word_lengths)
#     }


# real_readability = get_readability_metrics(sampled_real_tweets)
# generated_readability = get_readability_metrics(sampled_generated_tweets)


# 2. SEMANTIC SIMILARITY USING EMBEDDINGS

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Convert tweets to embeddings
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    # Ensure all texts are strings
    processed_texts = []
    for text in texts:
        # Convert None to empty string
        if text is None:
            processed_texts.append("")
        # Convert any non-string types to string
        elif not isinstance(text, str):
            try:
                processed_texts.append(str(text))
            except:
                processed_texts.append("")
        else:
            processed_texts.append(text)
    
    for i in tqdm(range(0, len(processed_texts), batch_size)):
        batch_texts = processed_texts[i:i+batch_size]
        # Replace or remove surrogates before printing to avoid UnicodeEncodeError
        cleaned_texts = [text.encode('utf-8', errors='ignore').decode('utf-8') for text in batch_texts]
        encoded_input = tokenizer(
            cleaned_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        # Move input tensors to the same device as the model
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask']).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Get embeddings for all texts across clusters and models
def compute_all_embeddings(datasets, use_cache=True, force_recompute=False):
    print("Computing embeddings...")
    
    # Create cache filename with SEED and SUBSAMPLE to invalidate cache when parameters change
    cache_filename = f"evaluation_embeddings_cache_seed_{SEED}"
    if SUBSAMPLE:
        cache_filename += f"_subsample_{SUBSAMPLE}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_filename}.pt")
    
    if use_cache and not force_recompute:
        print("Using cache for embeddings")
        # Create a cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading embeddings from cache: {cache_file}")
            try:
                embeddings = torch.load(cache_file, weights_only=False)
                # Verify cache has all needed cluster IDs and models
                valid_cache = True
                for cluster_id in datasets.keys():
                    if cluster_id not in embeddings:
                        valid_cache = False
                        break
                    for model_id in datasets[cluster_id].keys():
                        if model_id not in embeddings[cluster_id]:
                            valid_cache = False
                            break
                
                if valid_cache:
                    return embeddings
                else:
                    print("Cache missing some data, recomputing all embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing embeddings.")
        else:
            print("Cache file not found, recomputing embeddings.")
    
    # Load pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = {}
    for cluster_id, cluster_data in tqdm(datasets.items(), desc="Computing cluster embeddings"):
        embeddings[cluster_id] = {}
        for model_id, texts in cluster_data.items():
            embeddings[cluster_id][model_id] = get_embeddings(texts, model, tokenizer, device)
    
    # Save to cache
    if use_cache:
        print(f"Saving embeddings to cache: {cache_file}")
        torch.save(embeddings, cache_file)
    
    return embeddings


# 2.1. Cosine similarity between average embeddings
def avg_embed_cosine_sim(real_embeddings, generated_embeddings):
    avg_real_embedding = np.mean(real_embeddings, axis=0).reshape(1, -1)
    avg_generated_embedding = np.mean(generated_embeddings, axis=0).reshape(1, -1)
    avg_embedding_similarity = cosine_similarity(avg_real_embedding, avg_generated_embedding)[0][0]
    return avg_embedding_similarity

# 2.2. Maximum cosine similarity between pairs of embeddings
def max_cosine_sim(real_embeddings, generated_embeddings):
    semantic_sims = []
    for i in range(len(generated_embeddings)):
        gen_emb = generated_embeddings[i].reshape(1, -1)
        sims = cosine_similarity(gen_emb, real_embeddings).flatten()
        semantic_sims.append(np.max(sims))

    max_pair_similarity = np.max(semantic_sims)
    return max_pair_similarity


# 3. TOPIC DISTRIBUTION ANALYSIS

def tf_idf_jaccard(datasets):
    print("Calculating TF-IDF Jaccard similarity...")
    
    # Combine all texts from each cluster into one big text
    combined = {}
    all_texts = [] # Create list of all texts for fitting the vectorizer
    
    for cluster_id, data in datasets.items():
        combined[cluster_id] = {}
        for model_id, texts in data.items():
            combined[cluster_id][model_id] = " ".join(texts)
            all_texts.append(combined[cluster_id][model_id])
    
    # Fit TF-IDF vectorizer on all texts
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    tfidf_vectorizer.fit(all_texts)
    
    # Get TF-IDF vectors for each dataset
    tf_idf_vectors = {}
    
    for cluster_id in datasets.keys():
        tf_idf_vectors[cluster_id] = {}
        for model_id in datasets[cluster_id].keys():
            tf_idf_vectors[cluster_id][model_id] = tfidf_vectorizer.transform([combined[cluster_id][model_id]])

    # Get top N terms for each dataset
    n = 100  # Number of top terms to consider
    feature_names = tfidf_vectorizer.get_feature_names_out()

    top_terms = {}
    top_5_terms = {}
    
    for cluster_id in datasets.keys():
        top_terms[cluster_id] = {}
        top_5_terms[cluster_id] = {}
        for model_id in datasets[cluster_id].keys():
            # Get top terms for real data
            real_tfidf = tf_idf_vectors[cluster_id][model_id].toarray().flatten()
            real_sorted = real_tfidf.argsort()
            top_indices = real_sorted[-n:][::-1]
            top_terms[cluster_id][model_id] = set([feature_names[idx] for idx in top_indices])
            
            top_5_indices = real_sorted[-5:][::-1]
            top_5_terms[cluster_id][model_id] = [feature_names[idx] for idx in top_5_indices]

    # Calculate Jaccard similarity of top terms against validation set
    jaccard_results = {}

    for cluster_id in datasets.keys():
        jaccard_results[cluster_id] = {}
        validation_top_terms = top_terms[cluster_id]["validation"]
        for model_id in datasets[cluster_id].keys():
            if model_id == "validation":
                continue
            
            gen_top_terms = top_terms[cluster_id][model_id]
            intersection = len(validation_top_terms.intersection(gen_top_terms))
            union = len(validation_top_terms.union(gen_top_terms))
            jaccard_sim = intersection / union if union > 0 else 0
            
            jaccard_results[cluster_id][model_id] = jaccard_sim
    
    return jaccard_results, top_5_terms


# Calculate Jensen-Shannon divergence between distributions
def js_divergence(p, q):
    # Convert to probability distributions
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    # Get union of keys
    all_keys = set(p.keys()).union(set(q.keys()))

    # Normalize and calculate divergence
    p_norm = {k: p.get(k, 0)/p_sum for k in all_keys}
    q_norm = {k: q.get(k, 0)/q_sum for k in all_keys}

    # Calculate midpoint distribution
    m = {k: (p_norm[k] + q_norm[k])/2 for k in all_keys}

    # Calculate divergence
    kl_p_m = sum(p_norm[k] * np.log2(p_norm[k]/m[k])
                 for k in all_keys if p_norm[k] > 0)
    kl_q_m = sum(q_norm[k] * np.log2(q_norm[k]/m[k])
                 for k in all_keys if q_norm[k] > 0)

    return (kl_p_m + kl_q_m) / 2

def embedding_js_divergence(real_embeddings, generated_embeddings):
    """Compute JS divergence between embedding distributions"""
    # Compute distance matrix between all pairs
    combined_embeddings = np.vstack([real_embeddings, generated_embeddings])
    dist_matrix = 1 - cosine_similarity(combined_embeddings)
    
    # Calculate histograms for real and generated distributions
    n_real = len(real_embeddings)
    n_gen = len(generated_embeddings)
    
    # Compute pairwise distances within each set
    real_dists = dist_matrix[:n_real, :n_real].flatten()
    gen_dists = dist_matrix[n_real:, n_real:].flatten()
    
    # Create histograms
    bins = np.linspace(0, 2, 50)
    real_hist, _ = np.histogram(real_dists, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_dists, bins=bins, density=True)
    
    # Convert to dictionaries for js_divergence function
    real_dist = {i: v for i, v in enumerate(real_hist) if v > 0}
    gen_dist = {i: v for i, v in enumerate(gen_hist) if v > 0}
    
    return js_divergence(real_dist, gen_dist)


# 4. ACTION STATISTICS

# Compute real vs generated action label vectors where only the last action in
# each chain changes between real vs. generated
action_label_map = {"post": 0, "reply": 1, "like": 2, "repost": 3, "follow": 4, "ignore": 5}

def get_label_vector(data):
    label_vector = []
    for action in data:
        try:
            action = json.loads(action)
        except TypeError:
            pass
        if len(action) == 0:
            label_vector.append(action_label_map["post"])
        elif action["like"]:
            label_vector.append(action_label_map["like"])
        elif action["repost"]:
            label_vector.append(action_label_map["repost"])
        elif action["follow"]:
            label_vector.append(action_label_map["follow"])
        elif action["ignore"]:
            label_vector.append(action_label_map["ignore"])
        else:
            label_vector.append(action_label_map["post"])
        # Can't really differentiate between posts and replies at this stage. Should not matter

    return label_vector

def get_f1(real, generated):
    # Get label vectors
    real_labels = get_label_vector(real)
    gen_labels = get_label_vector(generated)
    
    # Compute F1 score
    tp = sum([1 for r, g in zip(real_labels, gen_labels) if r == g])
    fp = sum([1 for r, g in zip(real_labels, gen_labels) if r != g])
    fn = sum([1 for r, g in zip(real_labels, gen_labels) if r != g])
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

# 4. MODEL QUALITY SCORE
# Calculate an overall quality score for the model's ability to mimic real tweets
# Save results to json
# print("Calculating overall model quality score...")

# TODO

# Compute diagonality for each metric
def compute_diagonality(matrix):
    total_sum = 0
    diagonal_sum = 0
    
    # Get all unique cluster IDs and model IDs
    cluster_ids = list(matrix.keys())
    
    for cluster_id in cluster_ids:
        for model_id, value in matrix[cluster_id].items():
            if model_id == "validation":
                continue
                
            # Extract model number from model_id (e.g., "model_1" -> 1)
            try:
                model_num = int(model_id.split('_')[1])
                total_sum += value
                
                # Check if this is a diagonal element (cluster_id == model_num)
                if cluster_id == model_num:
                    diagonal_sum += value
            except:
                continue
    
    # Compute diagonality measure
    if total_sum == 0:
        return 0
    
    off_diagonal_sum = total_sum - diagonal_sum
    diagonality = 1 - off_diagonal_sum / total_sum
    
    return diagonality


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Load the real and generated dataset for each cluster
    datasets, actions_dataset = load_cluster_data()

    # Compute all embeddings
    embeddings = compute_all_embeddings(datasets, use_cache=True, force_recompute=FORCE_RECOMPUTE)
    
    # Compute all metrics for each real-generated pair
    
    jaccard_results, top_5_terms = tf_idf_jaccard(datasets)
    print(jaccard_results)
    # Print top 5 terms cleanly
    print("\nTop 5 Terms by Cluster and Model:")
    for cluster_id, cluster_data in top_5_terms.items():
        print(f"\nCluster {cluster_id}:")
        for model_id, terms in cluster_data.items():
            print(f"  {model_id}: {', '.join(terms)}")
    
    keys = datasets.keys()
    
    avg_embed_cosine_sims = {}
    max_cosine_sims = {}
    embedding_js_divergences = {}
    f1_scores = {}
    
    for cluster_id, data in datasets.items():
        validation = data["validation"]
        val_embeddings = embeddings[cluster_id]["validation"]
        
        avg_embed_cosine_sims[cluster_id] = {}
        max_cosine_sims[cluster_id] = {}
        embedding_js_divergences[cluster_id] = {}
        f1_scores[cluster_id] = {}
        
        for model_id, texts in data.items():
            
            if model_id == "validation":
                continue

            generated = texts
            generated_embeddings = embeddings[cluster_id][model_id]
            
            avg_embed_cosine_sims[cluster_id][model_id] = avg_embed_cosine_sim(val_embeddings, generated_embeddings)
            max_cosine_sims[cluster_id][model_id] = max_cosine_sim(val_embeddings, generated_embeddings)
            embedding_js_divergences[cluster_id][model_id] = embedding_js_divergence(val_embeddings, generated_embeddings)
            
            f1_scores[cluster_id][model_id] = get_f1(actions_dataset[cluster_id]["validation"], actions_dataset[cluster_id][model_id])

    # Calculate diagonality for each applicable metric
    diagonality_scores = {}

    metrics_to_check = {
        "jaccard_similarity": jaccard_results,
        "avg_embed_cosine_similarity": avg_embed_cosine_sims,
        "max_cosine_similarity": max_cosine_sims,
        "f1_scores": f1_scores
    }

    # For JS divergence, lower is better so we need to invert the values
    inverted_js_divergences = {}
    for cluster_id in embedding_js_divergences:
        inverted_js_divergences[cluster_id] = {}
        for model_id, value in embedding_js_divergences[cluster_id].items():
            # Invert the value (use 1/(1+value) to ensure positive values)
            inverted_js_divergences[cluster_id][model_id] = 1/(1+value)

    metrics_to_check["inverted_embedding_js_divergence"] = inverted_js_divergences

    for metric_name, metric_data in metrics_to_check.items():
        diagonality_scores[metric_name] = compute_diagonality(metric_data)

    print("\n===== DIAGONALITY SCORES =====")
    for metric_name, score in diagonality_scores.items():
        print(f"{metric_name}: {score:.4f}")
    
    # Prepare results dictionary with all metrics
    all_metrics = {
        "jaccard_similarity": jaccard_results,
        "avg_embed_cosine_similarity": avg_embed_cosine_sims,
        "max_cosine_similarity": max_cosine_sims,
        "embedding_js_divergence": embedding_js_divergences,
        "f1_scores": f1_scores
    }

    # Pretty print all stats
    print("\n===== EVALUATION METRICS =====")
    for metric_name, metric_data in all_metrics.items():
        print(f"\n{metric_name.upper()}:")
        for cluster_id, cluster_results in metric_data.items():
            print(f"  Cluster {cluster_id}:")
            for model_id, score in cluster_results.items():
                if model_id != "validation":
                    print(f"    {model_id}: {score:.4f}")
    
    # Calculate averages per model across all clusters for each metric
    print("\n===== AVERAGE METRICS BY MODEL =====")

    # Get all model IDs
    all_model_ids = set()
    for cluster_id, cluster_data in datasets.items():
        for model_id in cluster_data.keys():
            if model_id != "validation":
                all_model_ids.add(model_id)

    model_averages = {model_id: {} for model_id in all_model_ids}

    # Calculate average for each metric and model
    for metric_name, metric_data in all_metrics.items():
        for model_id in all_model_ids:
            values = []
            for cluster_id, cluster_results in metric_data.items():
                if model_id in cluster_results:
                    values.append(cluster_results[model_id])
            
            if values:
                model_averages[model_id][metric_name] = np.mean(values)

    # Print the averages by model
    for model_id in sorted(all_model_ids):
        print(f"\nModel: {model_id}")
        for metric_name, value in model_averages[model_id].items():
            print(f"  {metric_name}: {value:.4f}")

    # Save results to json
    
    # Add diagonality scores to the results
    all_metrics["diagonality_scores"] = diagonality_scores

    # Create directory for results file if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    # Convert NumPy values to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    # Save all metrics to JSON file
    print(f"\nSaving metrics to {RESULTS_PATH}")
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(all_metrics), f, indent=2)

    print(f"Evaluation metrics saved successfully to {RESULTS_PATH}")