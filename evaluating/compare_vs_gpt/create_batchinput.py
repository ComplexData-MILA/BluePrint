import json
import os
import random
import getpass
import toml

USERNAME = getpass.getuser()
FILENAME = "batchinput.jsonl"
SOLUTIONS_FILENAME = "solutions.json"
DATASET_PATH = f"/scratch/{USERNAME}/evaluation_dataset"
CHAINS_PER_CLUSTER = 200
PROMPTS_PATH = f"/home/{USERNAME}/SM-based-personas/shared/prompts.toml"

def load_prompts():
    # System prompt as used during training
    with open(PROMPTS_PATH, "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["SYSTEM_PROMPT"]
        reply_schema = prompts["CHATGPT_REPLY_SCHEMA"]
        post_schema = prompts["CHATGPT_POST_SCHEMA"]
    reply_schema = json.loads(reply_schema)
    post_schema = json.loads(post_schema)
    return system_prompt, reply_schema, post_schema

_, REPLY_SCHEMA, POST_SCHEMA = load_prompts()

# Load only validation tweet data
def load_tweet_data():
    data = {}
    base_path = DATASET_PATH

    print(f"Loading tweet data from {base_path}")

    try:
        # Get all cluster directories
        for cluster_dir in os.listdir(base_path):
            if not cluster_dir.startswith('cluster_'):
                continue

            cluster_id = int(cluster_dir.split('_')[1])
            cluster_path = os.path.join(base_path, cluster_dir)

            print(f"Processing cluster {cluster_id} at {cluster_path}")

            # Load validation data only
            validation_path = os.path.join(cluster_path, 'validation.jsonl')

            print(f"Looking for validation file: {validation_path}")

            if not os.path.exists(validation_path):
                print(f"Missing validation file for cluster {cluster_id}.")
                continue

            validation_data = []
            print(f"Reading validation data from {validation_path}")
            with open(validation_path, 'r') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        tweet_chain = json.loads(line)
                        validation_data.append(tweet_chain)

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in validation file, line {line_num}: {e}")
                        continue
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Error processing validation file, line {line_num}: {e}")
                        continue

            print(f"Loaded {len(validation_data)} valid chains from validation file")

            if validation_data:
                data[cluster_id] = validation_data # Store only validation data

    except Exception as e:
        print(f"Error loading tweet data: {e}")
        return {}

    print(f"Successfully loaded data for {len(data)} clusters")
    return data

def pick_indices(chains):
    eligible_A_indices = []
    other_indices = []

    # Categorize chain indices based on the criteria
    for chain_index, tweet_chain_in_cluster in enumerate(chains):
        is_A_candidate = False
        if 0 <= chain_index < 1000:
            # Assuming tweet_chain_in_cluster is non-empty and its elements are dicts,
            # as per data loading and subsequent usage in the original code.
            if tweet_chain_in_cluster: # Ensure chain is not empty
                last_message = tweet_chain_in_cluster[-1]
                if isinstance(last_message, dict): # Ensure last message is a dictionary
                    content_of_last_message = last_message.get('content')
                    
                    if isinstance(content_of_last_message, str):
                        try:
                            # The problem states: "last element loaded as a json. This json has a 'text' attribute"
                            # This implies content_of_last_message is a JSON string.
                            parsed_content = json.loads(content_of_last_message)
                            if isinstance(parsed_content, dict) and parsed_content.get('text') is not None:
                                is_A_candidate = True
                        except json.JSONDecodeError:
                            # Content is not a valid JSON string or not the expected structure
                            pass
        
        if is_A_candidate:
            eligible_A_indices.append(chain_index)
        else:
            other_indices.append(chain_index)

    # Shuffle within categories for random selection
    random.shuffle(eligible_A_indices)
    random.shuffle(other_indices)

    k = min(CHAINS_PER_CLUSTER, len(chains))
    indices = [] # Default to empty list if k is 0

    if k > 0:
        n_A = len(eligible_A_indices)
        n_B = len(other_indices)
        
        # Target at least half of samples from group A (eligible_A_indices)
        # (k + 1) // 2 calculates ceiling(k/2)
        target_A_samples = (k + 1) // 2

        # Determine actual number of samples from group A
        # Try to meet target_A_samples, but don't exceed available n_A
        num_from_A = min(n_A, target_A_samples)
        
        # Determine number of samples needed from group B to make up k total samples
        num_from_B = k - num_from_A

        # If we need more from group B than available, we must adjust.
        # This means we'll take all available from group B,
        # and the remainder must come from group A.
        if num_from_B > n_B:
            num_from_B = n_B # Take all available from B
            num_from_A = k - num_from_B # The rest must come from A
                            # This num_from_A is guaranteed to be <= n_A because k <= n_A + n_B
        
        # Select the determined number of indices from the shuffled lists
        sampled_A_indices = eligible_A_indices[:num_from_A]
        sampled_B_indices = other_indices[:num_from_B]
        
        indices = sampled_A_indices + sampled_B_indices
        
        # Shuffle the final combined list of indices
        random.shuffle(indices)
    
    return indices

# Creating an array of json tasks for generation
def data_into_tasks(data):

    tasks = []
    solutions = {} # Will store ground truth
    random.seed(42) # For reproducibility

    for cluster_id, chains in data.items():

        indices = pick_indices(chains)

        for chain_idx in indices:

            tweet_chain = chains[chain_idx]

            # The chain excluding the last message is the prompt
            prompt_chain = tweet_chain[:-1]
            ground_truth_tweet = tweet_chain[-1] # The message to predict
            custom_id = f"task-{cluster_id}-{chain_idx}"

            # Map user IDs consistently within the chain
            formatted_messages = []
            for tweet in prompt_chain:
                formatted_messages.append({
                    "role": tweet['role'] if tweet['role'] in ['user', 'assistant', 'system'] else "user",
                    "content": f"{tweet['role']}:\n{tweet['content']}"
                })

            # Store the ground truth message for evaluation
            solutions[custom_id] = {
                "ground_truth": f"{ground_truth_tweet['role']}:\n{ground_truth_tweet['content']}",
                "cluster_id": cluster_id,
                "chain_index": chain_idx # Use chain_idx instead of tweet_idx
            }

            # Skip if no messages left after removing the last one
            if not formatted_messages:
                print(f"Skipping task {custom_id} as it has no prompt messages after removing the last one.")
                del solutions[custom_id] # Remove corresponding solution entry
                continue

            # Determine which schema to use based on the number of messages
            if len(formatted_messages) > 1: # More than just the system prompt means it's a reply
                schema = REPLY_SCHEMA
                schema_name = "reply_schema"
            else: # Only one message (system prompt) means it's a new post
                schema = POST_SCHEMA
                schema_name = "post_schema"
            
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema
                }
            }

            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-mini",
                    "temperature": 0.7, # Adjust temperature for generation
                    "response_format": response_format, # Use the determined schema
                    "messages": formatted_messages, # Use the formatted chain as messages
                }
            }

            tasks.append(task)

    return tasks, solutions

def save_tasks(tasks):
    with open(FILENAME, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n') # Write each task on a new line

def save_solutions(solutions):
    with open(SOLUTIONS_FILENAME, 'w') as f: # Use renamed solutions file
        f.write(json.dumps(solutions, indent=2)) # Added indent for readability

if __name__ == "__main__":
    # Check if schemas were loaded successfully
    if not REPLY_SCHEMA or not POST_SCHEMA:
        print("Exiting due to errors loading schemas.")
    else:
        data = load_tweet_data()
        if data:
            tasks, solutions = data_into_tasks(data)
            save_tasks(tasks)
            save_solutions(solutions)
            print(f"Saved {len(tasks)} generation tasks to {FILENAME}")
            print(f"Saved {len(solutions)} ground truth solutions to {SOLUTIONS_FILENAME}")
        else:
            print("No data loaded, exiting.")
        