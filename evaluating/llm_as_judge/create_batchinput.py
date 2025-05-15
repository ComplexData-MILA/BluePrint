import json
import os
import random
import getpass

USERNAME = getpass.getuser()
FILENAME = "batchinput.jsonl"
DATASET_PATH = f"/scratch/{USERNAME}/test_evaluation_metrics"
TWEETS_PER_MODEL = 100

SYSTEM_PROMPT = """
You are in a Turing-test-like setting.
One of the following messages comes from a human on social media, and the other comes from a language model.
Please identify which message you think is from the AI. Provide a brief explanation before making your choice.
You will output a json object containing the following information:

```
{
    reasoning: string, // Reason for your choice
    label: string // "A" or "B". The message you think is from the AI
}
```"""

USER_PROMPT ="""
MESSAGE A:

```
{message_a}
```

MESSAGE B:

```
{message_b}
```"""

SYSTEM_PROMPT_WITH_CONTEXT = """
You are in a Turing-test-like setting.
You will be provided a real conversation (CONTEXT), and then two possible messages continuing that conversation.
One of the messages comes from a human on social media, and the other comes from a language model.
Please identify which message you think is from the AI. Provide a brief explanation before making your choice.
You will output a json object containing the following information:

```
{
    reasoning: string, // Reason for your choice
    label: string // "A" or "B". The message you think is from the AI
}
```"""

USER_PROMPT_WITH_CONTEXT ="""
CONTEXT:

```
{context}
```

MESSAGE A:

```
{message_a}
```

MESSAGE B:

```
{message_b}
```"""

# Load all tweet data at startup
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
            
            # Load validation data
            validation_path = os.path.join(cluster_path, 'validation.jsonl')
            model_path = os.path.join(cluster_path, f'model_{cluster_id}.jsonl')
            
            print(f"Looking for validation file: {validation_path}")
            print(f"Looking for model file: {model_path}")
            
            if not os.path.exists(validation_path) or not os.path.exists(model_path):
                print(f"Missing files for cluster {cluster_id}. Validation exists: {os.path.exists(validation_path)}. Model exists: {os.path.exists(model_path)}")
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
                        
                        # Skip chains where the last tweet is an action
                        if isinstance(tweet_chain, list) and tweet_chain:
                            last_tweet = tweet_chain[-1]['text']
                            if last_tweet.startswith('[action:'):
                                # print(f"Skipping line {line_num} due to action tweet: {last_tweet}")
                                continue
                            validation_data.append(tweet_chain)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in validation file, line {line_num}: {e}")
                        continue
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Error processing validation file, line {line_num}: {e}")
                        continue
            
            print(f"Loaded {len(validation_data)} valid chains from validation file")
            
            model_data = []
            print(f"Reading model data from {model_path}")
            with open(model_path, 'r') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        tweet_chain = json.loads(line)
                        
                        # Skip chains where the last tweet is an action
                        if isinstance(tweet_chain, list) and tweet_chain:
                            last_tweet = tweet_chain[-1]['text']
                            if last_tweet.startswith('[action:'):
                                # print(f"Skipping line {line_num} due to action tweet: {last_tweet}")
                                continue
                            model_data.append(tweet_chain)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in model file, line {line_num}: {e}")
                        continue
                    except (KeyError, IndexError, TypeError) as e:
                        print(f"Error processing model file, line {line_num}: {e}")
                        continue
                        
            print(f"Loaded {len(model_data)} valid chains from model file")
        
            # Only keep pairs where both validation and model have data
            min_len = min(len(validation_data), len(model_data))
            validation_data = validation_data[:min_len]
            model_data = model_data[:min_len]
            
            print(f"Final dataset for cluster {cluster_id}: {min_len} tweet chains")
            
            data[cluster_id] = {
                'validation': validation_data,
                'model': model_data
            }
    
    except Exception as e:
        print(f"Error loading tweet data: {e}")
        return {}
        
    print(f"Successfully loaded data for {len(data)} clusters")
    return data


# Creating an array of json tasks

def data_into_tasks(data):

    tasks = []
    solutions = {}

    for index, cluster in data.items():
        
        indices = random.sample(range(len(cluster['validation'])), k=min(TWEETS_PER_MODEL, len(cluster['validation'])))
        
        for tweet_idx in indices:
        
            real_tweet_chain = cluster['validation'][tweet_idx]
            ai_tweet_chain = cluster['model'][tweet_idx]
            
            # Only the last message differs - the rest is context
            context = real_tweet_chain[:-1] if len(real_tweet_chain) > 1 else None
            real_tweet = real_tweet_chain[-1]
            ai_tweet = ai_tweet_chain[-1]
            custom_id = f"task-{index}-{tweet_idx}"

            user_ids = {}
            for tweet in real_tweet_chain:
                if tweet['user_id'] not in user_ids:
                    user_ids[tweet['user_id']] = len(user_ids)

            if random.random() < 0.5:
                message_a = f"User{user_ids[ai_tweet['user_id']]}:\n{ai_tweet['text']}"
                message_b = f"User{user_ids[real_tweet['user_id']]}:\n{real_tweet['text']}"
                solutions[custom_id] = {"ai-tweet": "A", "cluster_id": index, "tweet_id": tweet_idx}
            else:
                message_a = f"User{user_ids[real_tweet['user_id']]}:\n{real_tweet['text']}"
                message_b = f"User{user_ids[ai_tweet['user_id']]}:\n{ai_tweet['text']}"
                solutions[custom_id] = {"ai-tweet": "B", "cluster_id": index, "tweet_id": tweet_idx}
            
            if context is not None:
                system_prompt = SYSTEM_PROMPT_WITH_CONTEXT
                user_prompt = USER_PROMPT_WITH_CONTEXT
                context_text = "\n\n".join([f"User{user_ids[tweet['user_id']]}:\n{tweet['text']}" for tweet in context])
                user_prompt = user_prompt.format(context = context_text, message_a = message_a, message_b = message_b)
            else:
                system_prompt = SYSTEM_PROMPT
                user_prompt = USER_PROMPT
                user_prompt = user_prompt.format(message_a = message_a, message_b = message_b)
            
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1, # Low temp for reliable classification
                    "response_format": { 
                        "type": "json_object"
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                }
            }
            
            tasks.append(task)
    
    return tasks, solutions

def save_tasks(tasks):
    with open(FILENAME, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

def save_solutions(solutions):
    with open("solutions.json", 'w') as f:
        f.write(json.dumps(solutions))
            
if __name__ == "__main__":
    data = load_tweet_data()
    tasks, solutions = data_into_tasks(data)
    save_tasks(tasks)
    save_solutions(solutions)
    print(f"Saved {len(tasks)} tasks to {FILENAME}")