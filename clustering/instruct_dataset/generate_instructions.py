\
import os
import json
import getpass
import random
import argparse
from pathlib import Path
import glob
import math
# import openai # Uncomment if using OpenAI API

# --- Configuration ---
USERNAME = getpass.getuser()
DEFAULT_INPUT_DIR = f"/scratch/{USERNAME}/pii_removed/processed_25_clusters_hashed"
DEFAULT_OUTPUT_DIR = f"/scratch/{USERNAME}/instruction_following"
DEFAULT_DOLLY_PATH = f"/scratch/{USERNAME}/databricks-dolly-15k.jsonl"
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Uncomment and set env var if using OpenAI
# if OPENAI_API_KEY:
#     openai.api_key = OPENAI_API_KEY

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate instruction-following dataset from social media clusters.')
parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing cluster*.jsonl files.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save generated instruction files.")
parser.add_argument("--dolly_path", type=str, default=DEFAULT_DOLLY_PATH, help="Path to databricks-dolly-15k.jsonl.")
parser.add_argument("--max_chains_per_cluster", type=int, default=10000, help="Maximum number of chains to process per cluster file.")
parser.add_argument("--dolly_mix_fraction", type=float, default=0.2, help="Fraction of Dolly instructions to mix in.")
parser.add_argument("--openai_fraction", type=float, default=0.0, help="Fraction of prompts to generate requiring OpenAI (summarization, paraphrase).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
args = parser.parse_args()

random.seed(args.seed)

# --- Helper Functions ---

def format_chain(chain, max_tweets=10):
    """Formats a chain of tweets into a readable string."""
    formatted = []
    user_ids = {}
    for i, tweet in enumerate(chain[-max_tweets:]): # Show only last N tweets for context
        user_id = tweet.get('user_id', 'unknown_user')
        if user_id not in user_ids:
            user_ids[user_id] = f"User_{len(user_ids)}"
        
        msg = {}
        msg['user_id'] = user_ids[user_id]
        if 'text' in tweet:
            msg['text'] = tweet['text']
        if 'actions' in tweet:
            msg['actions'] = tweet['actions']

        formatted.append(json.dumps(msg))
    return "\n".join(formatted)

def extract_text(tweet):
    """Extracts the main text content from a tweet dictionary."""
    msg = {}
    if 'text' in tweet:
        msg['text'] = tweet['text']
    if 'actions' in tweet:
        msg['actions'] = tweet['actions']

    return json.dumps(msg)

def build_user_histories(chains):
    """Groups chains by the user_id of the last tweet."""
    user_histories = {}
    for i, chain in enumerate(chains):
        if not chain: continue
        last_tweet = chain[-1]
        user_id = last_tweet.get('user_id')
        timestamp = last_tweet.get('unix_epoch', i) # Use index as fallback timestamp
        if user_id:
            if user_id not in user_histories:
                user_histories[user_id] = []
            user_histories[user_id].append({"chain": chain, "timestamp": timestamp})
    
    # Sort histories by timestamp
    for user_id in user_histories:
        user_histories[user_id].sort(key=lambda x: x["timestamp"])
    return user_histories

def get_user_history(user_histories, user_id, max_chains=20, max_tweets_per_chain=5):
    """Retrieves a user's history and formats it."""
    if user_id not in user_histories:
        return None
    user_chains = user_histories[user_id]
    # Sort by timestamp
    user_chains.sort(key=lambda x: x["timestamp"])
    
    # Take the most recent chains
    user_chains = user_chains[-max_chains:]
    for entry in user_chains:
        entry["chain"] = entry["chain"][-(max_tweets_per_chain+1):]
        if "text" not in entry["chain"][-1]:
            entry["chain"] = entry["chain"][:-1]
    return user_chains

def format_user_history(user_chains):
    """Formats a user's history into a readable string."""
    formatted = []
    for entry in user_chains:
        chain = entry["chain"]
        formatted.append(format_chain(chain))
    return "\n".join(formatted)

# --- Instruction Generation Functions ---

needle_haystack_instructions = [
    "Given the following conversation history:\n{history}\nWhat was the {n}{th} message?",
    "What’s the {n}{th} message in this conversation: {history}?",
    "Out of the messages in {history}, which one is number {n}?",
    "Identify the {n}{th} entry in this chat history: {history}.",
    "Retrieve the {n}{th} message from the dialogue below.\n\n{history}",
    "Which message was sent {n}{th} in this sequence?\n\n{history}",
    "In the conversation below, what was the {n}{th} message?\n\n{history}",
    "{history}\n\nCan you find the {n}{th} message in this chat?",
    "What was the {n}{th} message in this conversation?\n\n{history}",
]

def generate_needle_haystack(chain, user_histories):
    """Generates a 'find the n-th message' instruction."""
    user_id = chain[-1].get('user_id')
    if len(user_histories[user_id]) < 2:
        return None
    
    history = get_user_history(user_histories, user_id)
    flat_history = [msg for entry in history for msg in entry["chain"] ]
    n = random.randint(1, len(flat_history)) # Random n-th message
    nth_message_content = extract_text(flat_history[n-1])
    if not nth_message_content:
        return None
    
    history_str = format_user_history(history)
    th = "st" if n == 1 else "nd" if n == 2 else "rd" if n == 3 else "th"
    instruction = random.choice(needle_haystack_instructions).format(history=history_str, n=n, th=th)
    response = nth_message_content
    return [
        {"role": "system", "content": instruction},
        {"role": "assistant", "content": response}
    ]

summarization_instructions = [
    "Summarize the following conversation:\n{Conversation}",
    "Can you provide a summary of this conversation: {Conversation}?",
    "What’s a brief overview of the discussion below?\n\n{Conversation}",
    "Condense this chat into a short summary.\n\n{Conversation}",
    "{Conversation}\n\nSummarize the key points from this exchange.",
    "What happened in the following conversation?\n\n{Conversation}",
    "Summarize the main ideas from this chat:\n\n{Conversation}",
    "Can you give a brief summary of this conversation?\n\n{Conversation}",
    "What are the main takeaways from this discussion?\n\n{Conversation}",
]

def generate_summarization_prompt(chain, user_histories):
    """Generates a prompt for summarization (requires external LLM for response)."""
    if len(chain) < 3: # Need a few messages to summarize
        return None
    context_str = format_chain(chain)
    instruction = random.choice(summarization_instructions).format(Conversation=context_str)
    # Response needs to be generated by a strong LLM (e.g., OpenAI)
    # Placeholder: return {"instruction": instruction, "context": "", "response": "[LLM-Generated Summary]"}
    # For OpenAI Batch API, you'd collect these instructions and send them off.
    return [
        {"role": "system", "content": instruction},
        {"role": "assistant", "content": "[LLM-Generated Summary]"}
    ]

topic_instructions = [
    "Generate a social media post about the following topic: {topic}",
    "Write a post suitable for social media discussing: {topic}.",
    "Create a tweet (or Instagram caption, etc.) that centers around: {topic}.",
    "Craft a short social update related to: {topic}.",
    "Can you come up with a social media post idea about {topic}?",
    "Compose a shareable message on {topic} for social media.",
    "Generate a social media post that talks about: {topic}.",
    "What would you say in a post about {topic}?",
    "Draft a social media message that includes: {topic}.",
    "How would you write a post about {topic}?",
    "Can you create a social media post that mentions: {topic}?",
    "Write a social media post that discusses: {topic}.",
    "Generate a tweet about: {topic}.",
]

def generate_topic_generation(chain, user_histories):
    """Generates a 'write about topic X' instruction using a real post as response."""
    if not chain: return None
    # Find a tweet with some text to extract a pseudo-topic
    target_tweet = chain[-1]
    if 'text' not in target_tweet: return None
    text = extract_text(target_tweet)
    if not text or len(text.split()) < 3: # Need some content
        return None
        
    # Simple topic extraction (e.g., first few nouns/keywords - could be improved)
    # For simplicity, let's just use a snippet as the 'topic'
    topic_snippet = " ".join(text.split()[:5]) + "..."
    
    instruction = f"Generate a social media post about the following topic: {topic_snippet}"
    response = text # Use the original tweet as the target response
    return [
        {"role": "system", "content": instruction},
        {"role": "assistant", "content": response}
    ]

keywords_instructions = [
    "Write a social media post that contains the following words: {words}",
    "Create a social post including these words: {words}.",
    "Make a tweet using the following words: {words}.",
    "Can you write a short post that incorporates {words}?",
    "Use the words {words} in a social media caption.",
    "Draft a social update containing these terms: {words}.",
    "Generate a post that includes the words: {words}.",
    "Write a message that mentions: {words}.",
    "Create a post that uses the following keywords: {words}.",
    "Can you come up with a post that has these words: {words}?",
    "Write a social media message that includes: {words}.",
    "Generate a tweet that contains the words: {words}.",
    "Make a post that uses the following words: {words}.",
]

def generate_keyword_generation(chain, user_histories):
    """Generates a 'write using keywords X, Y, Z' instruction."""
    if not chain: return None
    target_tweet = chain[-1]
    if 'text' not in target_tweet: return None
    text = extract_text(target_tweet)
    words = target_tweet['text'].split()
    if len(words) < 5: return None # Need enough words

    # Select 3-5 random keywords (longer words are often more specific)
    potential_keywords = [w for w in words if len(w) > 4]
    if len(potential_keywords) < 1: return None
    
    num_keywords = random.randint(1, min(5, len(potential_keywords)))
    keywords = random.sample(potential_keywords, num_keywords)
    
    instruction = random.choice(keywords_instructions).format(words=", ".join(keywords))
    response = text
    return [
        {"role": "system", "content": instruction},
        {"role": "assistant", "content": response}
    ]

completion_instructions = [
    "Complete the following post: {beginning}",
    "Finish this post: {beginning}",
    "How would you complete this social media message: {beginning}",
    "Continue writing this post starting from: {beginning}",
    "Fill in the rest of this caption: {beginning}",
    "Add a conclusion to this social media post beginning with: {beginning}",
    "Can you finish this message: {beginning}",
    "Complete this social media update: {beginning}"
]

def generate_completion(chain, user_histories):
    """Generates a 'complete the post/thread' instruction."""
    if not chain: return None
    target_tweet = chain[-1]
    if 'text' not in target_tweet: return None
    text = extract_text(target_tweet)
    words = target_tweet['text'].split()
    if len(words) < 10: return None # Need a reasonably long post to split
        
    split_point = random.randint(3, len(words) - 3) # Split somewhere in the middle
    context_text = " ".join(words[:split_point]) + "..."
    response_text = " ".join(words[split_point:])
    
    instruction = random.choice(completion_instructions).format(beginning=context_text)
    response = json.dumps({'text': response_text})
    return [
        {"role": "system", "content": instruction},
        {"role": "assistant", "content": response}
    ]

paraphrase_instructions = [
    "Paraphrase the following post:{post}",
    "Rephrase this post: {post}.",
    "Can you say this differently: {post}?",
    "How would you express this in another way: {post}?",
    "Rewrite the following message: {post}.",
    "Paraphrase this social media update: {post}.",
    "Reword this post in a different way: {post}.",
    "{post}\n\nCan you say this post differently while keeping the same meaning?",
    "Rewrite the following social post using different phrasing.\n{post}",
    "Provide a paraphrased version of this message: {post}.",
    "{post}How would you express this post in other words?"
]

def generate_paraphrase_prompt(chain, user_histories):
    """Generates a prompt for paraphrasing (requires external LLM for response)."""
    if not chain: return None
    target_tweet = random.choice(chain)
    text = extract_text(target_tweet)
    if not text or len(text.split()) < 5: return None
        
    instruction = f"Paraphrase the following post:"
    context = text
    # Response needs to be generated by a strong LLM (e.g., OpenAI)
    return [
        {"role": "system", "content": instruction},
        {"role": "context", "content": context},
        {"role": "assistant", "content": "[LLM-Generated Paraphrase]"}
    ]

voting_intention_instruction = [
    "Based on your previous posts, who are you planning to vote for in the upcoming election?",
    "Given your past discussions, which candidate do you support for the next election?",
    "From your previous messages, can you share your voting intention for the upcoming election?",
    "Considering your past posts, who would you vote for in the next election?",
    "Reflecting on your previous discussions, which candidate do you favor for the upcoming election?",
    "Judging by your recent posts, can we infer who you’ll vote for?",
    "From your prior social media activity, who do you support in the upcoming election?",
    "Based on what you've posted, who seems to be your preferred candidate?",
    "Who do your posts suggest you're backing in the next election?",
    "What does your social media presence indicate about your voting intentions?"
]

def generate_voting_intention(user_history_entry):
    """Generates a 'who would you vote for' instruction based on history."""
    history_str = format_user_history(user_history_entry)
    if not history_str: return None
        
    instruction = "Based on your previous posts, who are you planning to vote for in the upcoming election?"
    context = history_str
    # Response generation is complex. Could be:
    # 1. Generic: "Based on my posts, I haven't explicitly stated my voting intention."
    # 2. LLM-generated based on inferred stance (requires careful prompting).
    # 3. Use a predefined neutral or leaning response if stance can be crudely estimated.
    # For now, using a generic placeholder.
    response = "Based on my previous posts, I focus more on [mention common topics from history, e.g., technology, local events] rather than explicitly stating voting intentions."
    return [
        {"role": "system", "content": instruction},
        {"role": "context", "content": context},
        {"role": "assistant", "content": response}
    ]

# --- Main Processing Logic ---

def main():
    print(f"Starting instruction generation...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dolly dataset path: {args.dolly_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Dolly dataset
    dolly_instructions = []
    if os.path.exists(args.dolly_path):
        print(f"Loading Dolly dataset from {args.dolly_path}...")
        try:
            with open(args.dolly_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        dolly_instructions.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in Dolly dataset: {line.strip()}")
            print(f"Loaded {len(dolly_instructions)} Dolly instructions.")
        except FileNotFoundError:
            print(f"Warning: Dolly dataset not found at {args.dolly_path}. Skipping mix-in.")
        except Exception as e:
            print(f"Warning: Error loading Dolly dataset: {e}. Skipping mix-in.")
    else:
        print(f"Warning: Dolly dataset path {args.dolly_path} does not exist. Skipping mix-in.")

    # Find cluster files
    cluster_files = glob.glob(os.path.join(args.input_dir, "cluster_*.jsonl"))
    print(f"Found {len(cluster_files)} cluster files.")

    if not cluster_files:
        print("Error: No cluster files found in the input directory. Exiting.")
        return

    # Define generation functions and their weights (how often to try each)
    # Add OpenAI-requiring functions separately
    core_generators = [
        (generate_needle_haystack, 0.3),
        # (generate_topic_generation, 0.2),
        (generate_keyword_generation, 0.3),
        (generate_completion, 0.3),
        # generate_voting_intention needs user history, handled separately
    ]
    openai_generators = [
        (generate_summarization_prompt, 0.5),
        (generate_paraphrase_prompt, 0.5),
    ]

    total_generated_count = 0

    for cluster_file in cluster_files:
        print(f"Processing {cluster_file}...")
        output_filename = os.path.join(args.output_dir, os.path.basename(cluster_file))
        generated_instructions = []
        
        try:
            with open(cluster_file, 'r', encoding='utf-8') as f:
                chains = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error reading {cluster_file}: {e}. Skipping.")
            continue

        if not chains:
            print(f"Warning: No chains found in {cluster_file}. Skipping.")
            continue
            
        # Subsample chains if necessary
        if len(chains) > args.max_chains_per_cluster:
            print(f"Subsampling {os.path.basename(cluster_file)} from {len(chains)} to {args.max_chains_per_cluster} chains.")
            chains = random.sample(chains, args.max_chains_per_cluster)

        # Build user histories for voting intention generation
        user_histories = build_user_histories(chains)

        # Generate instructions from chains
        for chain in chains:
            if not chain: continue

            # Decide whether to generate an OpenAI-requiring prompt
            if random.random() < args.openai_fraction:
                 # Try OpenAI generators
                gen_func, _ = random.choices(openai_generators, weights=[w for _, w in openai_generators], k=1)[0]
                instruction = gen_func(chain, user_histories)
                if instruction:
                    generated_instructions.append(instruction)
            else:
                # Try core generators
                gen_func, _ = random.choices(core_generators, weights=[w for _, w in core_generators], k=1)[0]
                instruction = gen_func(chain, user_histories)
                if instruction:
                    generated_instructions.append(instruction)

        # Generate voting intention instructions from user histories
        # Try to generate one for roughly 10% of users
        # num_voting_prompts = math.ceil(len(user_histories) * 0.1)
        # users_for_voting = random.sample(list(user_histories.keys()), min(num_voting_prompts, len(user_histories)))
        # for user_id in users_for_voting:
        #     instruction = generate_voting_intention(user_histories[user_id])
        #     if instruction:
        #         generated_instructions.append(instruction)

        # Mix in Dolly instructions
        if dolly_instructions:
            num_dolly_to_add = int(len(generated_instructions) * args.dolly_mix_fraction / (1 - args.dolly_mix_fraction))
            num_dolly_to_add = min(num_dolly_to_add, len(dolly_instructions)) # Ensure we don't try to add more than available
            if num_dolly_to_add > 0:
                print(f"Adding {num_dolly_to_add} Dolly instructions...")
                dolly_sample = random.sample(dolly_instructions, num_dolly_to_add)
                for instruction in dolly_sample:
                    # Reformat Dolly instructions to match the expected structure
                    msg = []
                    msg.append({"role": "system", "content": instruction["instruction"]})
                    if instruction['context']: 
                        msg.append({"role": "context", "content": instruction["context"]})
                    msg.append({"role": "assistant", "content": json.dumps({'text': instruction["response"]})})
                    generated_instructions.append(msg)


        # Shuffle and save
        random.shuffle(generated_instructions)
        
        # --- OpenAI Batch API Integration Point ---
        # If using OpenAI API, filter out instructions with placeholders ("PLACEHOLDER_SUMMARY", "PLACEHOLDER_PARAPHRASE")
        # Prepare a batch input file with these prompts.
        # Submit the batch job.
        # Wait for completion and retrieve results.
        # Replace placeholders in `generated_instructions` with actual LLM responses.
        # This part is complex and requires handling async API calls, costs, and rate limits.
        # Example structure (pseudo-code):
        # prompts_for_openai = [item for item in generated_instructions if item["response"].startswith("PLACEHOLDER")]
        # if prompts_for_openai:
        #     # 1. Format prompts_for_openai into OpenAI Batch API input JSONL
        #     # 2. Upload file to OpenAI
        #     # 3. Create batch job
        #     # 4. Poll job status
        #     # 5. Download results file
        #     # 6. Parse results and update generated_instructions
        #     print(f"Generated {len(prompts_for_openai)} prompts requiring OpenAI completion.")
        #     print("Please run these through the OpenAI Batch API and replace placeholders.")
        # -----------------------------------------

        print(f"Generated {len(generated_instructions)} instructions for {os.path.basename(cluster_file)}. Saving to {output_filename}...")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                for instruction_data in generated_instructions:
                    # Ensure basic structure compliance
                    f_out.write(json.dumps(instruction_data) + '\n')

            total_generated_count += len(generated_instructions)
        except Exception as e:
            print(f"Error writing {output_filename}: {e}")

    print(f"Finished generating instructions. Total instructions generated: {total_generated_count}")
    print(f"Output files are in: {args.output_dir}")
    # Remind about OpenAI placeholders if any were generated
    if any(g == generate_summarization_prompt or g == generate_paraphrase_prompt for g, _ in openai_generators) and args.openai_fraction > 0:
         print("\nReminder: Instructions requiring summarization or paraphrasing have placeholder responses (e.g., 'PLACEHOLDER_SUMMARY').")
         print("You will need to use an external LLM (like the OpenAI Batch API) to generate actual responses for these prompts.")


if __name__ == "__main__":
    main()
