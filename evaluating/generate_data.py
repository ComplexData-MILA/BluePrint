import os
import json
import torch
import getpass
import argparse
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel
from process_model_output import filter_text, is_complete_response
import toml
from outlines import models, generate, processors

# filepath: /home/s4yor1/SM-based-personas/evaluating/generate_data.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# Arguments
parser = argparse.ArgumentParser(description='Generate completions for validation data using a finetuned model')
parser.add_argument('--model_path', type=str, help='Path to the finetuned model directory')
parser.add_argument('--model_name', type=str, default=None, help='Name of the model to load (if not using a local path)')
parser.add_argument('--cluster', type=int, default=None, help='Specific cluster to process (None for all)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
parser.add_argument('--output', type=str, default=None, help='Name of the output files for generated completions')
parser.add_argument("--slurm_array_task_id", type=int, help="Task id for slurm array jobs", default=0)
args = parser.parse_args()

# Set file paths
USERNAME = getpass.getuser()
DATASET_PATH = f"/scratch/{USERNAME}/evaluation_dataset"
MODEL_PATH = args.model_path if args.model_path else f"/scratch/{USERNAME}/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-{args.slurm_array_task_id}"
MODEL_NAME = args.model_name if args.model_name else "Qwen/Qwen2.5-7B-Instruct"
OUTPUT = args.output if args.output else f"model_{args.slurm_array_task_id}"
os.environ["HF_HOME"] = f"/scratch/{USERNAME}/HF-cache"
TEST_WITHOUT_MODEL = False

# System prompt as used during training
with open("../shared/prompts.toml", "r") as f:
    prompts = toml.load(f)
    SYSTEM_PROMPT = prompts["SYSTEM_PROMPT"]
    REPLY_SCHEMA = prompts["REPLY_SCHEMA"]
    POST_SCHEMA = prompts["POST_SCHEMA"]


def load_model():
    print(f"Loading model from {MODEL_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model with fp16 precision
    if TEST_WITHOUT_MODEL:
        print("Testing without loading model")
        return None, tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=f"/scratch/{USERNAME}/HF-cache"
    )
    
    # If the path points to a LoRA adapter, load it
    if os.path.exists(os.path.join(MODEL_PATH, "adapter_model.bin")) or os.path.exists(os.path.join(MODEL_PATH, "lora_adapter")):
        lora_adapter_path = MODEL_PATH
        if os.path.exists(os.path.join(MODEL_PATH, "lora_adapter")):
            lora_adapter_path = os.path.join(MODEL_PATH, "lora_adapter")

        print(f"Loading LoRA adapter from {lora_adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            torch_dtype=torch.bfloat16
        )
    else:
        print("Failed to load LoRA adapter, using base model only")
        model = base_model
    
    # Set to evaluation mode
    model.eval()
    print("Model loaded successfully")
    return model, tokenizer

# Convert the text in the dataset to a json structured output
def text_to_structured_output(text, is_reply=False):
    if is_reply:
        structured_output = {
            "actions": {
                "like": False,
                "follow": False,
                "repost": False,
                "ignore": False
            }
        }

        actions = ["[action: ignore]", "[action: like]",
                "[action: follow]", "[action: repost]"]

        # Loop through each action and check if it's in the text
        for action in actions:
            if action in text:
                # Add to actions_taken with the action name as the key
                action_name = action.replace("[action: ", "").replace("]", "")
                structured_output["actions"][action_name] = True
                # Remove the action tag from the text
                text = text.replace(action, "")

        # Strip any extra whitespace
        text = text.strip()

        if len(text) > 0:
            structured_output["text"] = text

        return json.dumps(structured_output)
    else:
        return json.dumps({"text": text})

def apply_chat_template(messages):
    """Only use this method for Qwen"""
    formatted_text = ""
    
    for message in messages:
        formatted_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text


def extract_new_tokens(inputs, outputs, tokenizer):
    # Extract new tokens from the outputs
    new_tokens_list = []
    for b_idx in range(outputs.shape[0]):
        # Find where input tokens end (where attention_mask is 1)
        input_length = len(inputs['attention_mask'][0])
        
        # Get only the newly generated tokens for this batch item
        new_tokens = outputs[b_idx, input_length:]
        new_tokens_list.append(new_tokens)

    # Convert to a tensor for consistency (optional)
    if new_tokens_list:
        max_len = max(len(tokens) for tokens in new_tokens_list)
        padded_tokens = []
        for tokens in new_tokens_list:
            padded = torch.nn.functional.pad(tokens, (0, max_len - len(tokens)), value=tokenizer.pad_token_id)
            padded_tokens.append(padded)
        new_tokens_tensor = torch.stack(padded_tokens)
    
    return new_tokens_tensor


def generate_completions(model, tokenizer, dataset_path, cluster_id=None):
    
    model = models.Transformers(model, tokenizer)
    reply_generator = generate.json(model, REPLY_SCHEMA)
    post_generator = generate.json(model, POST_SCHEMA)
    
    # Process specific cluster if requested, otherwise process all
    if cluster_id is not None:
        clusters = [f"cluster_{cluster_id}"]
    else:
        clusters = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith("cluster_")]
    
    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_path = os.path.join(dataset_path, cluster)
        validation_path = os.path.join(cluster_path, "validation.jsonl")
        
        # Skip if validation file doesn't exist
        if not os.path.exists(validation_path):
            print(f"Skipping {cluster} - validation file not found")
            continue
            
        # Load validation data
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_chains = [json.loads(line) for line in f.readlines()]
        
        # Prepare data for model prediction
        generated_chains = []
        generated_texts = []

        for prompt_chain in validation_chains:
            # Extract the chain without the last message
            last_user_id = prompt_chain[-1]['user_id']
            prompt_chain = prompt_chain[:-1]
            
            # Create prompt for model
            user_ids = {}
            for tweet in prompt_chain:
                if tweet['user_id'] not in user_ids:
                    user_ids[tweet['user_id']] = f"user{len(user_ids)}"
            
            # The last user_id in the prompt will be the assistant
            user_ids[last_user_id] = "assistant"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            
            for i, tweet in enumerate(prompt_chain):
                uid = user_ids[tweet['user_id']]
                if i == 0:
                    messages.append({"role": uid, "content": text_to_structured_output(tweet['text'], is_reply=False)})
                else:
                    messages.append({"role": uid, "content": text_to_structured_output(tweet['text'], is_reply=True)})

            
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            if len(prompt_chain) >= 1:
                generated_texts.append(reply_generator(formatted_text))
            else:
                generated_texts.append(post_generator(formatted_text))

        for j, text in enumerate(generated_texts):

            # Create a new chain with generated response
            new_chain = validation_chains[i+j][:-1].copy()
            new_chain.append({
                "user_id": validation_chains[i+j][-1]["user_id"],
                "text": text
            })
            generated_chains.append(new_chain)

        # Extract cluster ID for output file name
        output_file = os.path.join(cluster_path, f"{OUTPUT}.jsonl")
        
        # Save generated chains
        with open(output_file, 'w', encoding='utf-8') as f:
            for chain in generated_chains:
                # Convert NumPy types to native Python types before serializing
                def convert_numpy_types(obj):
                    if hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(i) for i in obj]
                    else:
                        return obj
                
                chain_converted = convert_numpy_types(chain)
                f.write(json.dumps(chain_converted) + '\n')
        
        print(f"Generated {len(generated_chains)} completions for {cluster}, saved to {output_file}")

if __name__ == "__main__":

    # Load the model and tokenizer
    model, tokenizer = load_model()
    
    # Generate completions
    generate_completions(model, tokenizer, DATASET_PATH, args.cluster)
    
    print("Generation complete!")