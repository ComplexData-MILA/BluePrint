import json
import os
from outlines import models, generate, samplers
import getpass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
import torch
import argparse
import toml
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate completions for validation data using a finetuned model')
parser.add_argument("--slurm_array_task_id", type=int, help="Task id for slurm array jobs", default=0)
parser.add_argument("--use_lora", action="store_true", help="Use LoRA model")
parser.add_argument("--no_focal", action="store_true", help="Test without loading the model")
args = parser.parse_args()

USERNAME = getpass.getuser()
CHATGPT_INPUT_FILEPATH = "batchinput.jsonl"
EVALUATION_DATASET_PATH = os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset")
MODEL_PATH = os.path.expanduser(f"~/bluesky_blueprint/scratch/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-{args.slurm_array_task_id}{'-no-focal' if args.no_focal else ''}")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH = os.path.expanduser(f"~/bluesky_blueprint/scratch/test_outputs")

def load_prompts():
    # System prompt as used during training
    with open("../../shared/prompts.toml", "r") as f:
        prompts = toml.load(f)
        SYSTEM_PROMPT = prompts["SYSTEM_PROMPT"]
        REPLY_SCHEMA = prompts["REPLY_SCHEMA"]
        POST_SCHEMA = prompts["POST_SCHEMA"]

    return SYSTEM_PROMPT, REPLY_SCHEMA, POST_SCHEMA

_, REPLY_SCHEMA, POST_SCHEMA = load_prompts()

def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data

def apply_chat_template(messages):
    formatted_text = ""

    for message in messages:
        formatted_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text

def load_model(model_name, username=getpass.getuser(), test_without_model=False):
    print(f"Loading model from {model_name}")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model with fp16 precision
    if test_without_model:
        print("Testing without loading model")
        return None, tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=os.path.expanduser(f"~/bluesky_blueprint/scratch/HF-cache")
    )

    return base_model, tokenizer

def load_peft_model(model_path, base_model):
    print(f"Loading model from {model_path}")
    # If the path points to a LoRA adapter, load it
    if os.path.exists(os.path.join(model_path, "adapter_model.bin")) or os.path.exists(os.path.join(model_path, "lora_adapter")):
        lora_adapter_path = model_path
        if os.path.exists(os.path.join(model_path, "lora_adapter")):
            lora_adapter_path = os.path.join(model_path, "lora_adapter")

        print(f"Loading LoRA adapter from {lora_adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            torch_dtype=torch.bfloat16
        )
    else:
        print("Failed to load LoRA adapter, using base model only")
        model = base_model
    # model = base_model

    # Set to evaluation mode
    model.eval()
    print("Model loaded successfully")
    return model


data = load_jsonl(CHATGPT_INPUT_FILEPATH)
print(f"Loaded {len(data)} entries from {CHATGPT_INPUT_FILEPATH}")

# sort tasks by cluster id
clusters = {}
for entry in data:
    ids = entry["custom_id"].split("-") # task-id-chain_id
    cluster_id = ids[1]
    chain_id = ids[2]
    if cluster_id not in clusters:
        clusters[cluster_id] = set()
    clusters[cluster_id].add(int(chain_id))

base_model, tokenizer = load_model(MODEL_NAME, username=USERNAME)

cluster_id = args.slurm_array_task_id
chain_ids = clusters.get(str(cluster_id), None)
print(f"Cluster {cluster_id} has {len(chain_ids)} chains")

chains = load_jsonl(os.path.join(EVALUATION_DATASET_PATH, f"cluster_{cluster_id}", "validation.jsonl"))
selected_chains = [chains[i][:-1] for i in chain_ids]

if args.use_lora:
    model = load_peft_model(MODEL_PATH, base_model)
else:
    model = base_model

option_labels = ["A", "B", "C"]
sampler = samplers.multinomial(len(option_labels))
model = models.Transformers(model, tokenizer)
post_generator = generate.json(model, POST_SCHEMA, sampler=sampler)
reply_generator = generate.json(model, REPLY_SCHEMA, sampler=sampler)
selector = generate.regex(models.Transformers(base_model, tokenizer), r"[012]", sampler=samplers.multinomial(5))

generated_data = []

for i, chain_messages in zip(chain_ids, tqdm(selected_chains)):
    if len(chain_messages) > 1:
        generator = reply_generator
    else:
        generator = post_generator
    
    # This is the prompt for the initial generation of 0, 1, 2.
    generation_prompt_text = apply_chat_template(chain_messages)

    # Generate 3 answers
    generated_options = generator(generation_prompt_text) # generator returns a Python object (dict/list)

    # Construct the prompt for the selector model.
    # The selector model will choose the best option among 0, 1, 2.
    # The prompt should provide the context (original messages) and the generated options.

    selection_user_message_content = f"""Here are three messages. Two are fake and one is real.

Option 0:
{json.dumps(generated_options[0], indent=2)}

Option 1:
{json.dumps(generated_options[1], indent=2)}

Option 2:
{json.dumps(generated_options[2], indent=2)}

Which of these options (0, 1, or 2) do you think was written by a human? Answer with a single number: 0, 1, or 2. Note that if the text does not make sense, or contains strange characters, it probably is not human-written.
"""

    selector_messages_list= [{
        "role": "system",
        "content": selection_user_message_content
    }]

    selector_prompt_text = apply_chat_template(selector_messages_list)

    # Call the selector to get the chosen option ("A", "B", or "C")
    selected_options = selector(selector_prompt_text)
    # Count occurrences of each option
    option_counts = {'0': 0, '1': 0, '2': 0}
    for option in selected_options:
        if option in option_counts:
            option_counts[option] += 1
        else:
            print(f"Unexpected option: {option}")
    # Get the option with the highest count
    selected_option_char = max(option_counts, key=option_counts.get)

    final_selected_output = generated_options[0 if selected_option_char == "0" else 1 if selected_option_char == "1" else 2]

    generated_data.append({
        "custom_id": f"task-{cluster_id}-{i}",
        "model_output_0": generated_options[0],
        "model_output_1": generated_options[1],
        "model_output_2": generated_options[2],
        "selector_choice": selected_option_char,
        "final_model_output": final_selected_output
    })


output_path = os.path.join(OUTPUT_PATH, f"cluster_{cluster_id}{'_not_finetuned' if not args.use_lora else ''}{'-no-focal' if args.no_focal else ''}.jsonl")
with open(output_path, 'w') as f:
    for entry in generated_data:
        f.write(json.dumps(entry) + "\n")
print(f"Saved generated data to {output_path}")