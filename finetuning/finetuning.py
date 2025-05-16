import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Optional, Dict, Union, Any
import getpass
from pathlib import Path
import argparse
import toml
from datasets import concatenate_datasets

parser = argparse.ArgumentParser(description='Compute evaluation metrics for a model')
parser.add_argument("--slurm_array_task_id", type=int, help="Task id for slurm array jobs", default=0)
parser.add_argument("--model_name", type=str, help="Model name", default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--train_batch_size", type=int, help="Batch size", default=6)
parser.add_argument("--eval_batch_size", type=int, help="Batch size", default=4)
parser.add_argument("--grad_accumulation_steps", type=int, help="Gradient accumulation steps", default=8)
parser.add_argument("--focal_loss", action="store_true", help="Use focal loss")

args = parser.parse_args()

# Set cache directory and file paths
USERNAME = getpass.getuser()
MODEL_NAME = args.model_name
os.environ["HF_HOME"] = f"/scratch/{USERNAME}/HF-cache"
DATASET_PATH = f"/scratch/{USERNAME}/pii_removed/processed_25_clusters_hashed/cluster_{args.slurm_array_task_id}.jsonl"
INSTRUCTION_TUNING_DATASET_PATH = f"/scratch/{USERNAME}/instruction_following/cluster_{args.slurm_array_task_id}.jsonl"
OUTPUT_DIR = f"/scratch/{USERNAME}/{MODEL_NAME}-lora-finetuned-{args.slurm_array_task_id}{'-no-focal' if not args.focal_loss else ''}"
EVALUATION_DATASET_FOLDER = f"/scratch/{USERNAME}/evaluation_dataset"
MAX_NUM_SAMPLES = 30_000  # Subsample the dataset to a maximum number of samples
NUM_SAVED_VALIDATION_SAMPLES = 1000  # Number of validation samples to save
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVALUATION_DATASET_FOLDER, exist_ok=True)
SEED = 42
USE_FOCAL_LOSS = args.focal_loss

# System prompt as used during training
with open("../shared/prompts.toml", "r") as f:
    prompts = toml.load(f)
    SYSTEM_PROMPT = prompts["SYSTEM_PROMPT"]

def load_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def load_model():
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=f"/scratch/{USERNAME}/HF-cache"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # Apply LoRA to the model
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def apply_chat_template(messages):
    """Only use this method on Qwen"""
    formatted_text = ""
    
    for message in messages:
        formatted_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return formatted_text

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

def build_user_histories(chains):
    """Creates a single history of all chains, sorted by the timestamp of the last tweet."""
    all_chains_with_timestamp = []
    for i, chain in enumerate(chains):
        if not chain:
            continue
        last_tweet = chain[-1]
        # Use index as fallback if unix_epoch is missing
        timestamp = last_tweet.get('unix_epoch', i) if isinstance(last_tweet, dict) else i
        all_chains_with_timestamp.append({"chain": chain, "timestamp": timestamp})
    
    # Sort all chains by timestamp
    all_chains_with_timestamp.sort(key=lambda x: x["timestamp"])
    return all_chains_with_timestamp

def get_user_history(all_sorted_chains, unix_epoch, max_chains=10, max_tweets_per_chain=5):
    """Retrieves chains from the global history that occurred before a given unix_epoch."""
    if not all_sorted_chains:
        return []

    # Filter for chains before the given unix_epoch
    candidate_chains = [item for item in all_sorted_chains if item["timestamp"] < unix_epoch]
    
    # Take the most recent chains
    relevant_history_chains = candidate_chains[-max_chains:]
    
    processed_history_chains = []
    for entry in relevant_history_chains:
        # Create a copy of the chain to avoid modifying the original
        chain_copy = [tweet.copy() for tweet in entry["chain"]]
        
        # Truncate chain and validate the last tweet
        truncated_chain = chain_copy[-(max_tweets_per_chain + 1):]
        if truncated_chain and (
            "text" not in truncated_chain[-1] 
            or not truncated_chain[-1]["text"]
        ):
            truncated_chain = truncated_chain[:-1]
        
        if truncated_chain: # Add only if the chain is not empty after processing
             processed_history_chains.append({"chain": truncated_chain, "timestamp": entry["timestamp"]})

    return processed_history_chains

def format_user_history(user_chains):
    """Formats a user's history into a readable string."""
    formatted = []
    for entry in user_chains:
        chain = entry["chain"]
        formatted.append(format_chain(chain))
    return "\n".join(formatted)

# New function to prepare messages from chains
def prepare_messages(examples, user_histories):
    all_messages = []
    for chain in examples["chains"]:  # Iterate over batch
        if not chain: continue # Skip empty chains

        user_ids = {}
        for tweet in chain:
            if 'user_id' not in tweet: continue # Skip tweets without user_id
            if tweet['user_id'] not in user_ids:
                user_ids[tweet['user_id']] = f"user{len(user_ids)}"

        # Ensure the last tweet has a user_id before proceeding
        if 'user_id' not in chain[-1]: continue

        # Assign the last user_id to "assistant"
        last_user_id = chain[-1]['user_id']
        user_ids[last_user_id] = "assistant"

        # Get user history, insert it in system prompt
        unix_epoch = chain[-1].get('unix_epoch', 0)
        user_chains = None # get_user_history(user_histories, unix_epoch)

        if user_chains:
            user_history = format_user_history(user_chains)
            system_prompt = SYSTEM_PROMPT.replace("{{history}}", user_history)
        else:
            system_prompt = SYSTEM_PROMPT.replace("{{history}}", "No user history available.")

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for i, tweet in enumerate(chain):
            # Skip tweets without user_id
            if 'user_id' not in tweet: continue
            uid = user_ids[tweet.pop('user_id')]
            tweet.pop('unix_epoch', None)  # Remove unix_epoch if present
            messages.append({"role": uid, "content": json.dumps(tweet)})

        all_messages.append(messages)
    return {'chains': all_messages}

# Tokenize the dataset using the prepared messages
def tokenize_function(examples, tokenizer):

    formatted_texts = []
    for messages in examples['chains']:
        formatted_text = apply_chat_template(messages)
        # formatted_text = tokenizer.apply_chat_template(messages, tokenize=False) # Alternative if using tokenizer's template
        formatted_texts.append(formatted_text)

    return tokenizer(
        formatted_texts,
        truncation=True,
        max_length=8000,
        padding="do_not_pad"
    )

def strip_unix_epoch(chains):
    """Remove unix_epoch from each tweet in the chains"""
    for chain in chains:
        for tweet in chain:
            if 'unix_epoch' in tweet:
                del tweet['unix_epoch']
    return chains

def load_dataset(tokenizer):
    # Load and preprocess the dataset
    print(f"Loading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    # Create training dataset
    dataset = Dataset.from_dict({"chains": data})
    dataset = dataset.shuffle(seed=SEED)
    
    # Subsample the dataset to a maximum number of samples
    if len(dataset) > MAX_NUM_SAMPLES:
        print(f"Subsampling dataset from {len(dataset)} to {MAX_NUM_SAMPLES} examples")
        dataset = dataset.select(range(MAX_NUM_SAMPLES))
    
    dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
    user_histories = build_user_histories(dataset["train"]["chains"])

    print("Preparing dataset...")
    prepared_dataset = dataset.map(
        lambda x: prepare_messages(x, user_histories),
        batched=True,
        num_proc=4,
        remove_columns=["chains"]
    )

    with open(INSTRUCTION_TUNING_DATASET_PATH, 'r', encoding='utf-8') as f:
        instruction_data = [json.loads(line) for line in f.readlines()]
    instruction_dataset = Dataset.from_dict({"chains": instruction_data})
    instruction_dataset = instruction_dataset.shuffle(seed=SEED)
    instruction_dataset = instruction_dataset.train_test_split(test_size=0.1, seed=SEED)

    print("Saving evaluation dataset...")
    dataset_name = Path(DATASET_PATH).stem
    os.makedirs(os.path.join(EVALUATION_DATASET_FOLDER, dataset_name), exist_ok=True)
    with open(os.path.join(EVALUATION_DATASET_FOLDER, dataset_name, "validation.jsonl"), 'w', encoding='utf-8') as f:
        for item in prepared_dataset["test"][:NUM_SAVED_VALIDATION_SAMPLES]["chains"]:
            f.write(json.dumps(item) + "\n")
        for item in instruction_dataset["test"][:NUM_SAVED_VALIDATION_SAMPLES]["chains"]:
            f.write(json.dumps(item) + "\n")

    #merge the two datasets
    train_dataset = concatenate_datasets([prepared_dataset['train'], instruction_dataset['train']])
    test_dataset = concatenate_datasets([prepared_dataset['test'], instruction_dataset['test']])
    
    dataset = DatasetDict({
        'train': train_dataset.shuffle(seed=SEED),
        'test': test_dataset.shuffle(seed=SEED)
    })

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=["chains"]
    )

    print(f"Final dataset size - Train: {len(tokenized_dataset['train'])}, Test: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Don't reshape inputs and targets immediately
        # They need to have compatible batch dimensions
        
        # Mask for valid tokens
        mask = (targets != self.ignore_index)
        
        # Calculate focal loss on the masked tensors
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets.clamp(min=0))  # Clamp to avoid out-of-bounds
                focal_loss = alpha_t * focal_loss
        
        # Apply masking for reduction
        if self.reduction == "mean":
            return focal_loss.sum() / mask.sum() if mask.sum() > 0 else focal_loss.sum() * 0.0
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class FocalLossTrainer(Trainer):
    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = -100  # Default value for ignore_index
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=pad_token_id)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calculate focal loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.focal_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def save(trainer, model, tokenizer):
    # Save the trained model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save LoRA adapter separately
    lora_output_dir = f"{OUTPUT_DIR}/lora_adapter"
    os.makedirs(lora_output_dir, exist_ok=True)
    model.save_pretrained(lora_output_dir)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    tokenized_dataset = load_dataset(tokenizer)
    
    model= load_model()
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="steps",
        eval_steps=10000,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10000,
        save_strategy="steps",
        save_steps=10000,
        learning_rate=3e-4,
        warmup_steps=100,
        num_train_epochs=3,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},  # Add this line
        fp16=True,
        load_best_model_at_end=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Choose the appropriate trainer based on USE_FOCAL_LOSS flag
    if USE_FOCAL_LOSS:
        trainer = FocalLossTrainer(
            gamma=1.0,
            alpha=None,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )
    print("Starting training...")
    trainer.train()
    
    save(trainer, model, tokenizer)
    print("Training complete!")