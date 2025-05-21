import toml
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
import torch
import getpass
import json


def load_prompts():
    # System prompt as used during training
    with open("../shared/prompts.toml", "r") as f:
        prompts = toml.load(f)
        SYSTEM_PROMPT = prompts["SYSTEM_PROMPT"]
        REPLY_SCHEMA = prompts["REPLY_SCHEMA"]
        POST_SCHEMA = prompts["POST_SCHEMA"]

    return SYSTEM_PROMPT, REPLY_SCHEMA, POST_SCHEMA


def load_model(model_path, model_name, username=getpass.getuser(), test_without_model=False):
    print(f"Loading model from {model_path}")

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
    return model, tokenizer


def apply_chat_template(messages):
    formatted_text = ""

    for message in messages:
        formatted_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text


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

def chain_into_prompt(chain, system_prompt, tokenizer):
    # Extract the chain without the last message
    if len(chain) > 0:
        last_user_id = chain[-1]['user_id']
        chain = chain[:-1]

        user_ids = {}
        for tweet in chain:
            if tweet['user_id'] not in user_ids:
                user_ids[tweet['user_id']] = f"user{len(user_ids)}"
        
        # The last user_id in the prompt will be the assistant
        user_ids[last_user_id] = "assistant"
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    for i, tweet in enumerate(chain):
        uid = user_ids[tweet['user_id']]
        if i == 0:
            messages.append({"role": uid, "content": text_to_structured_output(tweet['text'], is_reply=False)})
        else:
            messages.append({"role": uid, "content": text_to_structured_output(tweet['text'], is_reply=True)})
    
    formatted_text = apply_chat_template(messages)
    # formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return formatted_text