import random
import json
import re
from process_model_output import filter_text

PATH = "/scratch/s4yor1/evaluation_dataset/cluster_1/model_7B_0.jsonl"
def get_random_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            random_line = random.choice(lines)
            return random_line
        else:
            return "File is empty"

random_line = get_random_line(PATH)

# Parse the JSON line and pretty print messages
try:
    messages = json.loads(random_line)
    print("\n=== CONVERSATION ===")
    for i, message in enumerate(messages, 1):
        actions, text = filter_text(message['text'])
        print(f"Message {i}:")
        print(f"  User ID: {message['user_id']}")
        # print(f"  Unfiltered Text: {message['text']}\n")
        print(f"  Text: {text}")
        print(f"  Actions Taken: {actions if actions else 'None'}\n")
except json.JSONDecodeError:
    print("Could not parse line as JSON")
