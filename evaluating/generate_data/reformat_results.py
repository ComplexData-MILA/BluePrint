import json
import os

with open("output_gpt4.1_mini.jsonl", "r") as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

clusters = {}
for item in data:
    ids = item["custom_id"].split("-") # task-id-chain_id
    cluster_id = ids[1]
    chain_id = ids[2]
    
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append({"custom_id": item["custom_id"], "content": item["response"]["body"]["choices"][0]["message"]["content"]})

for cluster_id, items in clusters.items():
    output_path = os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset"), f"cluster_{cluster_id}", f"gpt4.1mini.jsonl")
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps([item]) + "\n")

with open("output_o3_mini.jsonl", "r") as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

clusters = {}
for item in data:
    ids = item["custom_id"].split("-") # task-id-chain_id
    cluster_id = ids[1]
    chain_id = ids[2]
    
    if cluster_id not in clusters:
        clusters[cluster_id] = []

    content = item["response"]["body"]["choices"][0]["message"]["content"]
    if not content:
        content = item["response"]["body"]["choices"][0]["message"]["refusal"]
    clusters[cluster_id].append({"custom_id": item["custom_id"], "content": content})

for cluster_id, items in clusters.items():
    output_path = os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset"), f"cluster_{cluster_id}", f"o3mini.jsonl")
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps([item]) + "\n")


for file in os.listdir(os.path.expanduser(f"~/bluesky_blueprint/scratch/test_outputs")):
    if file.endswith(".jsonl"):
        with open(os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/test_outputs"), file), "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            print(f"File: {file}, Number of lines: {len(data)}")

    clusters = {}
    for item in data:
        ids = item["custom_id"].split("-") # task-id-chain_id
        cluster_id = ids[1]
        chain_id = ids[2]
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []

        content = item["final_model_output"]
        if not content:
            content = item["final_model_output"]
        clusters[cluster_id].append({"custom_id": item["custom_id"], "content": json.dumps(content)})

    for cluster_id, items in clusters.items():
        if file.endswith('_not_finetuned.jsonl'):
            output_path = os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset"), f"cluster_{cluster_id}", f"not_finetuned.jsonl")
        elif file.endswith('-no-focal.jsonl'):
            output_path = os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset"), f"cluster_{cluster_id}", f"no_focal.jsonl")
        else:
            output_path = os.path.join(os.path.expanduser(f"~/bluesky_blueprint/scratch/evaluation_dataset"), f"cluster_{cluster_id}", f"finetuned.jsonl")
        with open(output_path, "w") as f:
            for item in items:
                f.write(json.dumps([item]) + "\n")