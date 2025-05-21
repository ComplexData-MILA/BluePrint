from openai import OpenAI
import argparse

DEFAULT_FILEPATH = "batchinput.jsonl"
DEFAULT_OUTPUTPATH = "batchoutput.jsonl"

parser = argparse.ArgumentParser(description='Compute evaluation metrics for a model')
parser.add_argument("--run", help="If present, run the batch", action="store_true", default=False)
parser.add_argument("--file", type=str, help="jsonl input file containing the requests", default=DEFAULT_FILEPATH)
parser.add_argument("--download", nargs='?', type=str, help="Output a file containing the responses", const=DEFAULT_OUTPUTPATH, default=None)
parser.add_argument("--check", help="If present, check the status of the batch with specified id", action="store_true", default=False)
parser.add_argument("--cancel", help="If present, cancel the batch with specified id", action="store_true", default=False)
parser.add_argument("--id", type=str, help="ID of the batch to check or cancel", default=None)

args = parser.parse_args()

filepath = args.file

with open("openai.tok", 'r') as f:
    openai_api_key = f.read().strip()

client = OpenAI(api_key=openai_api_key)

if args.run:
    print("Uploading batch input file")
    batch_input_file = client.files.create(
        file=open(filepath, "rb"),
        purpose="batch"
    )

    print(batch_input_file)

    print("Creating batch")
    batch_input_file_id = batch_input_file.id
    batch_metadata = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "LLM as judge for personas"
        }
    )

    print(batch_metadata)
    
if args.check:
    if args.id is None:
        print("Please provide a batch id to check")
    else:
        client.batches.retrieve(args.id)

if args.cancel:
    if args.id is None:
        print("Please provide a batch id to cancel")
    else:
        client.batches.cancel(args.id)

if args.download:
    if args.id is None:
        print("Please provide a batch id to download")
    else:
        file_response = client.files.content(args.id)
        with open(args.download, "wb") as f:
            f.write(file_response.content)

# If there are no arguments, just list all batches
if not (args.run or args.check or args.cancel or args.download):
    print("Listing batches:")
    client.batches.list()