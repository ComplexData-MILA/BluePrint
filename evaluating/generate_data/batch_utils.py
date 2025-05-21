import os
from openai import OpenAI
import argparse

DEFAULT_FILEPATH = "batchinput.jsonl"
DEFAULT_OUTPUTPATH = "batchoutput.jsonl"
LAST_BATCH_ID_FILE = "last_batch_id.tok"

parser = argparse.ArgumentParser(description='Compute evaluation metrics for a model')
parser.add_argument("--run", help="If present, run the batch", action="store_true", default=False)
parser.add_argument("--file", type=str, help="jsonl input file containing the requests", default=DEFAULT_FILEPATH)
parser.add_argument("--download", type=str, help="Download the output file for a batch ID to the specified path. Requires --id or a previous run.", default=None)
parser.add_argument("--check", help="If present, check the status of the batch with specified id", action="store_true", default=False)
parser.add_argument("--cancel", help="If present, cancel the batch with specified id", action="store_true", default=False)
parser.add_argument("--id", type=str, help="ID of the batch to check, cancel, or download output for. If omitted, uses the ID from the last --run.", default=None)

args = parser.parse_args()

filepath = args.file

with open("openai.tok", 'r') as f:
    openai_api_key = f.read().strip()

client = OpenAI(api_key=openai_api_key)

def get_batch_id(args_id):
    if args_id:
        return args_id
    if os.path.exists(LAST_BATCH_ID_FILE):
        with open(LAST_BATCH_ID_FILE, 'r') as f:
            last_id = f.read().strip()
            if last_id:
                print(f"Using last saved batch ID: {last_id}")
                return last_id
    return None

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
            "description": "LLM as persona"
        }
    )

    print(batch_metadata)
    with open(LAST_BATCH_ID_FILE, 'w') as f:
        f.write(batch_metadata.id)
    print(f"Saved batch ID {batch_metadata.id} to {LAST_BATCH_ID_FILE}")

if args.check:
    batch_id = get_batch_id(args.id)
    if batch_id is None:
        print("Please provide a batch id using --id or run a batch first.")
    else:
        print(f"Checking status for batch ID: {batch_id}")
        print(client.batches.retrieve(batch_id))

if args.cancel:
    batch_id = get_batch_id(args.id)
    if batch_id is None:
        print("Please provide a batch id using --id or run a batch first.")
    else:
        print(f"Cancelling batch ID: {batch_id}")
        client.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancel request sent.")

if args.download:
    output_path = args.download
    batch_id = get_batch_id(args.id)
    if batch_id is None:
        print("Please provide a batch id using --id or run a batch first to specify which batch output to download.")
    else:
        print(f"Attempting to download output for batch ID: {batch_id} to {output_path}")
        try:
            batch_info = client.batches.retrieve(batch_id)
            if batch_info.status == 'completed':
                output_file_id = batch_info.output_file_id
                error_file_id = batch_info.error_file_id
                if output_file_id:
                    print(f"Batch completed. Downloading output file ID: {output_file_id}")
                    file_response = client.files.content(output_file_id)
                    with open(output_path, "wb") as f:
                        f.write(file_response.content)
                    print(f"Successfully downloaded output to {output_path}")
                elif error_file_id:
                    print(f"Batch completed but no output file ID was found. Downloading error file ID: {error_file_id}")
                    error_response = client.files.content(error_file_id)
                    error_output_path = output_path + ".errors.jsonl"
                    with open(error_output_path, "wb") as f:
                        f.write(error_response.content)
                    print(f"Successfully downloaded errors to {error_output_path}")
                else:
                    print("Error: Batch is completed but no output file ID was found.")
            elif batch_info.status == 'failed':
                 print(f"Batch failed. Error details: {batch_info.errors}")
                 error_file_id = batch_info.error_file_id
                 if error_file_id:
                     print(f"Downloading error file ID: {error_file_id}")
                     try:
                         error_response = client.files.content(error_file_id)
                         error_output_path = output_path + ".errors.jsonl"
                         with open(error_output_path, "wb") as f:
                             f.write(error_response.content)
                         print(f"Successfully downloaded errors to {error_output_path}")
                     except Exception as e:
                         print(f"Could not download error file: {e}")
            else:
                print(f"Batch status is '{batch_info.status}'. Output is not ready for download yet.")
        except Exception as e:
            print(f"An error occurred while trying to download batch output: {e}")

if not (args.run or args.check or args.cancel or args.download):
    print("Listing batches:")
    client.batches.list()