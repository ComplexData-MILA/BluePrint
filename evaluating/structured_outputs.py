from outlines import models, generate
import getpass
from utils import load_prompts, load_model, chain_into_prompt

USERNAME = getpass.getuser()
MODEL_PATH = f"/scratch/{USERNAME}/Qwen/Qwen2.5-7B-Instruct-lora-finetuned-2"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# System prompt as used during training
SYSTEM_PROMPT, REPLY_SCHEMA, POST_SCHEMA = load_prompts()

print(f"Running as {USERNAME}")

model, tokenizer = load_model(MODEL_PATH, MODEL_NAME, username=USERNAME)
model.eval()


TOPIC = "\n\nWrite a post on the following topic: JD Vance"
system_prompts = [
    "Write a post on the following topic: JD Vance",
    "Write a post on the following topic: JD Vance",
    "Write a post on the following topic: JD Vance",
    "Write a post on the following topic: JD Vance",
    "Write a post on the following topic: JD Vance",
    "Who would you vote for: Donald Trump or Kamala Harris? Answer only with the name of your preferred candidate",
    "Who would you vote for: Donald Trump or Kamala Harris? Answer only with the name of your preferred candidate",
    "Who would you vote for: Donald Trump or Kamala Harris? Answer only with the name of your preferred candidate",
    "Who would you vote for: Donald Trump or Kamala Harris? Answer only with the name of your preferred candidate",
    "Who would you vote for: Donald Trump or Kamala Harris? Answer only with the name of your preferred candidate",
    "What is your opinion on tariffs?",
    "What is your opinion on tariffs?",
    "What is your opinion on tariffs?",
    "What is your opinion on tariffs?"
    ]

batch_chains = [[] for i in system_prompts]
batch_prompts = []

for (prompt_chain, system_prompt) in zip(batch_chains, system_prompts):
    batch_prompts.append(chain_into_prompt([], system_prompt, tokenizer))

for i, prompt in enumerate(batch_prompts):
    print(f"Prompt {i}: {prompt}")

model = models.Transformers(model, tokenizer)
generator = generate.json(model, POST_SCHEMA)

for prompt in batch_prompts:
    result = generator(prompt)
    print(result)