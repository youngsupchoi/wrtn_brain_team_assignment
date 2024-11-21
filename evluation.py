import os
from typing import final
from langchain_core.documents import Document
from openai import OpenAI
# Load environment variables from .env file
from dotenv import load_dotenv
from pydantic import BaseModel
from agent import Agent
from tools import create_hybrid_retrieval, get_OPENAI_llm
# Get the API key
from openai import OpenAI

from datasets import load_dataset
from pprint import pprint
from tqdm import tqdm

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law")

train = ds['train']
dev = ds['dev']
test = ds['test']


load_dotenv()

import os
import json

# Evaluate the test set using the solve_and_check_answer function and calculate the score
correct_count = 0
total_count = len(test)
start_index = 0
error_indices = []
agent = Agent()

# Check if there's a checkpoint file and read the last processed index
if os.path.exists('checkpoint.txt'):
  with open('checkpoint.txt', 'r') as f:
    content = f.read().strip()
    if content:
      start_index = int(content)

for idx, item in tqdm(enumerate(test), total=total_count, desc="Processing items", ncols=100):
  if idx < start_index:
    continue
  try:
    # agent로직 시작
    result = agent.retrieve_and_summarize(item)

    # Assuming result is an instance of AnswerCheck
    with open('test_results.json', 'a') as f:
      json.dump(result, f, ensure_ascii=False)
      f.write('\n')
    if result['is_correct']:
      correct_count += 1
    # Update the checkpoint file
    with open('checkpoint.txt', 'w') as f:
      f.write(str(idx + 1))
      
    # Print progress
    # print(f"Processed item {idx + 1}/{total_count}")
    
    tqdm.write(f"Processed item {idx + 1}/{total_count}, correct count: {correct_count}")
      
  except Exception as e:
    print(f"Error processing item at index {idx}: {e}")
    error_indices.append(idx)

# Save error indices to a file
with open('error_indices.txt', 'w') as f:
  for index in error_indices:
    f.write(f"{index}\n")

score = correct_count / total_count * 100
print(f"Score: {score}%")
