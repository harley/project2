from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

print(documents)

from openai import OpenAI

client = OpenAI()


def generate_qa(prompt, text, temperature=0.2):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    print(content)

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("\n", 1)[0]

    conetent = content.strip()

    # attempt to parse the cleaned content as json
    try:
        json_data = json.loads(content)
        return json_data
    except json.JSONDecodeError:
        print("Failed to parse JSON" + content)
        return []


factual_prompt = """
You are an expert educational content creator tasked with generating factual questions and answers based on the following document excerpt. These questions should focus on retrieving specific details, figures, definitions, and key facts from the text.

Instructions:

- Generate **5** factual questions, each with a corresponding **expected_output**.
- Ensure all questions are directly related to the document excerpt.
- Present the output in the following structured JSON format:

[
  {
    "question": "What is the main purpose of the project described in the document?",
    "expected_output": "To develop a new framework for data security using AI-powered tools."
  },
  {
    "question": "Who authored the report mentioned in the document?",
    "expected_output": "Dr. Jane Smith."
  }
]
"""
import json
import os

dataset_file = "qa_dataset.json"

if os.path.exists(dataset_file):
    # load dataset from local file if exists
    with open(dataset_file, "r") as file:
        dataset = json.load(file)
else:
    dataset = []
    for doc in documents:
        qa_set = generate_qa(factual_prompt, doc.text)
        dataset.extend(qa_set)

    with open(dataset_file, "w") as file:
        json.dump(dataset, file, indent=4)

# create the dataset in Langfuse

import langfuse

langfuse = Langfuse()

dataset_name = "strategic_plan_qa_pairs"
langfuse.create_dataset(name=dataset_name)

for item in dataset:
    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input=item["question"],
        expected_output=item["expected_output"],
    )
