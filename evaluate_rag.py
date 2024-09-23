from dotenv import load_dotenv

load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What is the strategic plan?")

print(response)

from langfuse import Langfuse
import openai

langfuse = Langfuse()

import json


# use a very simple eval
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_eval(output, expected_output):
    client = openai.OpenAI()
    prompt = f"""
Compare the following output to the expected output and evaluate its accuracy.

Output: {output}

Expected Output: {expected_output}

Provide a score (0 for incorrect, 1 for correct) and a short explanation for the score.

Return the score and explanation in JSON format.
{{
"score": 0 or 1,
"reason": "short explanation for the score"
}}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with evaluating the accuracy of a response based on an expected output.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    result = response.choices[0].message.content

    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")

    result = json.loads(result)
    return result["score"], result["reason"]


from datetime import datetime


def rag_query(input):

    generationStartTime = datetime.now()

    response = query_engine.query(input)
    output = response.response
    print(output)

    langfuse_generation = langfuse.generation(
        name="strategic-plan-qa",
        input=input,
        output=output,
        model="gpt-3.5-turbo",
        start_time=generationStartTime,
        end_time=datetime.now(),
    )

    return output, langfuse_generation


def run_experiment(experiment_name):
    dataset = langfuse.get_dataset("strategic_plan_qa_pairs")

    for item in dataset.items:
        completion, langfuse_generation = rag_query(item.input)

        item.link(
            langfuse_generation, experiment_name
        )  # pass the observation/generation object or the id

        score, reason = llm_eval(completion, item.expected_output)
        langfuse_generation.score(name="accuracy", value=score, comment=reason)


run_experiment("Experiment 2")
