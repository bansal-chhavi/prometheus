import json
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

##############################################################################
# CONFIG
##############################################################################
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JUDGE_MODEL = "llama-3.3-70b-versatile"

##############################################################################
# DATASET
##############################################################################

DATASET = [
    {
        "id": "1",
        "question": "Who led the migration from Elasticsearch 7 to Elasticsearch 8?",
        # "answer": (
        #     "Aakash Arora led the migration from Elasticsearch 7 to "
        #     "Elasticsearch 8 and improved query latency by 35%."
        # ),
        "answer": ("Aakash Arora led the migration and improved latency by 50%"),
    },
    {
        "id": "2",
        "question": "What certifications does Priya Sharma hold?",
        "answer": (
            "Priya Sharma holds AWS Solutions Architect Professional "
            "and Google Professional Cloud Architect certifications."
        ),
    },
    {
        "id": "3",
        "question": "Which expert has experience in renewable energy and agile transformation?",
        "answer": (
            "Rohit Mehta has experience leading agile transformation "
            "programs in the renewable energy sector."
        ),
    },
    {
        "id": "4",
        "question": "Who implemented the recommendation engine using graph neural networks?",
        "answer": (
            "The documents do not contain information about a graph "
            "neural network recommendation engine."
        ),
    },
]

DOCUMENTS = [
    {
        "id": "doc_1",
        "content": """
Aakash Arora served as Lead Search Engineer from 2022 to 2025.

He led the migration from Elasticsearch 7 to Elasticsearch 8
across more than 40 production indices.

The migration reduced average query latency by 35%
and improved indexing throughput by 20%.
""",
    },
    {
        "id": "doc_2",
        "content": """
Priya Sharma is a Principal Cloud Architect.

Certifications:
- AWS Solutions Architect Professional
- Google Professional Cloud Architect
- Certified Kubernetes Administrator

She has designed cloud platforms for banking clients.
""",
    },
    {
        "id": "doc_3",
        "content": """
Rohit Mehta worked as an Agile Transformation Lead
for renewable energy companies.

He managed large-scale agile transformation programs
for solar and wind power organizations.
""",
    },
    {
        "id": "doc_4",
        "content": """
The company search platform uses Elasticsearch,
vector embeddings and large language models.

The architecture supports semantic retrieval,
filter extraction and conversational search.
""",
    },
]

##############################################################################
# JUDGE OUTPUT SCHEMA
##############################################################################


class EvaluationGrade(BaseModel):
    correctness: bool = Field(description="Whether answer matches ground truth")

    groundedness: bool = Field(
        description="Whether answer is supported by retrieved documents"
    )

    relevance: bool = Field(description="Whether answer addresses the question")

    retrieval_relevance: bool = Field(
        description="Whether retrieved documents are relevant"
    )

    reasoning: str = Field(description="Explanation of evaluation")


##############################################################################
# GROQ JUDGE
##############################################################################

judge_llm = ChatGroq(
    model=JUDGE_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
).with_structured_output(EvaluationGrade)


##############################################################################
# TARGET SYSTEM
##############################################################################
#
# Replace this class with your actual search/rag/chatbot pipeline.
#
##############################################################################


class DummyRAGPipeline:

    def invoke(self, question: str):

        if "elasticsearch" in question.lower():
            docs = [DOCUMENTS[0]]

            answer = (
                "Aakash Arora led the migration from Elasticsearch 7 "
                "to Elasticsearch 8 and improved latency by 35%."
            )

        elif "certifications" in question.lower():
            docs = [DOCUMENTS[1]]

            answer = (
                "Priya Sharma holds AWS Solutions Architect Professional "
                "and Google Professional Cloud Architect certifications."
            )

        elif "renewable energy" in question.lower():
            docs = [DOCUMENTS[2]]

            answer = (
                "Rohit Mehta led agile transformation programs "
                "within renewable energy organizations."
            )

        else:
            docs = [DOCUMENTS[3]]

            answer = (
                "The provided documents do not contain information " "about that topic."
            )

        return {"answer": answer, "documents": docs}


##############################################################################
# JUDGE PROMPT
##############################################################################


def evaluate_prediction(
    question: str,
    ground_truth: str,
    answer: str,
    documents: List[Dict[str, Any]],
):
    docs_text = "\n\n".join(d["content"] for d in documents)

    prompt = f"""
You are evaluating a RAG system.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

RETRIEVED DOCUMENTS:
{docs_text}

GENERATED ANSWER:
{answer}

Evaluate:

1. correctness
   - Is the answer factually consistent with the ground truth?

2. groundedness
   - Is the answer supported by the retrieved documents?

3. relevance
   - Does the answer address the question?

4. retrieval_relevance
   - Are the retrieved documents relevant to the question?

Return structured output.
"""

    return judge_llm.invoke(prompt)


##############################################################################
# EVALUATION RUNNER
##############################################################################


class EvaluationRunner:

    def __init__(self, target_system, dataset):
        self.target_system = target_system
        self.dataset = dataset

    def run(self):

        rows = []

        total = len(self.dataset)

        for idx, example in enumerate(self.dataset):

            print(f"[{idx + 1}/{total}] " f"Evaluating: {example['question']}")

            prediction = self.target_system.invoke(example["question"])

            evaluation = evaluate_prediction(
                question=example["question"],
                ground_truth=example["answer"],
                answer=prediction["answer"],
                documents=prediction["documents"],
            )

            rows.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "prediction": prediction["answer"],
                    "correctness": evaluation.correctness,
                    "groundedness": evaluation.groundedness,
                    "relevance": evaluation.relevance,
                    "retrieval_relevance": evaluation.retrieval_relevance,
                    "reasoning": evaluation.reasoning,
                }
            )

        return rows


##############################################################################
# REPORTING
##############################################################################


def generate_report(results):

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    metrics = [
        "correctness",
        "groundedness",
        "relevance",
        "retrieval_relevance",
    ]

    for metric in metrics:

        score = df[metric].astype(int).mean() * 100

        print(f"{metric:25s}: " f"{score:.2f}%")

    print("\n")

    df.to_csv(
        "evaluation_results.csv",
        index=False,
    )

    print("Detailed results saved to " "evaluation_results.csv")

    return df


##############################################################################
# MAIN
##############################################################################


def main():

    rag_pipeline = DummyRAGPipeline()

    runner = EvaluationRunner(
        target_system=rag_pipeline,
        dataset=DATASET,
    )

    results = runner.run()

    generate_report(results)


if __name__ == "__main__":
    main()
