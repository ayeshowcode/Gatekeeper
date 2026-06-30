"""
Agent: given a question (and optionally a set of accepted lessons), calls retrieve()
to fetch relevant context chunks, then prompts the OpenAI model to answer using only
that context. Also exposes reflect(question, wrong_answer, context, correct_answer)
which drafts a general lesson from a failure. Temperature is fixed at 0 for
deterministic, reproducible pass/fail results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = OpenAI()
MODEL = "gpt-4o-mini"


def answer(question: str, lessons: str = "") -> str:
    """
    Answer a question using retrieved context.
    The lessons slot is empty for now — accepted lessons will be injected here on Day 4.
    """
    context_chunks = retrieve(question, k=3)
    context = "\n\n".join(context_chunks)

    lessons_block = f"\n\nLessons learned from past mistakes:\n{lessons}" if lessons.strip() else ""

    prompt = f"""You are a precise question-answering assistant.
Answer using ONLY the context provided below. Reply with the shortest possible exact answer — a name, number, date, or short phrase. Do not explain. Do not say "based on the context." Just the answer.
{lessons_block}

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


def reflect(question: str, wrong_answer: str, context: str, correct_answer: str) -> str:
    """
    Given a failed question, draft one short general lesson that would help next time.
    The lesson must be general — transferable to other questions, not a memorized answer.
    """
    prompt = f"""A question-answering agent got a question wrong. Your job is to write ONE short, general lesson that would help it answer similar questions correctly in the future.

The lesson must be GENERAL — it should apply to a class of questions, not just memorize this one answer. Do NOT write "the answer to this question is X." Write a strategy.

Failed question: {question}
Agent's wrong answer: {wrong_answer}
Correct answer: {correct_answer}
Retrieved context that was available:
{context}

Write one lesson (1-2 sentences, starting with "When"):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_question = "How many moons does Jupiter have?"
    result = answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer:   {result}")
