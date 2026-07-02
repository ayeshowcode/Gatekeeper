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
from insights import load_insights

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = OpenAI()
MODEL = "gpt-4o-mini"


def answer(question: str, lessons: str = None) -> str:
    """
    Answer a question using retrieved context.
    If lessons is left as None, the accumulated insight store is loaded
    automatically -- this is what real callers get. Passing an explicit
    string (including "") overrides that and skips auto-loading; the gate
    and the harness's baseline runs rely on this to test one lesson (or no
    lesson at all) in isolation from everything already in the store.
    """
    if lessons is None:
        lessons = load_insights()

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
    Given a failed question, produce a specific, actionable lesson naming the exact
    error pattern and the concrete corrective rule. Generic advice is forbidden.
    """
    prompt = f"""A RAG-based question-answering agent gave a wrong answer. Diagnose the specific error and write one precise, actionable lesson to prevent this class of mistake.

The lesson MUST:
- Name the concrete failure pattern (e.g. "when the context contains two similar values for two different entities and the question identifies the entity indirectly...")
- State a specific corrective rule (e.g. "...first resolve which entity the question refers to, then extract only that entity's value, not a neighbouring entity's value from the same chunk")
- Be 2-3 sentences maximum

The lesson MUST NOT:
- Use vague advice like "verify the context", "cross-reference sources", or "double-check the answer"
- Mention the specific answer values from this question (the lesson must apply to a class of similar questions, not memorise this one case)

Failed question: {question}
Agent's wrong answer: {wrong_answer}
Correct answer: {correct_answer}
Context the agent retrieved:
{context}

Write the lesson (start with "When"):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_question = "How many moons does Jupiter have?"
    result = answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer:   {result}")
