"""
Agent: given a question (and optionally a set of accepted lessons), calls retrieve()
to fetch relevant context chunks, then prompts the OpenAI model to answer using only
that context. Also exposes reflect(question, wrong_answer, context, correct_answer)
which drafts a general lesson from a failure. Temperature is fixed at 0 for
deterministic, reproducible pass/fail results.
"""
