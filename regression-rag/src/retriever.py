"""
Retriever: loads documents from data/docs/, chunks them, embeds them with OpenAI,
and stores them in a ChromaDB vector store. Exposes retrieve(question, k=3) which
returns the k most semantically similar chunks for a given question.
"""
