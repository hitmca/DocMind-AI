from llm_engine import generate_answer

def compare_docs(chunks):

    doc1 = " ".join([c["text"] for c in chunks[:5]])
    doc2 = " ".join([c["text"] for c in chunks[5:10]])

    prompt = f"""
Compare the following two documents.

Document 1:
{doc1}

Document 2:
{doc2}

Explain similarities and differences.
"""

    result = generate_answer(doc1 + doc2, prompt)

    return result