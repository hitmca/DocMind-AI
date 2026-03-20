from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def generate_answer(context, question):

    prompt = f"""
You are an AI assistant answering questions from documents.

Rules:
- Use ONLY the provided document context
- Do NOT invent answers
- Return full names when asked
- Return the most relevant factual answer

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    result = generator(
        prompt,
        max_length=180,
        do_sample=False
    )

    return result[0]["generated_text"]