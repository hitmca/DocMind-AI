from sentence_transformers import SentenceTransformer
import numpy as np
import re

from reranker import rerank
from llm_engine import generate_answer
from hybrid_search import keyword_search

embed_model = SentenceTransformer("all-MiniLM-L12-v2")

# -----------------------------
# ENTITY STORAGE
# -----------------------------
document_entities = {}


# -----------------------------
# ENTITY EXTRACTION
# -----------------------------
def extract_entities(text):

    entities = {}

    # EMAIL
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email:
        entities["email"] = email[0]

    # NAME (top lines of document)
    lines = text.split("\n")[:20]

    for line in lines:
        line = line.strip()

        if len(line.split()) == 2:
            if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", line):
                entities["name"] = line
                break

    # DEGREE
    degree = re.findall(r"(M\.E\.|M\.Tech|B\.Tech|B\.E\.|Ph\.D)", text)

    if degree:
        entities["degree"] = degree[0]

    return entities


# -----------------------------
# REGISTER DOCUMENT
# -----------------------------
def register_document(doc_name, text):

    entities = extract_entities(text)
    document_entities[doc_name] = entities


# -----------------------------
# QUERY NORMALIZATION
# -----------------------------
def normalize_query(q):

    q = q.lower()

    mapping = {
        "candidate name": "name",
        "person name": "name",
        "my name": "name",
        "qualification": "degree",
        "highest qualification": "degree",
        "email id": "email"
    }

    for k, v in mapping.items():
        if k in q:
            q = v

    return q


# -----------------------------
# MAIN QA FUNCTION
# -----------------------------
def ask_question(query, index, chunks):

    if index is None:
        return {
            "answer": "Upload documents first.",
            "page": "-",
            "document": "-"
        }

    query = normalize_query(query)

    # -----------------------------
    # ENTITY LOOKUP
    # -----------------------------
    for doc in document_entities:

        ent = document_entities[doc]

        if query in ent:

            return {
                "answer": ent[query],
                "page": 1,
                "document": doc
            }

    # -----------------------------
    # VECTOR SEARCH
    # -----------------------------
    query_embedding = embed_model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k=10)

    vector_results = []

    for i in indices[0]:

        if i >= len(chunks):
            continue

        vector_results.append(chunks[i])

    # -----------------------------
    # KEYWORD SEARCH
    # -----------------------------
    keyword_results = keyword_search(query, chunks)

    # -----------------------------
    # MERGE RESULTS
    # -----------------------------
    combined = vector_results + keyword_results

    seen = set()
    merged_chunks = []

    for c in combined:

        text = c["text"]

        if text not in seen:
            seen.add(text)
            merged_chunks.append(c)

    retrieved_texts = [c["text"] for c in merged_chunks[:10]]
    retrieved_pages = [c["page"] for c in merged_chunks[:10]]
    retrieved_docs = [c["document"] for c in merged_chunks[:10]]

    if len(retrieved_texts) == 0:
        return {
            "answer": "No relevant information found.",
            "page": "-",
            "document": "-"
        }

    # -----------------------------
    # RERANK
    # -----------------------------
    ranked = rerank(query, retrieved_texts)

    context = "\n".join(ranked[:3])

    # -----------------------------
    # LLM ANSWER
    # -----------------------------
    answer = generate_answer(context, query)

    return {
        "answer": answer,
        "page": retrieved_pages[0],
        "document": retrieved_docs[0]
    }