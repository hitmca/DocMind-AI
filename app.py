from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import spacy

from document_loader import load_pdf
from text_chunks import split_text
from embeddings import create_embeddings
from vector_store import create_vector_store
from qa_engine import ask_question
from llm_engine import generate_answer

app = FastAPI()

templates = Jinja2Templates(directory="templates")

nlp = spacy.load("en_core_web_sm")

chunks = []
index = None


# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ---------------------------------------------------
# UPLOAD DOCUMENT
# ---------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    pages = load_pdf(path)

    global chunks

    new_chunks = split_text(pages, file.filename)

    chunks.extend(new_chunks)

    embeddings = create_embeddings(chunks)

    global index
    index = create_vector_store(embeddings)

    return {"message": "document uploaded"}


# ---------------------------------------------------
# QUESTION ANSWERING
# ---------------------------------------------------

@app.post("/ask")
async def ask(question: str = Form(...)):

    result = ask_question(question, index, chunks)

    return JSONResponse(result)


# ---------------------------------------------------
# LIST DOCUMENTS
# ---------------------------------------------------

@app.get("/documents")
def documents():

    docs = list(set([c["document"] for c in chunks]))

    return {"documents": docs}


# ---------------------------------------------------
# KNOWLEDGE GRAPH
# ---------------------------------------------------

@app.get("/graph")
def graph():

    nodes = []
    edges = []

    entity_map = {}
    idx = 0

    for c in chunks[:30]:

        doc = nlp(c["text"])

        entities = []

        for ent in doc.ents:

            if ent.text not in entity_map:

                entity_map[ent.text] = idx

                nodes.append({
                    "id": idx,
                    "label": ent.text
                })

                idx += 1

            entities.append(entity_map[ent.text])

        for i in range(len(entities)-1):

            edges.append({
                "from": entities[i],
                "to": entities[i+1]
            })

    return {
        "nodes": nodes,
        "edges": edges
    }


# ---------------------------------------------------
# AI INSIGHTS
# ---------------------------------------------------

@app.get("/insights")
def insights():

    entity_count = 0

    for c in chunks:

        doc = nlp(c["text"])

        entity_count += len(doc.ents)

    docs = list(set([c["document"] for c in chunks]))

    return {
        "chunks": len(chunks),
        "entities": entity_count,
        "documents": len(docs)
    }


# ---------------------------------------------------
# DOCUMENT TAGS
# ---------------------------------------------------

@app.get("/tags")
def tags():

    keywords = set()

    for c in chunks[:50]:

        doc = nlp(c["text"])

        for token in doc:

            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 4:

                keywords.add(token.text)

    return {
        "tags": list(keywords)[:12]
    }


# ---------------------------------------------------
# AI DOCUMENT COPILOT
# ---------------------------------------------------

@app.get("/copilot")
def copilot(mode: str):

    text = " ".join([c["text"] for c in chunks[:20]])

    if len(text) == 0:

        return {"result": "No documents uploaded yet."}

    if mode == "summary":

        result = generate_answer(
            text,
            "Summarize all uploaded documents."
        )

    elif mode == "analysis":

        result = generate_answer(
            text,
            "Analyze the documents and extract key insights."
        )

    else:

        result = generate_answer(
            text,
            "Explain the important information in the documents."
        )

    return {"result": result}

@app.get("/compare")
def compare():

    docs = {}

    for c in chunks:

        doc = c["document"]

        if doc not in docs:
            docs[doc] = []

        docs[doc].append(c["text"])

    result = ""

    for d in docs:

        text = " ".join(docs[d][:10])

        summary = generate_answer(
            text,
            f"Summarize the document {d}"
        )

        result += f"<b>{d}</b><br>{summary}<br><br>"

    return {"result": result}

@app.get("/timeline")
def timeline():

    text = " ".join([c["text"] for c in chunks[:20]])

    result = generate_answer(
        text,
        "Extract important events and dates from the documents."
    )

    return {"result": result}

@app.get("/relationships")
def relationships():

    text = " ".join([c["text"] for c in chunks[:20]])

    result = generate_answer(
        text,
        "Identify relationships between people, organizations, and locations."
    )

    return {"result": result}

@app.get("/report")
def report():

    text = " ".join([c["text"] for c in chunks[:30]])

    result = generate_answer(
        text,
        "Generate a structured report from the documents."
    )

    return {"result": result}

