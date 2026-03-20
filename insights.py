import spacy

nlp = spacy.load("en_core_web_sm")

def generate_insights(chunks):

    total_chunks = len(chunks)

    entities = []

    for chunk in chunks:

        doc = nlp(chunk["text"])

        for ent in doc.ents:
            entities.append(ent.label_)

    insights = {
        "chunks":total_chunks,
        "entities":len(entities),
        "documents":"Multiple",
    }

    return insights