import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")


def build_graph(chunks):

    G = nx.Graph()

    for chunk in chunks:

        doc = nlp(chunk["text"])

        entities = []

        for ent in doc.ents:

            if ent.label_ in ["PERSON","ORG","GPE","LOC"]:
                entities.append(ent.text)

        # connect entities inside same chunk
        for i in range(len(entities)-1):
            G.add_edge(entities[i], entities[i+1])

    nodes = [{"id": n, "label": n} for n in G.nodes()]
    edges = [{"from": u, "to": v} for u, v in G.edges()]

    return {
        "nodes": nodes,
        "edges": edges
    }