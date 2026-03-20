from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L12-v2")

def create_embeddings(chunks):

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts)

    return embeddings