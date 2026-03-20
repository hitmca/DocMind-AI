from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, passages):

    pairs = []

    for p in passages:
        pairs.append([query, p])

    scores = reranker.predict(pairs)

    ranked = list(zip(passages, scores))

    ranked.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked]