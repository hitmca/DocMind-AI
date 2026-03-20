import re

def keyword_search(query, chunks):

    query_words = query.lower().split()

    results = []

    for chunk in chunks:

        text = chunk["text"].lower()

        score = 0

        for word in query_words:
            if word in text:
                score += 1

        if score > 0:
            results.append((chunk, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[:5]]