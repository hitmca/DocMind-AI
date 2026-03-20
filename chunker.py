def create_chunks(pages):

    chunks = []

    for page in pages:

        text = page["text"]

        paragraphs = text.split("\n\n")

        for p in paragraphs:

            p = p.strip()

            if len(p) < 40:
                continue

            # remove headings
            if len(p.split()) < 3:
                continue

            chunks.append({
                "text": p,
                "page": page["page"]
            })

    return chunks