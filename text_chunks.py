def split_text(pages, filename, chunk_size=400, overlap=80):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": page_number,
                "document": filename
            })

            start += chunk_size - overlap

    return chunks