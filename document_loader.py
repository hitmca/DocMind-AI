from pypdf import PdfReader


def load_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_num + 1,
                "text": text
            })

    return pages