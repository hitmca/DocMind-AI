from transformers import pipeline

classifier = pipeline(
"zero-shot-classification",
model="facebook/bart-large-mnli"
)

labels = [
"Technology",
"Space",
"Research",
"Finance",
"Policy",
"Education"
]

def classify_document(text):

    result = classifier(text[:500], labels)

    return result["labels"][:3]