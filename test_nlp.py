from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How much can I save next month?",
    "What will my expenses be next month?"
]

embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings)