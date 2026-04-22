from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of strings, returns a list of embedding vectors.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


if __name__ == "__main__":
    texts = ["ReAct is a prompting framework", "ChromaDB stores vectors"]
    embeddings = embed_texts(texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")