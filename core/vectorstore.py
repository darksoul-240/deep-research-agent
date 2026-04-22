import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from core.embedder import embed_texts

client = chromadb.PersistentClient(path=".chromadb")

def get_or_create_collection(name: str) -> chromadb.Collection:
    return client.get_or_create_collection(name=name)


def add_documents(collection_name: str, docs: list[str], metadatas: list[dict]) -> None:
    """
    Embeds and stores documents in ChromaDB.
    """
    collection = get_or_create_collection(collection_name)
    embeddings = embed_texts(docs)
    ids = [str(i) for i in range(len(docs))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas
    )


def query_collection(collection_name: str, query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search — returns top n_results chunks most similar to query.
    """
    collection = get_or_create_collection(collection_name)
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append({"content": doc, "metadata": meta})
    return output


if __name__ == "__main__":
    docs = [
        "ReAct combines reasoning and acting in language models.",
        "ChromaDB is a vector database for storing embeddings.",
        "RAGAS is a framework for evaluating RAG pipelines.",
        "Trafilatura extracts clean text from web pages.",
    ]
    metadatas = [{"source": f"test_{i}"} for i in range(len(docs))]

    add_documents("test_collection", docs, metadatas)
    
    results = query_collection("test_collection", "how do we evaluate RAG systems?")
    for r in results:
        print(r["content"])
        print("---")