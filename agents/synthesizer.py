from core.llm import call_llm
from core.vectorstore import query_collection

SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. You will be given:
1. A research query
2. A set of retrieved context chunks with their sources

Your job is to write a structured, cited research report that directly answers the query.

Rules:
- Use ONLY information from the provided context
- Cite sources inline using [Source: URL] format
- Structure the report with clear sections
- If the context doesn't cover something, say so explicitly
- Do not hallucinate or add information not present in the context
- End with a 'Sources' section listing all URLs used"""


def synthesize(query: str, collection_name: str, n_results: int = 10) -> str:
    """
    Retrieves relevant chunks and generates a cited research report.
    """
    print(f"\n📚 Retrieving relevant chunks for: {query[:60]}...")
    
    chunks = query_collection(collection_name, query, n_results=n_results)
    
    if not chunks:
        return "No relevant context found in the vector store."
    
    # Build context block for the LLM
    context_block = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        content = chunk["content"]
        context_block += f"\n[Chunk {i}] Source: {source}\n{content}\n"
    
    prompt = f"""Research Query: {query}

Retrieved Context:
{context_block}

Write a comprehensive cited research report answering the query using only the context above."""

    print("✍️  Generating report...")
    report = call_llm(
        prompt=prompt,
        system=SYNTHESIZER_SYSTEM_PROMPT,
        temperature=0.3
    )
    
    return report


if __name__ == "__main__":
    query = "What are the latest approaches to reducing hallucination in LLMs?"
    report = synthesize(query, collection_name="research_test")
    print("\n" + "="*60)
    print(report)
    print("="*60)