from core.search import search_web
from core.scraper import scrape_url
from core.vectorstore import add_documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def research_sub_question(sub_question: str, collection_name: str) -> int:
    """
    For a single sub-question:
    1. Search the web
    2. Scrape each result URL
    3. Chunk the text
    4. Embed and store in ChromaDB
    Returns the number of chunks stored.
    """
    print(f"\n🔍 Researching: {sub_question}")
    
    results = search_web(sub_question, max_results=5)
    if not results:
        print("  No search results found.")
        return 0

    all_chunks = []
    all_metadatas = []

    for result in results:
        url = result.get("url", "")
        print(f"  Scraping: {url}")
        
        text = scrape_url(url)
        if not text:
            print(f"  Could not extract content from {url}, skipping.")
            continue

        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": url,
                "sub_question": sub_question
            })

    if all_chunks:
        add_documents(collection_name, all_chunks, all_metadatas)
        print(f"  Stored {len(all_chunks)} chunks from {sub_question[:50]}...")

    return len(all_chunks)


def research_all(sub_questions: list[str], collection_name: str) -> None:
    """
    Runs research for every sub-question.
    """
    total_chunks = 0
    for sub_question in sub_questions:
        count = research_sub_question(sub_question, collection_name)
        total_chunks += count
    
    print(f"\n✅ Research complete. Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    from agents.planner import plan_query
    
    query = "What are the latest approaches to reducing hallucination in LLMs?"
    sub_questions = plan_query(query)
    
    research_all(sub_questions, collection_name="research_test")