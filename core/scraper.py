import trafilatura

def scrape_url(url: str)-> str|None:
    downloaded=trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text=trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return text

if __name__=="__main__":
    url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    text=scrape_url(url)
    if text:
        print(text[:500])
    else:
        print("Failed to scrape the URL.")
