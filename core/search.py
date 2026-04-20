import os
from tavily import TavilyClient as Tavily
from dotenv import load_dotenv

load_dotenv()

client=Tavily(api_key=os.environ.get("TAVILY_API_KEY"))

def search_web(query: str, max_results: int = 5) -> list[dict]:
    response=client.search(query=query, max_results=max_results)
    return response.get("results", [])

if __name__=="__main__":
    results = search_web("ReAct prompting framework LLM agents")
    for r in results:
        print(r['title'])
        print(r['url'])
        print(r['content'][:150])
        print("---")
