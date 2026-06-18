from core.llm import call_llm
from core.vectorstore import query_collection

CRITIC_SYSTEM_PROMPT = """You are a strict fact-checker for AI-generated research reports.
You will be given a research report and the source context it was generated from.

Your job is to evaluate the report's faithfulness to the source material.

For each major claim in the report:
- Check if it is supported by the provided context
- Mark it as SUPPORTED, UNSUPPORTED, or PARTIAL

Then output:
1. A list of claims with their support status
2. An overall faithfulness score from 0.0 to 1.0
3. A brief verdict

Output format:
CLAIMS:
- [SUPPORTED/UNSUPPORTED/PARTIAL] <claim>

FAITHFULNESS SCORE: <0.0 to 1.0>
VERDICT: <one sentence summary>"""


def critique_report(report: str, query: str, collection_name: str) -> dict:
    """
    Evaluates a generated report against retrieved source context.
    Returns faithfulness score and detailed feedback.
    """
    print("\n🔎 Critiquing report faithfulness...")

    # Pull context from vectorstore to fact-check against
    chunks = query_collection(collection_name, query, n_results=15)
    context_block = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "Unknown")
        context_block += f"\n[Chunk {i}] Source: {source}\n{chunk['content']}\n"

    prompt = f"""Research Report to Evaluate:
{report}

Source Context:
{context_block}

Evaluate the report's faithfulness to the source context."""

    response = call_llm(
        prompt=prompt,
        system=CRITIC_SYSTEM_PROMPT,
        temperature=0.1  # Low temp — we want consistent, precise judgment
    )

    # Parse faithfulness score
    score = None
    for line in response.split("\n"):
        if "FAITHFULNESS SCORE:" in line:
            try:
                score = float(line.split(":")[-1].strip())
            except ValueError:
                score = None

    return {
        "feedback": response,
        "faithfulness_score": score
    }


if __name__ == "__main__":
    from agents.planner import plan_query
    from agents.synthesizer import synthesize

    query = "What are the latest approaches to reducing hallucination in LLMs?"
    collection_name = "research_test"

    # Use already stored chunks from previous run, just synthesize + critique
    report = synthesize(query, collection_name)
    result = critique_report(report, query, collection_name)

    print("\n" + "="*60)
    print("CRITIC FEEDBACK:")
    print(result["feedback"])
    print("="*60)
    print(f"\nFaithfulness Score: {result['faithfulness_score']}")