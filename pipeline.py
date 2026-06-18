from agents.planner import plan_query
from agents.researcher import research_all
from agents.synthesizer import synthesize
from agents.critic import critique_report
import chromadb
import uuid

def run_pipeline(query: str) -> dict:
    """
    Full end-to-end research pipeline.
    Takes a query, returns a cited report with a faithfulness score.
    """
    # Unique collection per query so runs don't bleed into each other
    collection_name = f"research_{uuid.uuid4().hex[:8]}"

    print(f"\n🚀 Starting Deep Research Agent")
    print(f"Query: {query}")
    print("="*60)

    # Step 1 — Plan
    print("\n📋 Step 1: Planning sub-questions...")
    sub_questions = plan_query(query)
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    # Step 2 — Research
    print("\n🔍 Step 2: Researching each sub-question...")
    research_all(sub_questions, collection_name)

    # Step 3 — Synthesize
    print("\n✍️  Step 3: Synthesizing report...")
    report = synthesize(query, collection_name)

    # Step 4 — Critique
    print("\n🔎 Step 4: Critiquing faithfulness...")
    critique = critique_report(report, query, collection_name)

    return {
        "query": query,
        "sub_questions": sub_questions,
        "report": report,
        "faithfulness_score": critique["faithfulness_score"],
        "critic_feedback": critique["feedback"],
        "collection_name": collection_name
    }


if __name__ == "__main__":
    query = "How does reinforcement learning work?"
    result = run_pipeline(query)

    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result["report"])
    print("="*60)
    print(f"\n✅ Faithfulness Score: {result['faithfulness_score']}")