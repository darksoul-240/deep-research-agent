import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import call_llm

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. 
Your job is to break down a complex research query into 3-5 focused sub-questions.

Rules:
- Each sub-question must be specific and self-contained
- Sub-questions should cover different aspects of the main query
- Output ONLY a numbered list, nothing else
- No preamble, no explanation, just the list

Example:
Query: How is AI changing drug discovery?
1. What machine learning models are used in protein structure prediction?
2. How is generative AI being applied to molecular design?
3. What are the current clinical trial success rates for AI-discovered drugs?
4. What are the main regulatory challenges for AI-developed pharmaceuticals?"""


def plan_query(query: str) -> list[str]:
    """
    Takes a research query, returns a list of focused sub-questions.
    """
    prompt = f"Query: {query}"
    
    response = call_llm(
        prompt=prompt,
        system=PLANNER_SYSTEM_PROMPT,
        temperature=0.4
    )
    
    # Parse the numbered list into a clean Python list
    lines = response.strip().split("\n")
    sub_questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip the number and dot from the start e.g. "1. What is..."
        if line[0].isdigit() and len(line) > 2:
            line = line.split(".", 1)[-1].strip()
        sub_questions.append(line)
    
    return sub_questions


if __name__ == "__main__":
    query = "What are the latest approaches to reducing hallucination in LLMs?"
    sub_questions = plan_query(query)
    
    print(f"Original query: {query}\n")
    print("Sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")