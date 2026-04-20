import os 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client=Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm(
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
)->str:
    response=client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user", "content":prompt}
        ],
    temperature=temperature
    )
    return response.choices[0].message.content

