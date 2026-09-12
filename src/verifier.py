import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

SYSTEM_INSTRUCTION = """You are an academic regulations verification assistant for university policy documents.
Your task is to analyze a user query against retrieved candidate passages and classify the response state into exactly one of three states:

1. "ANSWERS":
   - The retrieved passages contain clear, direct, and explicit textual evidence that directly answers the user query.
   - There are NO conflicting or incompatible statements among the retrieved passages.
   - Provide a direct answer synthesized from the passages.
   - Provide exact citations with "file", "section", and "excerpt" (verbatim quote from the text).

2. "CONTRADICTION":
   - Two or more retrieved passages make conflicting, incompatible, or contradictory assertions regarding the query topic (e.g., one rule states 75% minimum attendance with no waiver, while another section permits instructor reduction to 65%; or one rule assigns prerequisite waiver authority exclusively to the Dean, while another section gives final authority to the Department Head; or one rule requires medical hospital notes within 48 hours while another allows Campus Health Center validation within 7 days).
   - In your answer, explain clearly the conflicting statements and why they contradict.
   - Provide citations for ALL conflicting passages (at least two citations).

3. "UNANSWERED":
   - The retrieved passages do NOT contain explicit textual proof to answer the user query.
   - The passages are missing the necessary facts, or are tangential, irrelevant, or incomplete.
   - DO NOT extrapolate, infer, guess, or use external knowledge beyond the provided text.
   - If the explicit answer is absent in the provided passages, state MUST be "UNANSWERED".
   - Return an empty list [] for citations.

You MUST respond strictly with valid JSON conforming to the following structure:
{
  "state": "ANSWERS | UNANSWERED | CONTRADICTION",
  "answer": "string summary",
  "citations": [
    {
      "file": "string filename (e.g. handbook.md, academic_regulations.pdf, fee_schedule.csv)",
      "section": "string section title",
      "excerpt": "exact string quote"
    }
  ]
}
"""

RESOLUTION_INSTRUCTION = """You are an expert university policy officer and legal analyst.
Your task is to analyze two or more conflicting policy provisions identified in a university rulebook and produce a structured, reconciled policy proposal that resolves the contradiction cleanly.

You MUST respond strictly with valid JSON conforming to the following structure:
{
  "reconciled_rule": "Detailed explanation of how the policy conflict should be unified and resolved",
  "recommended_hierarchy": "Clear declaration of which institutional authority or policy timeline takes precedence and why",
  "proposed_redraft": "Exact verbatim redrafted policy clause to replace the conflicting sections in official policy manuals"
}
"""

class Verifier:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        
        self.client = Groq(api_key=api_key)
        self.models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b"
        ]

    def _call_groq_with_fallback(self, system_instruction: str, user_prompt: str):
        last_exception = None
        models_to_try = list(self.models)
        for model_name in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                return response
            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "not_found" in err_msg.lower() or "does not exist" in err_msg.lower():
                    if model_name in self.models and len(self.models) > 1:
                        self.models.remove(model_name)
                    last_exception = e
                    continue
                elif any(k in err_msg.lower() for k in ["429", "rate_limit", "quota"]):
                    print(f"[Model Fallback] {model_name} rate limit / quota exceeded ({err_msg[:60]}...). Trying fallback.", flush=True)
                    last_exception = e
                    continue
                else:
                    raise e
        if last_exception:
            raise last_exception

    def resolve_contradiction(self, query: str, citations: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        citations_text = []
        for idx, cit in enumerate(citations, 1):
            citations_text.append(f"Citation [{idx}] (File: {cit.get('file')}, Section: {cit.get('section')}):\n\"{cit.get('excerpt')}\"")
        c_str = "\n\n".join(citations_text)

        prompt = f"""USER QUERY / TOPIC:
{query}

CURRENT CONTRADICTION ANALYSIS:
{answer}

CONFLICTING CITATIONS:
{c_str}

Please generate the structured policy reconciliation proposal.
"""
        try:
            response = self._call_groq_with_fallback(RESOLUTION_INSTRUCTION, prompt)
            raw_text = response.choices[0].message.content.strip()
            parsed = json.loads(raw_text)
            return {
                "reconciled_rule": parsed.get("reconciled_rule", "Reconciled rule summary unavailable."),
                "recommended_hierarchy": parsed.get("recommended_hierarchy", "Hierarchy declaration unavailable."),
                "proposed_redraft": parsed.get("proposed_redraft", "Proposed redraft text unavailable.")
            }
        except Exception as e:
            return {
                "reconciled_rule": f"Resolution generation failed: {str(e)}",
                "recommended_hierarchy": "Error encountered during synthesis.",
                "proposed_redraft": "Unable to produce redraft."
            }

    def verify(self, query: str, retrieved_chunks: List[Dict[str, Any]], max_retries: int = 5) -> Dict[str, Any]:
        formatted_passages = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            file_info = chunk.get("file", "Unknown")
            sec_info = chunk.get("section", "Unknown")
            page_info = f", Page {chunk.get('page')}" if chunk.get("page") else ""
            formatted_passages.append(
                f"Passage [{idx}] (File: {file_info}, Section: {sec_info}{page_info}):\n{chunk.get('content')}"
            )
        
        passages_text = "\n\n--------------------\n\n".join(formatted_passages)

        user_prompt = f"""USER QUERY:
{query}

RETRIEVED CANDIDATE PASSAGES:
{passages_text}

Analyze the query against the candidate passages above and produce the final verification JSON object.
"""

        for attempt in range(max_retries):
            try:
                response = self._call_groq_with_fallback(SYSTEM_INSTRUCTION, user_prompt)
                raw_text = response.choices[0].message.content.strip()
                parsed = json.loads(raw_text)
                
                state = parsed.get("state", "UNANSWERED").upper()
                if state not in ["ANSWERS", "CONTRADICTION", "UNANSWERED"]:
                    state = "UNANSWERED"
                
                answer = parsed.get("answer", "")
                citations = parsed.get("citations", [])
                if not isinstance(citations, list):
                    citations = []

                return {
                    "state": state,
                    "answer": answer,
                    "citations": citations
                }
            except Exception as e:
                err_str = str(e)
                print(f"[Verifier Attempt {attempt+1}/{max_retries}] Error encountered: {err_str}", flush=True)
                if attempt == max_retries - 1:
                    return {
                        "state": "UNANSWERED",
                        "answer": f"Verification failed due to error: {str(e)}",
                        "citations": []
                    }
                
                # Check for rate limit / quota 429
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait_time = 5.0
                    import re
                    match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_str, re.IGNORECASE)
                    if match:
                        wait_time = float(match.group(1)) + 1.0
                    print(f"Rate limit exceeded. Waiting {wait_time:.1f}s before retry...", flush=True)
                    time.sleep(wait_time)
                else:
                    time.sleep(1 * (attempt + 1))

if __name__ == "__main__":
    from src.retriever import Retriever
    from src.parser import parse_all_documents
    
    chunks = parse_all_documents()
    retriever = Retriever()
    retriever.build_index(chunks)
    
    verifier = Verifier()
    
    test_query = "Who controls the official academic record when an informal advising comment differs from registrar records?"
    retrieved = retriever.retrieve(test_query, top_k=5)
    result = verifier.verify(test_query, retrieved)
    print("Verification Result:\n", json.dumps(result, indent=2))

