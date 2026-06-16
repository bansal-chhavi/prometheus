#!/usr/bin/env python3
"""
Comprehensive API test suite for Prometheus Faithfulness Evaluator.
Tests various scenarios: health, faithful answers, hallucinations, edge cases.
"""
import requests
import json
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8001"

# Test scenarios
SCENARIOS = [
    {
        "name": "Health Check",
        "endpoint": "/health",
        "method": "GET",
        "payload": None,
    },
    {
        "name": "Faithful Answer (Supported)",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "What is the capital of France?",
            "rag_answer": "The capital of France is Paris.",
            "sources": [
                {
                    "text": "France is a country in Western Europe. Its capital city is Paris, located along the Seine River.",
                    "doc_id": "doc1",
                    "chunk_id": 1
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "Hallucinated Answer (Unsupported)",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "What is the capital of France?",
            "rag_answer": "The capital of France is Berlin and the population is 50 million.",
            "sources": [
                {
                    "text": "France is a country in Western Europe. Its capital city is Paris, located along the Seine River.",
                    "doc_id": "doc1",
                    "chunk_id": 1
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "Partially Hallucinated Answer",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "Tell me about Python programming.",
            "rag_answer": "Python is a programming language created by Guido van Rossum in 1991. It supports quantum computing natively.",
            "sources": [
                {
                    "text": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991.",
                    "doc_id": "doc1",
                    "chunk_id": 1
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "Multiple Document Chunks",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "Who won the Nobel Prize in Physics in 2020?",
            "rag_answer": "Roger Penrose won half the Nobel Prize in Physics in 2020 for black hole discovery.",
            "sources": [
                {
                    "text": "The 2020 Nobel Prize in Physics was awarded to Roger Penrose, Reinhard Genzel, and Andrea Ghez.",
                    "doc_id": "doc1",
                    "chunk_id": 1
                },
                {
                    "text": "Roger Penrose received half the prize for his black hole work. Genzel and Ghez shared the other half.",
                    "doc_id": "doc1",
                    "chunk_id": 2
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "Complex Answer with Mixed Support",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "What are the primary renewable energy sources?",
            "rag_answer": "The primary renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass. Solar panels convert sunlight directly into electricity instantaneously.",
            "sources": [
                {
                    "text": "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass energy.",
                    "doc_id": "doc1",
                    "chunk_id": 1
                },
                {
                    "text": "Solar panels use the photovoltaic effect to convert sunlight into electricity.",
                    "doc_id": "doc1",
                    "chunk_id": 2
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "Empty Answer",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "What is X?",
            "rag_answer": "",
            "sources": [
                {
                    "text": "Some source text",
                    "doc_id": "doc1",
                    "chunk_id": 1
                }
            ],
            "task_type": "factual"
        }
    },
    {
        "name": "No Documents",
        "endpoint": "/evaluate",
        "method": "POST",
        "payload": {
            "query": "What is X?",
            "rag_answer": "X is something.",
            "sources": [],
            "task_type": "factual"
        }
    },
]


def run_test(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single test scenario.
    
    Args:
        scenario: Test scenario dict with endpoint, method, and payload
        
    Returns:
        Result dict with status, response code, and response body
    """
    url = BASE_URL + scenario["endpoint"]
    method = scenario["method"]
    payload = scenario["payload"]
    
    try:
        if method == "GET":
            resp = requests.get(url, timeout=30)
        else:  # POST
            resp = requests.post(url, json=payload, timeout=30)
        
        try:
            body = resp.json()
        except:
            body = resp.text
        
        return {
            "status": "pass" if 200 <= resp.status_code < 300 else "fail",
            "status_code": resp.status_code,
            "response": body
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "status_code": None,
            "response": "Connection refused - is the server running on 127.0.0.1:8000?"
        }
    except Exception as e:
        return {
            "status": "error",
            "status_code": None,
            "response": str(e)
        }


def print_result(scenario: Dict[str, Any], result: Dict[str, Any]):
    """Print a formatted test result."""
    name = scenario["name"]
    status = result["status"].upper()
    code = result["status_code"]
    
    # Status indicator
    if result["status"] == "pass":
        indicator = "✅"
    elif result["status"] == "fail":
        indicator = "❌"
    else:
        indicator = "⚠️"
    
    print(f"\n{indicator} [{status}] {name} (HTTP {code})")
    print("-" * 80)
    
    response = result["response"]
    if isinstance(response, dict):
        # For JSON responses, show key fields
        if "score" in response:
            print(f"  Faithfulness Score: {response.get('score', 'N/A'):.2f}")
        if "flags" in response:
            print(f"  Flags: {response.get('flags', [])}")
        if "details" in response and "hallucinated_spans" in response["details"]:
            spans = response["details"]["hallucinated_spans"]
            if spans:
                print(f"  Hallucinated Spans: {spans}")
            else:
                print(f"  Hallucinated Spans: (none detected)")
        if "error" in response or "detail" in response:
            err = response.get("error") or response.get("detail")
            print(f"  Error: {err}")
        # Show full response in compact JSON format
        print(f"\n  Full Response:\n{json.dumps(response, indent=2)}")
    else:
        print(f"  Response: {response}")


def main():
    """Main test runner."""
    print("=" * 80)
    print("PROMETHEUS FAITHFULNESS EVALUATOR - API TEST SUITE")
    print("=" * 80)
    print(f"\nTarget: {BASE_URL}")
    print(f"Total Scenarios: {len(SCENARIOS)}\n")
    
    results = []
    pass_count = 0
    fail_count = 0
    error_count = 0
    
    for scenario in SCENARIOS:
        print(f"\n[Running] {scenario['name']}...")
        result = run_test(scenario)
        results.append((scenario, result))
        
        print_result(scenario, result)
        
        if result["status"] == "pass":
            pass_count += 1
        elif result["status"] == "fail":
            fail_count += 1
        else:
            error_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed:  {pass_count}")
    print(f"Failed:  {fail_count}")
    print(f"Errors:  {error_count}")
    print(f"Total:   {len(SCENARIOS)}")
    
    if error_count > 0:
        print("\n⚠️  Errors detected. Ensure:")
        print("  - Server is running: .venv\\Scripts\\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000")
        print("  - LLM configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    return error_count == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
