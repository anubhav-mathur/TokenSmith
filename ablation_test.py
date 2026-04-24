"""
Ablation test: source routing ON vs OFF
Run with API server active: python ablation_test.py
"""
import requests
import json

API = "http://localhost:8000/api/chat"

QUERIES = [
    {
        "query": "what does the textbook say about B+ trees",
        "expected_doc_type": "document"
    },
    {
        "query": "what does the textbook say about ACID properties",
        "expected_doc_type": "document"
    },
    {
        "query": "what does the textbook say about deadlock handling",
        "expected_doc_type": "document"
    },
    {
        "query": "what does the textbook say about query optimization",
        "expected_doc_type": "document"
    },
    {
        "query": "what does the textbook say about transaction isolation",
        "expected_doc_type": "document"
    },
]

def source_purity(sources, expected_doc_type):
    if not sources:
        return 0.0
    correct = sum(1 for s in sources if s.get("doc_type") == expected_doc_type)
    return round(correct / len(sources), 3)

def run_query(query):
    resp = requests.post(API, json={"query": query}, timeout=120)
    return resp.json()

print(f"{'Query':<55} {'Purity'}")
print("-" * 65)

purities = []
for item in QUERIES:
    result = run_query(item["query"])
    sources = result.get("sources", [])
    purity = source_purity(sources, item["expected_doc_type"])
    purities.append(purity)
    short = item["query"][:53]
    print(f"{short:<55} {purity:.3f}")

avg = sum(purities) / len(purities) if purities else 0
print("-" * 65)
print(f"{'Average purity':<55} {avg:.3f}")