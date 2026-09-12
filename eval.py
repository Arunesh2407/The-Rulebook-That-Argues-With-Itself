import os
import sys
import json
import time
from tabulate import tabulate
from src.parser import parse_all_documents
from src.retriever import Retriever
from src.verifier import Verifier

def run_evaluation(dataset_path: str = "eval_dataset.json", output_path: str = "eval_results.json"):
    print("Initializing pipeline components...")
    chunks = parse_all_documents()
    print(f"Parsed {len(chunks)} total passages.")
    
    retriever = Retriever()
    retriever.build_index(chunks, force_rebuild=True)
    print("ChromaDB index successfully built.")

    verifier = Verifier()

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    total_queries = len(eval_data)
    print(f"Starting evaluation of {total_queries} queries (pacing at 0.5s per call)...")

    results = []
    correct_states = 0
    
    contradiction_total = 0
    contradiction_correct = 0
    
    unanswered_total = 0
    unanswered_correct = 0

    answers_total = 0
    answers_correct = 0

    start_time = time.time()

    for idx, item in enumerate(eval_data, 1):
        q_id = item["id"]
        query = item["query"]
        expected_state = item["expected_state"].upper()

        retrieved_chunks = retriever.retrieve(query, top_k=5)
        
        verification = verifier.verify(query, retrieved_chunks)
        predicted_state = verification["state"].upper()
        
        is_correct = (predicted_state == expected_state)
        if is_correct:
            correct_states += 1

        if expected_state == "CONTRADICTION":
            contradiction_total += 1
            if is_correct:
                contradiction_correct += 1
        elif expected_state == "UNANSWERED":
            unanswered_total += 1
            if is_correct:
                unanswered_correct += 1
        elif expected_state == "ANSWERS":
            answers_total += 1
            if is_correct:
                answers_correct += 1

        res_entry = {
            "id": q_id,
            "query": query,
            "expected_state": expected_state,
            "predicted_state": predicted_state,
            "is_correct": is_correct,
            "answer": verification["answer"],
            "citations": verification["citations"],
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "file": c["file"],
                    "section": c["section"],
                    "page": c["page"]
                } for c in retrieved_chunks
            ]
        }
        results.append(res_entry)

        status_str = "CORRECT" if is_correct else "MISMATCH"
        print(f"[{idx:02d}/{total_queries:02d}] Query #{q_id} | Expected: {expected_state:<13} | Predicted: {predicted_state:<13} | Status: {status_str}", flush=True)

        if idx < total_queries:
            time.sleep(0.5)

    elapsed_time = time.time() - start_time
    overall_accuracy = (correct_states / total_queries) * 100 if total_queries else 0.0
    contradiction_acc = (contradiction_correct / contradiction_total) * 100 if contradiction_total else 0.0
    unanswered_rate = (unanswered_correct / unanswered_total) * 100 if unanswered_total else 0.0
    answers_acc = (answers_correct / answers_total) * 100 if answers_total else 0.0

    summary_metrics = {
        "total_queries": total_queries,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "contradiction_accuracy": f"{contradiction_correct}/{contradiction_total}",
        "contradiction_accuracy_pct": round(contradiction_acc, 2),
        "unanswerable_rejection_rate": f"{unanswered_correct}/{unanswered_total}",
        "unanswerable_rejection_rate_pct": round(unanswered_rate, 2),
        "answers_accuracy": f"{answers_correct}/{answers_total}",
        "answers_accuracy_pct": round(answers_acc, 2),
        "elapsed_seconds": round(elapsed_time, 2)
    }

    eval_output = {
        "summary": summary_metrics,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    print("\n" + "="*70)
    print("                      EVALUATION SCORECARD                     ")
    print("="*70)

    scorecard_data = [
        ["Overall State Accuracy", f"{correct_states}/{total_queries}", f"{overall_accuracy:.2f}%"],
        ["Contradiction Detection Accuracy", f"{contradiction_correct}/{contradiction_total}", f"{contradiction_acc:.2f}% (Target: 3/3)"],
        ["Unanswerable Rejection Rate", f"{unanswered_correct}/{unanswered_total}", f"{unanswered_rate:.2f}% (Target: 25/25)"],
        ["Direct Answers Accuracy", f"{answers_correct}/{answers_total}", f"{answers_acc:.2f}%"]
    ]

    print(tabulate(scorecard_data, headers=["Metric", "Raw Count", "Score / Target"], tablefmt="grid"))
    print(f"\nFull evaluation report exported to: {output_path}")

if __name__ == "__main__":
    run_evaluation()
