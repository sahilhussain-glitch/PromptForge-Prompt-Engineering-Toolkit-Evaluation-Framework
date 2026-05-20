"""Prompt evaluation: benchmark variants against ground truth test cases."""
import json
import statistics
from typing import Any
from engine.store import load_prompt
from engine.runner import run_prompt


def _check_match(output: str, expected: dict) -> bool:
    """Check if LLM output matches expected values (partial JSON match)."""
    try:
        parsed = json.loads(output)
        for key, val in expected.items():
            if str(parsed.get(key, "")).lower() != str(val).lower():
                return False
        return True
    except Exception:
        # Fallback: substring match
        return all(str(v).lower() in output.lower() for v in expected.values())


def run_benchmark(
    prompt_name: str,
    versions: list[str],
    test_cases: list[dict],
    runs_per_case: int = 3,
) -> dict:
    """
    Benchmark multiple versions of a prompt against test cases.

    test_cases: [{"input": {variables}, "expected": {key: value}}, ...]
    Returns per-version metrics: accuracy, avg_tokens, consistency_score, avg_latency_ms
    """
    results = {}

    for version in versions:
        try:
            pt = load_prompt(prompt_name, version)
        except FileNotFoundError:
            results[version] = {"error": f"Version {version} not found"}
            continue

        version_results = []
        for case in test_cases:
            case_outputs = []
            correct = 0
            token_counts = []
            latencies = []

            for _ in range(runs_per_case):
                rendered = pt.render(case["input"])
                result = run_prompt(rendered, json_mode=bool(pt.output_schema))
                output = result["output"]
                case_outputs.append(output)
                token_counts.append(result["tokens_used"])
                latencies.append(result["latency_ms"])

                if _check_match(output, case.get("expected", {})):
                    correct += 1

            # Consistency: fraction of runs that match the majority output
            from collections import Counter
            most_common = Counter(case_outputs).most_common(1)[0][1]
            consistency = most_common / runs_per_case

            version_results.append({
                "input": case["input"],
                "expected": case.get("expected", {}),
                "accuracy": correct / runs_per_case,
                "consistency": consistency,
                "avg_tokens": sum(token_counts) / len(token_counts),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "outputs": case_outputs,
            })

        all_accuracies = [r["accuracy"] for r in version_results]
        all_tokens = [r["avg_tokens"] for r in version_results]
        all_latencies = [r["avg_latency_ms"] for r in version_results]
        all_consistency = [r["consistency"] for r in version_results]

        results[version] = {
            "overall_accuracy": round(sum(all_accuracies) / len(all_accuracies), 3),
            "avg_tokens": round(sum(all_tokens) / len(all_tokens), 1),
            "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 1),
            "consistency_score": round(sum(all_consistency) / len(all_consistency), 3),
            "test_cases": len(test_cases),
            "per_case": version_results,
        }

    return {
        "prompt_name": prompt_name,
        "versions_evaluated": versions,
        "results": results,
        "winner": max(
            (v for v in versions if "error" not in results.get(v, {})),
            key=lambda v: results[v]["overall_accuracy"],
            default=None,
        ),
    }
