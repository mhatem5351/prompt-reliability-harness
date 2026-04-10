#!/usr/bin/env python3
"""Prompt reliability evaluation harness.

Runs test cases against an OpenAI model, measures response consistency
across prompt variants, and reports scoring metrics.

Usage:
    python run_eval.py --test-file test_cases.json --runs 3 --model gpt-4o-mini
    python run_eval.py --runs 5 --output results.json --scorers regex,semantic
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from scoring import (
    score_exact,
    score_regex,
    score_semantic,
    compute_consistency,
    compute_flip_rate,
)


def call_llm(client: OpenAI, model: str, prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI chat completions and return the response text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def run_variant(client: OpenAI, model: str, variant: dict, num_runs: int, delay: float) -> list[str]:
    """Run a single variant N times, return list of responses."""
    responses = []
    for i in range(num_runs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = call_llm(
                    client, model,
                    variant["prompt"],
                    variant.get("system_prompt"),
                )
                responses.append(resp)
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"    [rate-limited] Retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"    [error] Run {i+1} failed: {e}")
                responses.append(f"[ERROR: {e}]")
                break
        if delay > 0:
            time.sleep(delay)
    return responses


def score_responses(responses: list[str], expected: dict, scorers: list[str], client: OpenAI = None) -> dict:
    """Score a list of responses against expected values. Returns per-scorer average scores."""
    results = {}
    for scorer in scorers:
        scores = []
        for resp in responses:
            if scorer == "exact":
                s = score_exact(resp, expected.get("exact"))
            elif scorer == "regex":
                s = score_regex(resp, expected.get("regex"))
            elif scorer == "semantic":
                if client and expected.get("semantic_anchor"):
                    s = score_semantic(resp, expected["semantic_anchor"], client)
                else:
                    s = -1.0
            else:
                continue
            if s >= 0:
                scores.append(s)
        results[scorer] = sum(scores) / len(scores) if scores else -1.0
    return results


def run_evaluation(test_cases: list, client: OpenAI, model: str, num_runs: int,
                   delay: float, scorers_filter: list[str] = None) -> list[dict]:
    """Run all test cases and return detailed results."""
    all_results = []

    for case in test_cases:
        case_id = case["id"]
        case_type = case["type"]
        expected = case["expected"]
        case_scorers = scorers_filter or case.get("scoring", ["regex"])

        print(f"\n[{case_id}] ({case_type}) — {len(case['variants'])} variants x {num_runs} runs")

        variant_responses = {}
        variant_scores = {}
        baseline_responses = None

        for variant in case["variants"]:
            vid = variant["variant_id"]
            print(f"  variant: {vid}", end="", flush=True)

            responses = run_variant(client, model, variant, num_runs, delay)
            variant_responses[vid] = responses

            if variant.get("is_baseline"):
                baseline_responses = responses

            scores = score_responses(responses, expected, case_scorers, client)
            variant_scores[vid] = scores
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items() if v >= 0)
            print(f" — {score_str}")

        # Compute consistency metrics
        all_responses = []
        for resps in variant_responses.values():
            all_responses.extend(resps)

        consistency = compute_consistency(all_responses)

        # Compute flip rates for non-baseline variants
        flip_rates = {}
        if baseline_responses:
            for vid, resps in variant_responses.items():
                is_baseline = any(v.get("is_baseline") and v["variant_id"] == vid for v in case["variants"])
                if not is_baseline:
                    flip_rates[vid] = compute_flip_rate(baseline_responses, resps)

        avg_flip_rate = sum(flip_rates.values()) / len(flip_rates) if flip_rates else 0.0

        case_result = {
            "id": case_id,
            "type": case_type,
            "category": case.get("category", ""),
            "consistency": consistency,
            "flip_rate": avg_flip_rate,
            "variant_responses": variant_responses,
            "variant_scores": variant_scores,
            "flip_rates": flip_rates,
        }
        all_results.append(case_result)

    return all_results


def print_summary(results: list[dict]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 95)
    print(f"{'Case ID':<35} {'Type':<14} {'Consistency':>12} {'Regex':>8} {'Flip Rate':>10} {'Notes'}")
    print("-" * 95)

    total_consistency = 0
    total_regex = 0
    regex_count = 0

    for r in results:
        # Get average regex score across variants
        regex_scores = []
        for vid, scores in r["variant_scores"].items():
            if "regex" in scores and scores["regex"] >= 0:
                regex_scores.append(scores["regex"])
        avg_regex = sum(regex_scores) / len(regex_scores) if regex_scores else -1.0

        notes = ""
        if r["type"] == "perturbation" and r["flip_rate"] > 0:
            notes = f"flip_rate={r['flip_rate']:.2f}"
        elif r["type"] == "invariance" and r["consistency"] < 1.0:
            notes = "INCONSISTENT"

        regex_str = f"{avg_regex:.2f}" if avg_regex >= 0 else "n/a"
        flip_str = f"{r['flip_rate']:.2f}" if r["type"] == "perturbation" else ""

        print(f"{r['id']:<35} {r['type']:<14} {r['consistency']:>12.2f} {regex_str:>8} {flip_str:>10} {notes}")

        total_consistency += r["consistency"]
        if avg_regex >= 0:
            total_regex += avg_regex
            regex_count += 1

    print("-" * 95)
    avg_consistency = total_consistency / len(results) if results else 0
    avg_regex_overall = total_regex / regex_count if regex_count else 0
    print(f"{'OVERALL':<35} {'':14} {avg_consistency:>12.2f} {avg_regex_overall:>8.2f}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Prompt reliability evaluation harness")
    parser.add_argument("--test-file", default="test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per variant (default: 3)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--output", default=None, help="Path to write detailed JSON results")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--scorers", default=None, help="Comma-separated list of scorers to use (exact,regex,semantic). Default: per test case.")
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv("OpenAI_KEY_TOKEN")
    if not api_key:
        print("Error: OpenAI_KEY_TOKEN not found in .env file", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load test cases
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}", file=sys.stderr)
        sys.exit(1)

    with open(test_file) as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {test_file}")
    print(f"Model: {args.model} | Runs per variant: {args.runs} | Delay: {args.delay}s")

    scorers_filter = args.scorers.split(",") if args.scorers else None

    # Run evaluation
    results = run_evaluation(test_cases, client, args.model, args.runs, args.delay, scorers_filter)

    # Print summary
    print_summary(results)

    # Write detailed results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "model": args.model,
            "runs_per_variant": args.runs,
            "test_cases_file": str(test_file),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {output_path}")


if __name__ == "__main__":
    main()
