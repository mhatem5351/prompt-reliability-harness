# Prompt Reliability Evaluation Harness

A lean test harness that measures LLM response consistency across prompt variations — catching flaky answers before users do.

## Problem

The same question asked with a typo, different phrasing, or extra context sometimes gives inconsistent answers. This harness quantifies that drift.

## How It Works

1. **10 test cases** (5 synthetic + 5 real-world), each with 2-3 prompt variants
2. Each variant is sent to the LLM multiple times
3. Responses are scored for correctness (regex/exact match) and stability (consistency/flip rate)

### Test Types

- **Invariance tests** — paraphrase, typos, noise. The answer should NOT change.
- **Perturbation tests** — reordered options, distractors, framing shifts. The answer MIGHT change — we measure how often.

### Scoring Metrics

| Metric | What it measures |
|---|---|
| Exact match | Binary correctness for short answers |
| Regex match | Flexible correctness — tolerates surrounding text |
| Semantic similarity | Content overlap via OpenAI embeddings |
| Consistency | How often all variants produce the same text |
| Flip rate | How often a perturbation changes the answer vs baseline |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Usage

```bash
# Smoke test (1 run per variant)
python run_eval.py --runs 1

# Full evaluation (3 runs per variant)
python run_eval.py --runs 3 --output results.json

# Custom model
python run_eval.py --runs 3 --model gpt-4o

# Specific scorers
python run_eval.py --runs 3 --scorers regex,semantic
```

## Project Structure

```
test_cases.json   — 10 test cases with variants and expected answers
run_eval.py       — Main runner: calls OpenAI, scores, reports
scoring.py        — Scoring functions (exact, regex, semantic)
requirements.txt  — openai + python-dotenv
SHIP_NOTES.md     — What to ship first vs later
```

## Sample Output

```
Case ID                             Type          Consistency  Regex  Flip Rate
syn-inv-typo-01                     invariance    1.00         1.00            
syn-inv-noise-01                    invariance    0.44         1.00            INCONSISTENT
real-pert-anchoring-01              perturbation  0.67         1.00  0.50      flip_rate=0.50
OVERALL                                           0.49         1.00
```

## Ship First vs Later

See [SHIP_NOTES.md](SHIP_NOTES.md) for the full breakdown.

**Now:** 10 test cases + runner + regex/exact scoring + CI-ready consistency metrics.

**Later:** Auto-generated variants, statistical significance, result trending, async execution, multi-model comparison.
