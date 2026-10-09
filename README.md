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
| Consistency | Share of all responses (every variant and run) that match the most common answer, after lowercasing |
| Flip rate | How often a perturbation changes the answer vs baseline |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env (or export it in your shell).
# The older variable name OpenAI_KEY_TOKEN is still accepted.
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
results.json      — Output of the run shown below (gpt-4o-mini, 3 runs per variant)
requirements.txt  — openai + python-dotenv
.env.example      — Template for the API key
DESIGN.md         — Design decisions, test-case design, scoring approach
SHIP_NOTES.md     — What to ship first vs later
```

## Sample Output

Excerpt from the committed run in `results.json` (gpt-4o-mini, 3 runs per variant, temperature 0):

```
Case ID                             Type          Consistency  Regex  Flip Rate
syn-inv-typo-01                     invariance    1.00         1.00            
syn-inv-noise-01                    invariance    0.44         1.00            INCONSISTENT
real-pert-anchoring-01              perturbation  0.67         1.00  0.50      flip_rate=0.50
OVERALL                                           0.49         1.00
```

Across all 10 cases in that run:

- Regex correctness was 1.00 on every case: every response matched the expected answer pattern.
- Mean consistency was 0.49 (from 0.11 on `real-inv-codeswitch-01` to 1.00 on `syn-inv-typo-01`), mostly because the same answer was worded differently.
- Flip rate was 1.00 on 4 of the 5 perturbation cases and 0.50 on `real-pert-anchoring-01`.

## Limitations

- Consistency and flip rate compare whole responses as exact strings after lowercasing, so harmless rewording of a correct answer counts as inconsistent. They are most useful for short or structured answers.
- 3 runs per variant and 10 hand-written cases are enough to spot drift, not to make statistically significant claims.

## Design Notes

See [DESIGN.md](DESIGN.md) for the design decisions (model choice, JSON test cases, metrics, invariance vs perturbation tests, temperature 0) and how the test cases were built. The original document is [Prompt_Reliability_Design_Doc.docx](Prompt_Reliability_Design_Doc.docx).

## Ship First vs Later

See [SHIP_NOTES.md](SHIP_NOTES.md) for the full breakdown.

**Now:** 10 test cases + runner + regex/exact scoring + CI-ready consistency metrics.

**Later:** Auto-generated variants, statistical significance, result trending, async execution, multi-model comparison.

## License

MIT, see [LICENSE](LICENSE).
