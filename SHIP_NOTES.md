# Ship First vs Later

## Ship First (this harness)

- **10 hand-rolled test cases** covering 6 failure modes: paraphrase, typo, noise, reorder, distractor, framing
- **run_eval.py** with exact match + regex scoring
- **Consistency score** (cross-variant agreement) and **flip rate** (perturbation drift)
- CLI that runs end-to-end, prints a summary table, optionally writes JSON

This is enough to catch regressions when you change prompts, switch models, or adjust temperature. Run in CI with `--runs 3` and fail the build if consistency drops below a threshold.

## Ship Later

1. **Semantic similarity scoring** — OpenAI embeddings are wired up but optional. Enable when you have open-ended generation tests where regex is insufficient.
2. **Async API calls** — add when test set exceeds ~50 cases and wall-clock time matters.
3. **Auto-generated test variants** — script that takes a baseline prompt and produces typo/paraphrase/noise mutations automatically. Saves manual authoring.
4. **Statistical significance** — with N=3 runs, you can't make strong claims. Increase to N=10+ and add bootstrap confidence intervals.
5. **Result persistence & trending** — append results to JSONL or SQLite, plot consistency over time across model versions.
6. **Temperature sweep** — run at temp 0.0, 0.3, 0.7, 1.0 to characterize the consistency-creativity tradeoff.
7. **Multi-model comparison** — loop over models, produce a comparison matrix.
8. **CI integration** — GitHub Action that runs the harness on PR and comments with the consistency delta.
