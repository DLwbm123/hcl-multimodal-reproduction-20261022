# Aggregate artifacts

- `aggregate_results.json`: pilot, strict and recovery metrics, gate summaries, call/phase/category counts and resource receipts. No per-example outputs.
- `final_comparison.csv`: paper Table 2 versus this run, 0–100 scale.
- `sample_ids_*.json`: public dataset IDs only, deterministic splits and prior-test exclusion.
- `offline_tests.txt`: 29 passing server tests from the experimental environment.

Strict-prefix and recovery costs must be added; the checkpoint reuses earlier stages but is not a second independent run. Legacy pilot scope labels and missing task IDs limit precise category attribution. No token count was measured.
