# Repair verification — 12 October 2026

- 24 deterministic regression tests: passed.
- Python compilation and JavaScript syntax checks: passed.
- Serper live search: succeeded using the supplied configuration.
- Original Groq model: HTTP 404; not listed in the account's available models.
- Updated Groq model: openai/gpt-oss-120b.
- End-to-end live smoke test: 3 of 3 expected labels returned, each with source evidence.

| Claim | Returned label | Correction |
|---|---|---|
| PSIT is located in Lucknow | CONTRADICTED | Kanpur, Uttar Pradesh |
| Australia's capital is Canberra | SUPPORTED | Not needed |
| Google was founded by Mark Zuckerberg | CONTRADICTED | Larry Page and Sergey Brin |

See live_test_report.json for the actual provider-backed run, queries and citations.
The tests are small engineering checks, not a representative accuracy benchmark.
There is no guarantee of correct classification for arbitrary input or future searches.
