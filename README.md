# ClaimWeave AI — repaired live verification

## Start on Windows

Extract the ZIP. Open a terminal inside **ClaimWeave_AI_Fixed**.
Stop the previous server first (Ctrl+C), then run:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
```

Open **http://127.0.0.1:8000** and hard-refresh with **Ctrl+F5**.
Live Web · Groq + Serper is now the default. Paste your generated answer and select Analyze Claims.
After dependencies are installed, `start.bat` also starts the application.

The supplied API keys in `.env` are retained. The Groq model is now
`openai/gpt-oss-120b`: the previous `llama-3.3-70b-versatile` returned HTTP 404
and was absent from the account's live model list. If setting up a fresh copy, copy `.env.example` to `.env`
and replace the placeholders with your own Groq and Serper keys. Restart after changing it.
Never share `.env` or commit API keys. A “configured” indicator means a key is present;
it does not mean the provider has accepted it or that credits remain.

## What was wrong

- The configured Groq model was unavailable for the supplied account (HTTP 404).
- The original UI and API defaulted to local evidence matching. Arbitrary inputs
  never reached the web unless Live Web was selected manually.
- Even live mode changed otherwise usable verdicts into insufficient evidence
  when Groq did not reproduce an exact quotation.
- Service failures were incorrectly included in insufficient-evidence counts.
- Only limited search fields were read, and there was no targeted retrieval retry.
- Sequential claim processing could exceed the browser's 90-second deadline.

## Changes

- Live verification is the default in both the UI and API; local modes remain explicit.
- Groq extracts atomic claims and neutral search queries together, preserving false
  assertions, numbers and negation rather than silently correcting the input.
- Serper searches for the underlying fact. Source-linked answer boxes, organic snippets,
  related answers and knowledge graph fields are included when available.
- Groq selects existing source IDs. Python copies the actual retrieved excerpt and URL;
  no generated quote matching is needed. Invalid IDs trigger one repair attempt, then an error.
- An inconclusive claim gets one targeted follow-up search and another assessment.
- Verdicts include explanations, evidence, sources and corrections for contradictions.
- Search queries and all considered excerpts can be expanded in each card, including
  insufficient-evidence and error cards. Additional cited passages are visible.
- Connection/rate-limit retries, two concurrent claim workers, a longer browser timeout,
  explicit error counts and clearer configuration messages are included.
- `.env` is loaded relative to the application file, independent of the terminal directory.
- Blank inputs and answers over 8000 characters are rejected. More than 12 extracted
  factual claims requires splitting into smaller batches. Questions, opinions and
  random text produce a clear “no checkable factual assertions” message.

## Quick examples

| Input | Expected result when decisive evidence is retrieved |
|---|---|
| Pranveer Singh Institute of Technology (PSIT) is located in Lucknow. | Contradicted; correction should identify Kanpur |
| The capital of Australia is Canberra. | Supported |
| The capital of Australia is Sydney. | Contradicted; correction should identify Canberra |
| Google was founded by Mark Zuckerberg. | Contradicted; correction should name Larry Page and Sergey Brin |
| My private model achieved 99.9% accuracy yesterday. | Insufficient evidence without public supporting records |
| Hello, how are you? | No checkable factual assertions |

A paragraph can contain several different verdicts. Search-source availability and
model reasoning affect outcomes; these are examples, not hard-coded live rules.

## Testing

```powershell
python -m unittest discover -s tests -v
```

24 deterministic regression tests cover the original local mode and repaired live
pipeline, using mocked provider responses. These test implementation behavior, not
real-world classification accuracy.

To exercise both APIs with your keys and current web results (uses API credits):

```powershell
python check_live.py
```

The JSON report includes returned labels, evidence, corrections, queries and source links.
Compare the passages against the claims yourself. No universal accuracy percentage is claimed.

## Troubleshooting

- **Missing key / HTTP 401:** check the appropriate key in `.env` and restart.
- **HTTP 429:** rate limit or exhausted credits; check the provider dashboard and retry.
- **HTTP 400/404 from Groq:** check the `GROQ_MODEL` value and model access for your account.
- **Connection error / HTTP 403:** check network access and provider permissions.
- **Local evidence shown:** switch to Live Web, or hard-refresh the browser after restarting.
- **Genuine insufficient evidence:** the returned passages do not establish the claim;
  add the full entity name, date or other disambiguating context and retry.

## Scope

Live mode verifies public factual assertions using **search excerpts**, not full downloaded
webpages. It cannot guarantee correct classification for everything typed. Search results
can be incomplete, stale or misleading, and Groq can misinterpret them. A copied citation
establishes traceability, not proof that the interpretation is correct. Review important
verdicts at their original sources. Private facts and ambiguous claims may stay unverified.

Uploaded files and bundled datasets are used only in explicitly selected local modes.
They are not a training dataset or a substitute for live retrieval. Local matching has
limited coverage and is retained for demonstrations.

API references: https://console.groq.com/docs/api-reference and https://serper.dev/.
