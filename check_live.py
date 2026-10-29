"""Optional real API smoke test. Uses your API credits; not an accuracy benchmark."""
import json
import server  # Loads .env from this project's directory.
from live_engine import run_live, VerificationError

if __name__ == '__main__':
    text = '\n'.join([
        'Pranveer Singh Institute of Technology (PSIT) is located in Lucknow.',
        'The capital of Australia is Canberra.',
        'Google was founded by Mark Zuckerberg.',
    ])
    try:
        report = run_live(text)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        labels = [r['label'] for r in report['results']]
        expected = ['CONTRADICTED', 'SUPPORTED', 'CONTRADICTED']
        print('\nExpected labels:', expected)
        print('Returned labels:', labels)
        raise SystemExit(0 if labels == expected else 1)
    except VerificationError as exc:
        print('SERVICE ERROR:', exc)
        raise SystemExit(2)
