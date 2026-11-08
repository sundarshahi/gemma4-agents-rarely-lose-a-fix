"""How often is the agent's own passing check right? Split by what it ran: the repo's test suite or a scratch script.

A "check" is a run_command that runs a test runner or a python script named like repro/verify/test/check/debug.
A check "passes" when it exits 0; it is "right" when the workspace at that moment also passes the hidden tests.
Counted per check run (a state checked twice counts twice). Usage: python analysis/checks.py results/
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import wilson  # noqa: E402

REPO_TESTS = re.compile(r'(^|[\s/])tests?(/|::|\s|$)')
RUNNER = re.compile(r'\b(pytest|py\.test|unittest|tox|nox)\b')
SCRIPT = re.compile(r'python3?\s+\S*(repro|verify|test|check|debug)\S*\.py')


def kind(cmd):
    # repo suite: names the tests/ directory, or a test runner given no .py file (it then collects the repo's suite)
    if REPO_TESTS.search(cmd) or (RUNNER.search(cmd) and '.py' not in cmd):
        return 'repo test suite'
    return 'own scratch script'


def main(root):
    root = Path(root)
    V = {}
    for f in glob.glob(str(root / 'verify-*' / '**' / 'verdicts.jsonl'), recursive=True):
        for line in open(f):
            v = json.loads(line)
            if v['kind'] == 'state' and v['resolved'] is not None:
                V[(v['iid'], v['sha'])] = v['resolved']
    valid = set(json.load(open(root / 'summary.json'))['valid_tasks'])
    kept = {(r['run'], r['iid']) for r in map(json.loads, open(root / 'run-tn' / 'final.jsonl')) if r['replay_ok']}
    out = {}
    for model, path in (('E4B', 'run-e4b'), ('31B', 'run-tn')):
        C = defaultdict(lambda: [0, 0])
        for line in open(root / path / 'states.jsonl'):
            s = json.loads(line)
            if s['iid'] not in valid or s['sha'] == 'EMPTY' or s['tool'] != 'run_command' or s['exit_code'] != 0:
                continue
            if model == '31B' and (s['run'], s['iid']) not in kept:
                continue
            if not (s['is_test_cmd'] or SCRIPT.search(s['arg'])):
                continue
            v = V.get((s['iid'], s['sha']))
            if v is not None:
                C[kind(s['arg'])][0] += 1
                C[kind(s['arg'])][1] += v
        out[model] = {k: {'checks': n, 'hidden_pass': p, 'rate': round(p / n, 3), 'ci': wilson(p, n)} for k, (n, p) in C.items()}
        for k, d in out[model].items():
            print(f"{model} {k:20s} {d['hidden_pass']}/{d['checks']} = {100 * d['rate']:.1f}%  95% CI {d['ci']}")
    json.dump(out, open(root / 'checks.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        assert kind('pytest tests/test_repr.py') == 'repo test suite'
        assert kind('python -m pytest tests') == 'repo test suite'
        assert kind('python3 repro.py') == 'own scratch script'
        assert kind('pytest test_capture_record.py') == 'own scratch script'
        assert kind('pytest -q') == 'repo test suite' and kind('cd /workspace && pytest tests::x') == 'repo test suite'
        assert SCRIPT.search('python3 verify_fix.py') and not SCRIPT.search('python3 setup.py')
        print('selftest ok')
    else:
        main(sys.argv[1])
