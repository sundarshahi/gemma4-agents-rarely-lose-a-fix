"""Quick look at the audit and the trace replay. Usage: python analysis/inspect_audit_replay.py results/ [technaom valid_tasks.txt]"""
import collections
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
A = [json.loads(l) for l in open(R / 'audit/verdicts.jsonl')]
g = {a['iid']: a for a in A if a['kind'] == 'gold'}
e = {a['iid']: a for a in A if a['kind'] == 'empty'}
print('audit rows', len(A), 'gold', len(g), 'empty', len(e))
print('gold pass', sum(bool(a['resolved']) for a in g.values()), 'empty pass', sum(bool(a['resolved']) for a in e.values()),
      'verifier crash', sum(a['resolved'] is None for a in A))
valid = {i for i in g if g[i]['resolved'] and e.get(i, {}).get('resolved') is False}
print('valid', len(valid), dict(collections.Counter(i.split('_')[0] for i in valid)))
if len(sys.argv) > 2:
    tn = set(open(sys.argv[2]).read().split())
    print('TechNaom valid', len(tn), 'overlap', len(valid & tn), 'ours-only', sorted(valid - tn), 'theirs-only', sorted(tn - valid))
print('gold-fail reasons', collections.Counter(((a['error'] or 'tests failed')[:70]) for a in g.values() if not a['resolved']).most_common(6))

F = [json.loads(l) for l in open(R / 'run-tn/replay_fidelity.jsonl')]
print('replay runs', len(F), 'truncated', sum(f['truncated_at_call'] is not None for f in F),
      'final==logged', dict(collections.Counter(f['final_matches_logged_patch'] for f in F)),
      'runs with status mismatches', sum(f['status_mismatches'] > 0 for f in F))
fi = [json.loads(l) for l in open(R / 'run-tn/final.jsonl')]
print('replay_ok', sum(f['replay_ok'] for f in fi), 'resolved (logged)', sum(f['resolved'] for f in fi),
      'resolved & replay_ok', sum(f['resolved'] and f['replay_ok'] for f in fi))
print('runs', dict(collections.Counter(f['run'] for f in fi)))
S = [json.loads(l) for l in open(R / 'run-tn/states.jsonl')]
per = collections.defaultdict(set)
for s in S:
    if s['sha'] != 'EMPTY':
        per[(s['run'], s['iid'])].add(s['sha'])
n = sorted(len(v) for v in per.values())
print('state records', len(S), 'distinct states per run: median', n[len(n) // 2] if n else 0, 'max', n[-1] if n else 0,
      'total distinct', sum(n))
