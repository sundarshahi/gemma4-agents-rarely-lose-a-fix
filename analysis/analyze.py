"""Intra-trajectory regret + counterfactual selection policies.

Inputs (downloaded kernel outputs):  results/<run>/states.jsonl, results/<run>/final.jsonl,
                                     results/verify-*/verdicts.jsonl, results/audit/verdicts.jsonl
Usage:  python analysis/analyze.py results/      (writes results/summary.json + prints tables)
        python analysis/analyze.py --selftest
"""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(pattern):
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                # live-run final.jsonl rows carry no run id; their states use '<model>-s1', e.g. run-e4b -> e4b-s1
                r.setdefault('run', Path(f).parent.name.replace('run-', '') + '-s1')
                out.append(r)
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value; b = base-only wins, c = policy-only wins."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# ---- selection policies: each maps an ordered list of state records -> chosen sha (or None = final) ----
def _green(recs):  # states at which the agent's own test command exited 0
    return [r for r in recs if r.get('is_test_cmd') and r.get('exit_code') == 0 and r['sha'] != 'EMPTY']


POLICIES = {
    'final (harness default)': lambda recs: None,
    'last workspace state': lambda recs: next((r['sha'] for r in reversed(recs) if r['sha'] != 'EMPTY'), None),
    'first submitted patch': lambda recs: next((r['submitted_sha'] for r in recs if r.get('submitted_sha')), None),
    'last agent-green state': lambda recs: (_green(recs)[-1]['sha'] if _green(recs) else None),
    'first agent-green state': lambda recs: (_green(recs)[0]['sha'] if _green(recs) else None),
}


def analyze(states, finals, verdicts, valid=None):
    seen = defaultdict(set)  # (iid, sha) -> set of verdicts across repeated scorings
    for v in verdicts:
        if v['kind'] == 'state' and v['resolved'] is not None:
            seen[(v['iid'], v['sha'])].add(bool(v['resolved']))
    # a state scored twice with different outcomes is flaky: treat as unknown, never as a pass
    V = {k: next(iter(s)) for k, s in seen.items() if len(s) == 1}
    analyze.flaky = sorted(k for k, s in seen.items() if len(s) > 1)
    F = {(f['run'], f['iid']): f for f in finals}
    traj = defaultdict(list)
    for s in states:
        traj[(s['run'], s['iid'])].append(s)

    runs = sorted({r for r, _ in F})
    report = {}
    for run in runs:
        keys = [k for k in F if k[0] == run and (valid is None or k[1] in valid)]
        rows = []
        for k in keys:
            recs = traj.get(k, [])
            final_ok = bool(F[k]['resolved'])
            order = list(dict.fromkeys(r['sha'] for r in recs if r['sha'] != 'EMPTY'))
            verdict_seq = [V.get((k[1], s)) for s in order]
            any_ok = final_ok or any(x is True for x in verdict_seq)
            first_pass_idx = next((i for i, x in enumerate(verdict_seq) if x is True), None)
            row = {'iid': k[1], 'final': final_ok, 'any': any_ok, 'n_states': len(order),
                   'n_verified': sum(x is not None for x in verdict_seq),
                   'n_after_first_pass': (len(order) - 1 - first_pass_idx) if first_pass_idx is not None else None,
                   'pass_seq': ''.join('P' if x is True else ('f' if x is False else '?') for x in verdict_seq)}
            for name, pol in POLICIES.items():
                sha = pol(recs)
                row[name] = final_ok if sha is None else V.get((k[1], sha), final_ok if sha == (order[-1] if order else None) else None)
            rows.append(row)
        n = len(rows)
        lost = [r for r in rows if r['any'] and not r['final']]
        rep = {'n_tasks': n, 'final_resolved': sum(r['final'] for r in rows),
               'any_time_resolved': sum(r['any'] for r in rows), 'lost_fixes': [r['iid'] for r in lost],
               'oscillating': [r['iid'] for r in rows if 'Pf' in r['pass_seq'] or 'fP' in r['pass_seq'].lstrip('f')],
               'states_per_task_median': sorted(r['n_states'] for r in rows)[n // 2] if n else 0,
               'policies': {}}
        rep['final_ci'] = wilson(rep['final_resolved'], n)
        rep['any_ci'] = wilson(rep['any_time_resolved'], n)
        for name in POLICIES:
            known = [r for r in rows if r[name] is not None]
            b = sum(1 for r in known if r['final (harness default)'] and not r[name])
            c = sum(1 for r in known if r[name] and not r['final (harness default)'])
            rep['policies'][name] = {'resolved': sum(bool(r[name]) for r in known), 'n_known': len(known),
                                     'lost_vs_final': b, 'gained_vs_final': c, 'mcnemar_p': round(mcnemar_exact(b, c), 4)}
        rep['rows'] = rows
        report[run] = rep
    return report


def pooled(states, finals, verdicts, valid, prefix):
    """Pool all runs whose id starts with prefix: totals, agent-green precision, label agreement."""
    rep = analyze(states, [f for f in finals if f['run'].startswith(prefix)], verdicts, valid)
    rows = [r for v in rep.values() for r in v['rows']]
    V = {}
    for v in verdicts:
        if v['kind'] == 'state' and v['resolved'] is not None:
            V.setdefault((v['iid'], v['sha']), set()).add(bool(v['resolved']))
    green = [s for s in states if s['run'].startswith(prefix) and s.get('is_test_cmd') and s.get('exit_code') == 0
             and s['sha'] != 'EMPTY' and (valid is None or s['iid'] in valid)]
    gk = [next(iter(V[(s['iid'], s['sha'])])) for s in green if len(V.get((s['iid'], s['sha']), ())) == 1]
    # label agreement: logged final verdict vs our re-verification of the run's last recorded state
    last = {}
    for s in states:
        if s['run'].startswith(prefix) and s['sha'] != 'EMPTY':
            last[(s['run'], s['iid'])] = s['sha']
    agree = dis = 0
    for f in finals:
        if f['run'].startswith(prefix) and (valid is None or f['iid'] in valid) and f.get('replay_ok', True) is not False:
            sha = last.get((f['run'], f['iid']))
            vv = V.get((f['iid'], sha)) if sha else None
            if vv and len(vv) == 1:
                agree += (next(iter(vv)) == bool(f['resolved'])); dis += (next(iter(vv)) != bool(f['resolved']))
    out = {'task_runs': len(rows), 'final': sum(r['final'] for r in rows), 'any_time': sum(r['any'] for r in rows),
           'lost': sum(r['any'] and not r['final'] for r in rows),
           'states_scored': sum(r['n_verified'] for r in rows), 'states_total': sum(r['n_states'] for r in rows),
           'agent_green_states': len(green), 'agent_green_scored': len(gk), 'agent_green_hidden_pass': sum(gk),
           'label_agree': agree, 'label_disagree': dis,
           'runs_with_oscillation': sum(('P' in r['pass_seq']) and ('f' in r['pass_seq'].split('P', 1)[1]) for r in rows)}
    for name in POLICIES:
        known = [r for r in rows if r[name] is not None]
        b = sum(1 for r in known if r['final (harness default)'] and not r[name])
        c = sum(1 for r in known if r[name] and not r['final (harness default)'])
        out[name] = {'resolved': sum(bool(r[name]) for r in known), 'n': len(known), 'lost': b, 'gained': c,
                     'p': round(mcnemar_exact(b, c), 4)}
    return out


def print_report(report):
    for run, rep in report.items():
        n = rep['n_tasks']
        print(f'\n== {run}: {n} tasks')
        print(f"final resolved     {rep['final_resolved']:>3}  ({rep['final_resolved']/max(n,1):.1%}, 95% CI {rep['final_ci']})")
        print(f"any-time resolved  {rep['any_time_resolved']:>3}  ({rep['any_time_resolved']/max(n,1):.1%}, 95% CI {rep['any_ci']})")
        print(f"lost fixes         {len(rep['lost_fixes']):>3}  {rep['lost_fixes']}")
        for name, p in rep['policies'].items():
            print(f"  {name:28s} {p['resolved']:>3}/{p['n_known']:<3} +{p['gained_vs_final']} -{p['lost_vs_final']}  p={p['mcnemar_p']}")


def selftest():
    st = [  # task a: passes at state 2 then edited away; task b: never passes; task c: final passes
        {'run': 'r', 'iid': 'a', 'sha': 's1', 'is_test_cmd': False, 'exit_code': None},
        {'run': 'r', 'iid': 'a', 'sha': 's2', 'is_test_cmd': True, 'exit_code': 0},
        {'run': 'r', 'iid': 'a', 'sha': 's3', 'is_test_cmd': False, 'exit_code': None, 'submitted_sha': 's3'},
        {'run': 'r', 'iid': 'b', 'sha': 't1', 'is_test_cmd': True, 'exit_code': 1},
        {'run': 'r', 'iid': 'c', 'sha': 'u1', 'is_test_cmd': True, 'exit_code': 0, 'submitted_sha': 'u1'},
    ]
    fi = [{'run': 'r', 'iid': 'a', 'resolved': False}, {'run': 'r', 'iid': 'b', 'resolved': False},
          {'run': 'r', 'iid': 'c', 'resolved': True}]
    ve = [{'kind': 'state', 'iid': 'a', 'sha': 's1', 'resolved': False},
          {'kind': 'state', 'iid': 'a', 'sha': 's2', 'resolved': True},
          {'kind': 'state', 'iid': 'a', 'sha': 's3', 'resolved': False},
          {'kind': 'state', 'iid': 'b', 'sha': 't1', 'resolved': False},
          {'kind': 'state', 'iid': 'c', 'sha': 'u1', 'resolved': True}]
    rep = analyze(st, fi, ve)['r']
    assert rep['final_resolved'] == 1 and rep['any_time_resolved'] == 2, rep
    assert rep['lost_fixes'] == ['a']
    assert rep['policies']['last agent-green state']['resolved'] == 2
    assert rep['policies']['last agent-green state']['gained_vs_final'] == 1
    assert rep['policies']['first submitted patch']['resolved'] == 1
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-9
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
        sys.exit()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
    audit = load_jsonl(str(root / 'audit' / '**' / 'verdicts.jsonl'))
    valid = None
    if audit:
        g = {a['iid']: a['resolved'] for a in audit if a['kind'] == 'gold'}
        e = {a['iid']: a['resolved'] for a in audit if a['kind'] == 'empty'}
        valid = {i for i in g if g[i] and e.get(i) is False}
        print(f'audit: {len(g)} tasks, gold passes {sum(map(bool, g.values()))}, '
              f'empty passes {sum(map(bool, e.values()))}, valid {len(valid)}')
    finals = load_jsonl(str(root / 'run-*' / '**' / 'final.jsonl'))
    bad = [f for f in finals if f.get('replay_ok') is False]  # replay diverged from the logged run
    print(f'{len(finals)} task-runs, {len(bad)} dropped for replay infidelity')
    rep = analyze(load_jsonl(str(root / 'run-*' / '**' / 'states.jsonl')),
                  [f for f in finals if f.get('replay_ok') is not False],
                  load_jsonl(str(root / 'verify-*' / '**' / 'verdicts.jsonl')), valid)
    print(f'flaky states (conflicting repeat verdicts): {len(analyze.flaky)}')
    print_report(rep)
    S = load_jsonl(str(root / 'run-*' / '**' / 'states.jsonl'))
    Vd = load_jsonl(str(root / 'verify-*' / '**' / 'verdicts.jsonl'))
    keep = [f for f in finals if f.get('replay_ok') is not False]
    pools = {name: pooled(S, keep, Vd, valid, pre) for name, pre in (('E4B live', 'e4b'), ('31B replay (all configs)', 'tn-'))}
    for name, p in pools.items():
        print(f'\n## {name}: ' + json.dumps(p))
    json.dump({'valid_tasks': sorted(valid) if valid else None, 'flaky_states': analyze.flaky, 'pooled': pools, 'report': rep},
              open(root / 'summary.json', 'w'), indent=1)
