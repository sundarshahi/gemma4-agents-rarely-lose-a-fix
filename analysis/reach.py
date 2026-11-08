"""Did failed runs ever edit the right code? (file-level and line-range overlap with the gold patch)

Comparable to Kim et al.'s "reached and edited the correct functions" (arXiv 2603.24631), but measured on
lines of the ORIGINAL file: an agent hunk "hits" gold if its old-file line range overlaps a gold hunk's range
(widened by SLACK lines). Usage: python analysis/reach.py results/ results/gold_hunks.json   (or the competition tasks.jsonl to rebuild it)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load_jsonl  # noqa: E402

SLACK = 3
HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')


def hunks(diff):
    """{file: [(start, end)]} in old-file line numbers; skips test files (stripped by the harness anyway)."""
    out, cur = defaultdict(list), None
    for line in (diff or '').splitlines():
        if line.startswith('--- '):
            p = line[4:].strip()
            cur = p[2:] if p.startswith('a/') else None  # new files (/dev/null) have no gold overlap
        elif cur and (m := HUNK.match(line)):
            a, n = int(m.group(1)), int(m.group(2) or 1)
            out[cur].append((a, a + max(n, 1) - 1))
    return out


def overlap(h1, h2):
    return any(f in h2 and any(s1 <= e2 + SLACK and s2 <= e1 + SLACK for s1, e1 in h1[f] for s2, e2 in h2[f])
               for f in h1)


def main(root, tasks_path):
    if tasks_path.endswith('.jsonl'):  # competition tasks.jsonl -> derive gold hunk ranges and save them
        gold = {}
        for l in open(tasks_path):
            t = json.loads(l)
            gold[t['instance_id']] = hunks(t['patch'])
        json.dump(gold, open(Path(root) / 'gold_hunks.json', 'w'), indent=0)
    else:  # results/gold_hunks.json (derived labels: file + line ranges only, publishable per the hosts)
        gold = {k: defaultdict(list, {f: [tuple(x) for x in v] for f, v in d.items()})
                for k, d in json.load(open(tasks_path)).items()}
    summary = json.load(open(Path(root) / 'summary.json'))
    valid = set(summary['valid_tasks'] or [])
    states = load_jsonl(str(Path(root) / 'run-*' / '**' / 'states.jsonl'))
    diffs = defaultdict(list)
    for s in states:
        if 'diff' in s and s['sha'] != 'EMPTY':
            diffs[(s['run'], s['iid'])].append(s['diff'])
    out = {}
    for run, rep in summary['report'].items():
        fails = [r['iid'] for r in rep['rows'] if not r['final'] and r['iid'] in valid]
        n_file = n_line = n_none = 0
        for iid in fails:
            ds = diffs.get((run, iid), [])
            if not ds:
                n_none += 1
                continue
            g = gold[iid]
            hs = [hunks(d) for d in ds]
            n_file += any(set(h) & set(g) for h in hs)
            n_line += any(overlap(h, g) for h in hs)
        out[run] = {'failed': len(fails), 'no_edit': n_none, 'touched_gold_file': n_file, 'overlapped_gold_lines': n_line}
        print(f'{run:14s} failed {len(fails):3d}  no edit {n_none:3d}  gold file {n_file:3d}  gold lines {n_line:3d}')
    json.dump(out, open(Path(root) / 'reach.json', 'w'), indent=1)


def selftest():
    g = hunks('--- a/x.py\n+++ b/x.py\n@@ -10,4 +10,5 @@ def f():\n')
    assert g == {'x.py': [(10, 13)]}
    assert overlap(hunks('--- a/x.py\n+++ b/x.py\n@@ -15,2 +15,2 @@\n'), g)  # within SLACK
    assert not overlap(hunks('--- a/x.py\n+++ b/x.py\n@@ -40,2 +40,2 @@\n'), g)
    assert not overlap(hunks('--- a/y.py\n+++ b/y.py\n@@ -10,2 +10,2 @@\n'), g)
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(sys.argv[1], sys.argv[2])
