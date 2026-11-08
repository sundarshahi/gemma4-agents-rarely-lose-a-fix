"""Writeup cover (2:1) and square thumbnail. Usage: <venv>/python analysis/cover.py results/ paper/fig"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
PASS, FAIL = '#008300', '#e34948'

V = {}
for f in glob.glob(str(root / 'verify-*' / '**' / 'verdicts.jsonl'), recursive=True):
    for line in open(f):
        v = json.loads(line)
        if v['kind'] == 'state' and v['resolved'] is not None:
            V[(v['iid'], v['sha'])] = v['resolved']
recs = [json.loads(l) for l in open(root / 'run-tn' / 'states.jsonl')]
recs = [r for r in recs if r['run'] == 'tn-abl3-k4' and r['iid'] == 'rich_3470']
end = max(r['tool_calls_used'] for r in recs) + 1


def strip(ax, y, h):
    for k, r in enumerate(recs):
        x0 = r['tool_calls_used'] - 0.5
        x1 = (recs[k + 1]['tool_calls_used'] - 0.5) if k + 1 < len(recs) else end
        ok = V.get((r['iid'], r['sha'])) if r['sha'] != 'EMPTY' else None
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, h, color=PASS if ok else (FAIL if ok is False else GRID), lw=0))


def make(w, h, name, title_size, stat_size, stat):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, end)
    ax.set_ylim(0, 10)
    ax.axis('off')
    strip(ax, 1.2, 1.1)
    ax.text(end * 0.06, 8.6, 'Run the\nReal Tests', fontsize=title_size, weight='bold', color=INK,
            va='top', linespacing=1.15)
    ax.text(end * 0.06, 3.15, stat, fontsize=stat_size, color=INK2, va='bottom', linespacing=1.4)
    fig.savefig(out / name, dpi=200, facecolor=SURF)
    plt.close(fig)


make(8, 4, 'cover.png', 30, 10.5, "A passing check from the repo's own tests was right 68% of the time;\nfrom the agent's scratch script, 38%. 1,431 Gemma 4 agent steps scored.")
make(5, 5, 'thumbnail.png', 30, 12, 'Repo tests: right 68%\nScratch scripts: right 38%')
print('wrote cover.png, thumbnail.png')
