"""Paper figures from results/summary.json, results/reach.json and the state logs.
Usage: ~/yodran/arc-prize-2026/.venv/bin/python analysis/figures.py results/ paper/fig
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Validated reference palette (dataviz skill), slots 1-2; text stays in ink tones, never series colour.
C31, CE4 = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
PASS, FAIL = '#008300', '#e34948'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
reach = json.load(open(root / 'reach.json'))


def pool(prefix):
    keys = ('failed', 'no_edit', 'touched_gold_file', 'overlapped_gold_lines')
    return {k: sum(v[k] for r, v in reach.items() if r.startswith(prefix)) for k in keys}


# ---- Figure 1: where failed runs end up (share of failed runs, mutually exclusive stages) ----
cats = ['No edit', 'Edited other files only', 'Right file,\nother lines', 'Edited the\ngold lines']
fig, ax = plt.subplots(figsize=(7.2, 3.2))
w = 0.38
for i, (name, pre, col) in enumerate((('Gemma 4 E4B (live)', 'e4b', CE4), ('Gemma 4 31B (replayed, 17 configs)', 'tn-', C31))):
    p = pool(pre)
    n = p['failed']
    vals = [p['no_edit'], n - p['no_edit'] - p['touched_gold_file'], p['touched_gold_file'] - p['overlapped_gold_lines'],
            p['overlapped_gold_lines']]
    xs = [k + (i - 0.5) * w for k in range(4)]
    bars = ax.bar(xs, [100 * v / n for v in vals], width=w - 0.04, color=col, label=f'{name}, n={n} failed runs')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f'{100 * v / n:.0f}%', ha='center', va='bottom',
                fontsize=8.5, color=INK)
ax.set_xticks(range(4), cats)
ax.set_ylabel('% of failed runs')
ax.set_ylim(0, 75)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.set_title('Where failed runs end up (any intermediate state counts)', fontsize=10.5, color=INK, loc='left')
fig.tight_layout()
fig.savefig(out / 'fig1_reach.png', dpi=200)

# ---- Figure 2: the one lost fix, step by step ----
V = {}
for f in glob.glob(str(root / 'verify-*' / '**' / 'verdicts.jsonl'), recursive=True):
    for line in open(f):
        v = json.loads(line)
        if v['kind'] == 'state' and v['resolved'] is not None:
            V[(v['iid'], v['sha'])] = v['resolved']
recs = [json.loads(l) for l in open(root / 'run-tn' / 'states.jsonl')]
recs = [r for r in recs if r['run'] == 'tn-abl3-k4' and r['iid'] == 'rich_3470']
fig, ax = plt.subplots(figsize=(7.2, 2.3))
end = max(r['tool_calls_used'] for r in recs) + 1
for k, r in enumerate(recs):  # a state lasts until the next recorded mutating/test call
    x0 = r['tool_calls_used'] - 0.5
    x1 = (recs[k + 1]['tool_calls_used'] - 0.5) if k + 1 < len(recs) else end
    ok = V.get((r['iid'], r['sha'])) if r['sha'] != 'EMPTY' else None
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 1, color=PASS if ok else (FAIL if ok is False else GRID), lw=0))
    if r['tool'] in ('edit_file', 'write_file') and r['ok']:
        ax.plot([r['tool_calls_used']], [1.2], marker='v', ms=5, color=INK2, ls='')
    if r['is_test_cmd'] and r['exit_code'] == 0:
        ax.plot([r['tool_calls_used']], [-0.2], marker='^', ms=5, color=INK2, ls='')
    if r.get('submitted_sha'):
        ax.text(r['tool_calls_used'], 1.55, 'submit', ha='center', fontsize=8, color=INK)
ax.annotate('call 57: edit makes it pass', (57, 1.0), (40, 1.95), fontsize=8, color=INK, ha='center',
            arrowprops={'arrowstyle': '->', 'color': INK2, 'lw': 0.8})
ax.annotate('call 63: second edit, no test in between → fails', (63, 0.0), (44, -1.0), fontsize=8, color=INK,
            ha='center', arrowprops={'arrowstyle': '->', 'color': INK2, 'lw': 0.8})
ax.set_xlim(0, max(r['tool_calls_used'] for r in recs) + 2)
ax.set_ylim(-1.3, 2.3)
ax.set_yticks([])
ax.set_xlabel('tool call')
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.set_title('rich_3470, 31B: workspace state after each call (green = passes hidden tests, red = fails, grey = no diff)'
             '\n▼ successful edit   ▲ agent test command exited 0', fontsize=8.5, color=INK, loc='left')
fig.tight_layout()
fig.savefig(out / 'fig2_lost_fix.png', dpi=200)
print('wrote', sorted(p.name for p in out.glob('*.png')))
