# Run the Real Tests

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24050355.svg)](https://doi.org/10.5281/zenodo.24050355)

Paper (PDF, DOI): https://doi.org/10.5281/zenodo.24050355

Code and derived data for the Kaggle *Gemma 4 Developer Agent* Paper Track entry by Sundar Shahi Thakuri.

The question: does a Gemma 4 coding agent ever have a workspace that passes the hidden tests and then submit a
worse patch? We score **every intermediate workspace state** with the official Phase-2 verifier, for the official
sample agent on Gemma 4 E4B (live) and for 298 public Gemma 4 31B trajectories from 17 agent configurations
(replayed edit by edit).

## Layout

| Path | What |
|---|---|
| `kernels/run_body.py` | GPU kernel: official sample agent + an observation hook that logs `git diff` after every mutating tool call |
| `kernels/replay_body.py` | CPU kernel: rebuilds intermediate states of public traces with the harness's own edit matcher |
| `kernels/verify_body.py` | CPU kernel: scores patches with `swegemma.harness.verification.verify_task` (audit mode: gold + empty patch) |
| `kernels/make_kernels.py` | builds the Kaggle kernel folders (`python kernels/make_kernels.py`, then `kaggle kernels push -p kernels/build/<slug>`) |
| `analysis/analyze.py` | any-time vs final resolution, lost fixes, selection rules, exact McNemar tests (`--selftest`) |
| `analysis/reach.py` | did failed runs edit the gold patch's lines? (`--selftest`) |
| `analysis/figures.py` | paper figures |
| `results/` | derived data only: per-state diffs, verdicts, audit verdicts, summaries |
| `paper/` | the writeup and figures |

## Reproduce

```
python analysis/analyze.py results/            # prints every table in the paper, writes results/summary.json
python analysis/reach.py results/ results/gold_hunks.json   # gold patch file + line ranges (derived labels)
python analysis/figures.py results/ paper/fig   # needs matplotlib
```

Re-running the kernels needs a Kaggle account that has joined the competition (data and L4 GPUs are gated).

## Data policy

No competition data is included: no problem statements, test patches, snapshots, sample-submission files, traces
or logs. `results/` holds agent-produced diffs and verdicts, which the hosts confirmed may be published as derived
labels (competition discussion 749803). The replayed trajectories are from
[TechNaom/gemma4-agentic-swe](https://github.com/TechNaom/gemma4-agentic-swe) (Apache-2.0, commit 0758ce1).

## License

Apache-2.0.

## Cite

Sundar Shahi Thakuri (2026). *Run the Real Tests: Scoring every intermediate step of 345 Gemma 4 coding-agent runs.* Preprint, Zenodo. https://doi.org/10.5281/zenodo.24050355
