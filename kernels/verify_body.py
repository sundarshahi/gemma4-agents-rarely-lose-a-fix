# CPU kernel body: score patches with the official Phase-2 verifier (swegemma.harness.verification.verify_task).
#   mode 'audit'  -> gold patch and empty patch for every public task (which tasks can discriminate agents)
#   mode 'states' -> every distinct intermediate workspace diff recorded by the run kernels
# CONFIG (dict) is injected above this file by make_kernels.py.
import asyncio, hashlib, time
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from swegemma.sandbox import SubprocessManager

tasks = {t.instance_id: t for t in load_tasks(DATA_DIR / 'tasks.jsonl')}
cfg = EvalConfig(tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots',
                 results_dir=WORK / 'vres', submission_dir=WORK, models={}, sandbox='subprocess',
                 timeout_seconds=300, verification_timeout_seconds=300, wheels_dir=DATA_DIR / 'wheels',
                 graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
                 display_mode='quiet')
sandbox = SubprocessManager(timeout_seconds=300)

def pick_evenly(items, room):
    """Up to `room` items spread evenly over `items`; all of them when room >= len(items)."""
    if room >= len(items):
        return list(items)
    if room <= 0:
        return []
    step = len(items) / room
    return [items[int(k * step)] for k in range(room)]

assert pick_evenly(list('abc'), 100) == list('abc') and pick_evenly(list('abcd'), 2) == ['a', 'c']
assert pick_evenly(list('abc'), 0) == [] and len(set(pick_evenly(list(range(10)), 7))) == 7

jobs = []  # (kind, run, iid, sha, patch)
if CONFIG['mode'] == 'audit':
    for iid, t in tasks.items():
        jobs.append(('gold', '-', iid, 'GOLD', t.patch))
        jobs.append(('empty', '-', iid, 'EMPTY', ''))
else:
    # ponytail: cap of CAP distinct states per (run, task); priority states always kept, rest sampled evenly.
    # Raise CAP if the CPU budget allows; the paper reports how many states were skipped.
    CAP = int(CONFIG.get('cap', 15))
    per = {}  # (run, iid) -> ordered list of records
    diffs = {}
    for f in sorted(glob.glob('/kaggle/input/**/states.jsonl', recursive=True)):
        for line in open(f):
            r = json.loads(line)
            if 'diff' in r:
                diffs[(r['iid'], r['sha'])] = r['diff']
            per.setdefault((r['run'], r['iid']), []).append(r)
    seen, skipped = set(), []
    for (run, iid), recs in per.items():
        order = list(dict.fromkeys(r['sha'] for r in recs if r['sha'] != 'EMPTY'))
        prio = {recs[-1]['sha']} | {r['submitted_sha'] for r in recs if r.get('submitted_sha')} | \
               {r['sha'] for r in recs if r.get('is_test_cmd') and r.get('exit_code') == 0}
        keep = [s for s in order if s in prio]
        rest = [s for s in order if s not in prio]
        keep += pick_evenly(rest, max(0, CAP - len(keep)))
        skipped += [(run, iid, s) for s in order if s not in keep]
        for s in keep:
            if s == 'EMPTY' or (iid, s) in seen or (iid, s) not in diffs:
                continue
            seen.add((iid, s))  # same diff on same task => same verdict, verify once
            jobs.append(('state', run, iid, s, diffs[(iid, s)]))
    json.dump(skipped, open(WORK / 'skipped_states.json', 'w'))
    print(f'{len(skipped)} states skipped by cap={CAP}', flush=True)
jobs.sort(key=lambda j: hashlib.md5(f'{j[2]}{j[3]}'.encode()).hexdigest())
n_sh, i_sh = CONFIG.get('num_shards', 1), CONFIG.get('shard', 0)
jobs = jobs[i_sh::n_sh]
print(f'{CONFIG["mode"]}: {len(jobs)} verifications (shard {i_sh}/{n_sh})', flush=True)

out = (WORK / 'verdicts.jsonl').open('w')
sem = asyncio.Semaphore(int(CONFIG.get('concurrency', 4)))
done = 0; t_start = time.time()

async def one(kind, run, iid, sha, patch):
    global done
    t = tasks[iid]
    snap, base, pp = resolve_task_snapshot_paths(cfg.snapshots_dir, iid, t.repo)
    async with sem:
        t0 = time.time()
        try:  # verify_task measures duration with perf_counter, so start_time must use the same clock
            r = await verify_task(sandbox, cfg, t, snap, base_snapshot_path=base, patch_path=pp,
                                  agent_patch=patch, start_time=time.perf_counter())
            rec = {'kind': kind, 'run': run, 'iid': iid, 'sha': sha, 'resolved': bool(r.resolved),
                   'exit_code': r.test_exit_code, 'error': r.error,
                   'tail': (r.test_output or '')[-1500:], 'dur': round(time.time() - t0, 1)}
        except Exception as e:
            rec = {'kind': kind, 'run': run, 'iid': iid, 'sha': sha, 'resolved': None,
                   'exit_code': None, 'error': f'verifier crash: {e!r}', 'tail': '', 'dur': round(time.time() - t0, 1)}
    out.write(json.dumps(rec) + '\n'); out.flush()
    done += 1
    if done % 10 == 0:
        print(f'{done}/{len(jobs)} in {(time.time()-t_start)/60:.1f} min', flush=True)

async def main():
    await asyncio.gather(*(one(*j) for j in jobs))

asyncio.run(main())
out.close()
print('verify done', flush=True)
