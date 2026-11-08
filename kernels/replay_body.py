# CPU kernel body: rebuild every intermediate workspace state of public Gemma 4 31B traces
# (TechNaom/gemma4-agentic-swe, Apache-2.0) by replaying edit_file/write_file calls onto the task
# snapshot with the harness's own matcher, then emit states.jsonl + final.jsonl in the same format
# as the live run kernels, so verify/analysis treat both sources identically.
# CONFIG (dict) is injected above this file by make_kernels.py.
import asyncio, hashlib, re, time
from swegemma.sandbox import sandbox_start, sandbox_stop
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.edit import apply_replacement
from swegemma.harness.container_setup import extract_snapshot, setup_baseline_commit, setup_git_exclude
from swegemma.harness.verification import strip_protected_files_from_patch
from swegemma.models import load_tasks
from swegemma.sandbox import SubprocessManager

REPO_URL, REPO_SHA = 'https://github.com/TechNaom/gemma4-agentic-swe', CONFIG['repo_sha']
SRC = Path('/tmp/technaom')
if not SRC.exists():
    subprocess.run(['git', 'clone', '-q', REPO_URL, str(SRC)], check=True)
    subprocess.run(['git', '-C', str(SRC), 'checkout', '-q', REPO_SHA], check=True)

tasks = {t.instance_id: t for t in load_tasks(DATA_DIR / 'tasks.jsonl')}
TEST_RE = re.compile(r'\b(pytest|py\.test|unittest|tox|nox)\b|python3? +-m +pytest|\btest_\w*\.py\b')
# Shell commands that may change files; they are re-executed during replay (see below).
MUT_RE = re.compile(r'sed\s+-i|\btee\b|cat\s*<<|\bgit\s+(checkout|restore|stash|apply|reset|am|mv|rm)\b|'
                    r'\b(mv|cp|rm|patch|truncate|perl\s+-pi)\b|open\([^)]*[\'"][wa]\+?[\'"]|write_text|'
                    r'>>?\s*(?!/tmp|/dev/null|&)[\w./-]+')

# collect traces; identical files across experiment folders are the same run (ablation-full duplicates)
runs, seen = [], set()
for tf in sorted(SRC.glob('experiments/g4swe-*/*/traces/trace_*.json')):
    h = hashlib.sha1(tf.read_bytes()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    rundir = tf.parent.parent
    iid = tf.stem[len('trace_'):]
    if iid not in tasks:
        continue
    lab = {}
    if (rundir / 'runs.jsonl').exists():
        for l in open(rundir / 'runs.jsonl'):
            r = json.loads(l)
            lab[r['instance_id']] = r
    pf = rundir / 'patches' / f'{iid}.patch'
    runs.append({'run': f'tn-{rundir.parent.name.split("-")[1]}-{rundir.name.split("_agent_")[-1]}',
                 'iid': iid, 'trace': tf, 'resolved': (lab.get(iid) or {}).get('resolved'),
                 'patch': pf.read_text() if pf.exists() else None})
print(f'{len(runs)} unique traces over {len({r["iid"] for r in runs})} tasks', flush=True)


def calls(trace):
    """Ordered (name, args, observed_status, exit_code) for every tool call in a trace."""
    for s in trace.get('steps', []):
        tcs = s.get('tool_calls') or []
        if isinstance(tcs, str):
            continue
        obs = s.get('observation')
        obs_list = obs if isinstance(obs, list) else ([obs] if obs else [])
        for i, tc in enumerate(tcs):
            st, ec = None, None
            o = obs_list[i] if i < len(obs_list) and len(obs_list) == len(tcs) else None
            if o:
                try:
                    c = json.loads(o.get('content', '{}'))
                    st = c.get('status')
                    ec = c.get('exit_code', (c.get('details') or {}).get('exit_code'))
                except Exception:
                    pass
            yield tc.get('function_name'), tc.get('arguments') or {}, st, ec


sandbox = SubprocessManager(timeout_seconds=300)
states_f = (WORK / 'states.jsonl').open('w')
final_f = (WORK / 'final.jsonl').open('w')
fid_f = (WORK / 'replay_fidelity.jsonl').open('w')
DIFF = 'cd /workspace && git add -N . >/dev/null 2>&1; git diff --binary _swegemma_baseline'
RESET = 'cd /workspace && git reset -q --hard _swegemma_baseline && git clean -fdq'

by_task = {}
for r in runs:
    by_task.setdefault(r['iid'], []).append(r)
t_start = time.time()
for n_task, (iid, trs) in enumerate(sorted(by_task.items()), 1):
    t = tasks[iid]
    snap, base, pp = resolve_task_snapshot_paths(DATA_DIR / 'snapshots', iid, t.repo)
    try:
        sid = asyncio.run(sandbox_start(sandbox))
        extract_snapshot(sandbox, sid, snap, patch_path=pp, base_snapshot_path=base)
        setup_git_exclude(sandbox, sid)
        setup_baseline_commit(sandbox, sid, 'replay_baseline')
        ws = Path(sandbox.sandboxes[sid]['workspace'])
        protected = list(getattr(t, 'test_files', None) or ())
        for r in trs:
            sandbox.exec(sid, RESET)
            last, k, mism, truncated = None, 0, 0, None
            for name, args, st, ec in calls(json.loads(r['trace'].read_text())):
                k += 1
                applied = None
                if name in ('edit_file', 'write_file'):
                    fp = str(args.get('filepath', ''))
                    fp = fp[len('/workspace/'):] if fp.startswith('/workspace/') else fp.lstrip('/')
                    p = (ws / fp).resolve()
                    try:
                        if not p.is_relative_to(ws.resolve()):
                            raise ValueError('outside workspace')
                        if name == 'write_file':
                            p.parent.mkdir(parents=True, exist_ok=True)
                            p.write_text(str(args.get('content', '')), encoding='utf-8')
                        else:
                            txt = open(p, encoding='utf-8', errors='replace', newline='').read()
                            if not txt:
                                raise ValueError('empty file')
                            am = args.get('allow_multiple', False)
                            am = am.strip().lower() in ('true', '1', 'yes') if isinstance(am, str) else bool(am)
                            res = apply_replacement(txt, args['old_string'], args['new_string'], filepath=fp, allow_multiple=am)
                            open(p, 'w', encoding='utf-8', newline='').write(res.new_content)
                        applied = True
                    except Exception:
                        applied = False
                    if st is not None and (st == 'ok') != applied:
                        mism += 1
                elif name == 'run_command' and st is not None and MUT_RE.search(str(args.get('command', ''))):
                    # Possibly file-changing shell command: re-execute it as the harness did. If its
                    # success/failure differs from the log, the replayed state can no longer be trusted.
                    rr = sandbox.exec(sid, f'cd /workspace && {args.get("command", "")}', timeout=120)
                    if (rr.exit_code == 0) != (st == 'ok'):
                        truncated = k
                        break
                if name in ('edit_file', 'write_file', 'run_command', 'submit_patch'):
                    diff = sandbox.exec(sid, DIFF).stdout or ''
                    if diff.strip():
                        diff = strip_protected_files_from_patch(diff, extra_protected_paths=protected)
                    sha = hashlib.sha1(diff.encode()).hexdigest()[:16] if diff.strip() else 'EMPTY'
                    cmd = str(args.get('command', args.get('filepath', '')))
                    rec = {'run': r['run'], 'model': 'gemma-4-31b-it-qat-w4a16-ct', 'iid': iid, 'tool': name,
                           'tool_calls_used': k, 'arg': cmd[:300], 'ok': st == 'ok' if st else applied,
                           'exit_code': ec if name == 'run_command' else None,
                           'is_test_cmd': bool(name == 'run_command' and TEST_RE.search(cmd)),
                           'sha': sha, 'changed': sha != last,
                           'submitted_sha': sha if (name == 'submit_patch' and sha != 'EMPTY') else None}
                    if sha != last:
                        rec['diff'] = diff
                    last = sha
                    states_f.write(json.dumps(rec) + '\n')
            final_diff = sandbox.exec(sid, DIFF).stdout or ''
            final_diff = strip_protected_files_from_patch(final_diff, extra_protected_paths=protected) if final_diff.strip() else ''
            norm = lambda d: '\n'.join(l for l in (d or '').splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')))
            fid = {'run': r['run'], 'iid': iid, 'status_mismatches': mism, 'truncated_at_call': truncated,
                   'final_matches_logged_patch': (norm(final_diff) == norm(r['patch'])) if r['patch'] is not None else None}
            fid_f.write(json.dumps(fid) + '\n')
            final_f.write(json.dumps({'run': r['run'], 'iid': iid, 'resolved': bool(r['resolved']),
                                      'patch': r['patch'] or '', 'replay_ok': truncated is None
                                      and fid['final_matches_logged_patch'] is not False}) + '\n')
        asyncio.run(sandbox_stop(sandbox, sid))
    except Exception as e:
        print(f'[replay] {iid} failed: {e!r}', flush=True)
    for fh in (states_f, final_f, fid_f):
        fh.flush()
    print(f'{n_task}/{len(by_task)} tasks, {(time.time()-t_start)/60:.1f} min', flush=True)
for fh in (states_f, final_f, fid_f):
    fh.close()
print('replay done', flush=True)
