# GPU kernel body: run the official sample agent on the public tasks and record the workspace
# diff after every mutating tool call. The agent, prompts and budgets are untouched; we only observe.
# CONFIG (dict) is injected above this file by make_kernels.py.
import asyncio, collections, hashlib, re, threading, time

MODEL = CONFIG['model']
RUN_ID = CONFIG['run_id']

from swegemma.models import load_tasks
tasks = load_tasks(DATA_DIR / 'tasks.jsonl')
task_ids = [t.instance_id for t in tasks]
if CONFIG.get('task_ids'):
    task_ids = [t for t in task_ids if t in set(CONFIG['task_ids'])]
print(f'{len(task_ids)} tasks, model={MODEL}, run={RUN_ID}', flush=True)

# --- agent config: the official sample submission, model swapped (as the official notebook does) ---
AGENT_DIR = WORK / 'agent'
shutil.rmtree(AGENT_DIR, ignore_errors=True)
shutil.copytree(DATA_DIR / 'sample_submission', AGENT_DIR)
if MODEL != 'gemma-4-31b-it-qat-w4a16-ct':
    for yp in [AGENT_DIR / 'agent.yaml', *sorted((AGENT_DIR / 'sub_agents').glob('*.yaml'))]:
        if yp.is_file():
            txt = yp.read_text()
            txt = re.sub(r'^(\s*model:\s*).*$', rf'\g<1>{MODEL}', txt, flags=re.M)
            txt = re.sub(r'^\s*adapter:\s*.*\n?', '', txt, flags=re.M)
            yp.write_text(txt)
    shutil.rmtree(AGENT_DIR / 'adapters', ignore_errors=True)
else:  # sample adapters are dummies; drop them so every run is the plain base model
    for yp in [AGENT_DIR / 'agent.yaml', *sorted((AGENT_DIR / 'sub_agents').glob('*.yaml'))]:
        if yp.is_file():
            yp.write_text(re.sub(r'^\s*adapter:\s*.*\n?', '', yp.read_text(), flags=re.M))
    shutil.rmtree(AGENT_DIR / 'adapters', ignore_errors=True)
for p in sorted(AGENT_DIR.rglob('*')):
    if p.is_file() and p.suffix in ('.yaml', '.md'):
        print(f'---- {p.relative_to(AGENT_DIR)}\n{p.read_text()[:4000]}', flush=True)

# --- observation hook: snapshot `git diff` after each mutating tool call ---
import swegemma.tools.execution as E
import swegemma.tools.workspace as W
from swegemma.harness.verification import strip_protected_files_from_patch

STATES = (WORK / 'states.jsonl').open('a')
LOCK = threading.Lock()
LAST_SHA = {}
# A private index copy keeps the agent's own `git status`/index untouched.
DIFF_CMD = ('cd /workspace && cp .git/index .git/_obs_index 2>/dev/null; '
            'export GIT_INDEX_FILE=.git/_obs_index; git add -N . >/dev/null 2>&1; '
            '(git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD); '
            'rm -f .git/_obs_index')
TEST_RE = re.compile(r'\b(pytest|py\.test|unittest|tox|nox)\b|python3? +-m +pytest|\btest_\w*\.py\b')

def _outcome(out):
    try:
        d = json.loads(out) if isinstance(out, str) else dict(out)
    except Exception:
        return None, None
    ok = d.get('status') == 'ok' if 'status' in d else ('error' not in d)
    ec = d.get('exit_code', (d.get('details') or {}).get('exit_code') if isinstance(d.get('details'), dict) else None)
    return ok, ec

def _record(ctx, tool, args, kwargs, out):
    task = ctx.task
    res = ctx.docker.exec(ctx.container_id, DIFF_CMD)
    diff = res.stdout or ''
    if diff.strip():
        diff = strip_protected_files_from_patch(diff, extra_protected_paths=list(getattr(task, 'test_files', None) or ()))
    sha = hashlib.sha1(diff.encode()).hexdigest()[:16] if diff.strip() else 'EMPTY'
    ok, ec = _outcome(out)
    arg0 = args[0] if args else (kwargs.get('command') or kwargs.get('filepath') or '')
    rec = {
        'run': RUN_ID, 'model': MODEL, 'iid': task.instance_id, 'tool': tool,
        'tool_calls_used': getattr(ctx, 'tool_calls_used', None),
        'llm_calls_used': getattr(ctx, 'llm_calls_used', None),
        'elapsed_s': round(getattr(ctx, 'agent_elapsed_seconds', 0) or 0, 1),
        'arg': str(arg0)[:300], 'ok': ok, 'exit_code': ec,
        'is_test_cmd': bool(tool == 'run_command' and TEST_RE.search(str(arg0))),
        'sha': sha, 'changed': LAST_SHA.get(task.instance_id) != sha,
        'submitted_sha': (hashlib.sha1(ctx.submitted_patch.encode()).hexdigest()[:16]
                          if getattr(ctx, 'submitted_patch', None) and ctx.submitted_patch.strip() else None),
        'wall': time.time(),
    }
    if rec['changed']:
        rec['diff'] = diff
    LAST_SHA[task.instance_id] = sha
    with LOCK:
        STATES.write(json.dumps(rec) + '\n'); STATES.flush()

def _wrap(tool, fn):
    def wrapped(ctx, *a, **k):
        out = fn(ctx, *a, **k)
        try:
            _record(ctx, tool, a, k, out)
        except Exception as e:  # observation must never change the run
            print(f'[obs] {tool} record failed: {e!r}', flush=True)
        return out
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped

E.run_command = _wrap('run_command', E.run_command)
E.submit_patch = _wrap('submit_patch', E.submit_patch)
W.edit_file = _wrap('edit_file', W.edit_file)
W.write_file = _wrap('write_file', W.write_file)

# --- vLLM server (as in the official notebook) ---
import litellm, torch, yaml
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
litellm.drop_params = True
MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{MODEL}/2')
if not MODEL_PATH.exists():
    cands = glob.glob(f'/kaggle/input/**/{MODEL}/**/config.json', recursive=True)
    MODEL_PATH = Path(cands[0]).parent
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpus = torch.cuda.device_count()
extra = ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}') \
    if MODEL == 'gemma-4-12b-it-qat-w4a16-ct' else ()
server = VllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
    reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False, max_loras=1,
    max_lora_rank=128, scheduler_reserve_full_isl=False,
    tensor_parallel_size=4 if gpus >= 4 else (2 if gpus >= 2 else 1),
    startup_timeout=60 * 20, extra_args=extra), adapter_manifest=adapters)
t0 = time.time(); server.start(); print(f'vLLM up in {time.time()-t0:.0f}s, gpus={gpus}', flush=True)
models = server.create_model_registry(aliases=[MODEL], model_prefix='openai/', api_key='EMPTY')

# --- evaluate; the sample's eval_config (10 calls / 1 min) is too short to study, so budgets come from CONFIG ---
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.evaluate import Evaluator
ecf = next((c for n in ('eval_config.yaml', 'eval_config.yml') if (c := AGENT_DIR / n).is_file()), None)
ev = (yaml.safe_load(ecf.read_text()) or {}) if ecf else {}
ev = ev.get('evaluation', ev)
print('sample eval_config:', ev, flush=True)
limits, gen = build_submission_limits()
cfg = EvalConfig(
    tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots',
    results_dir=WORK / 'results', submission_dir=AGENT_DIR, models=models, sandbox='subprocess',
    timeout_seconds=int(CONFIG['command_timeout_s']),
    verification_timeout_seconds=300,
    max_time_minutes=float(CONFIG['max_time_minutes']),
    max_tool_calls=int(CONFIG['max_tool_calls']),
    max_turns=int(CONFIG['max_turns']),
    limits=limits, generation_constraints=gen, adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                    token_threshold=14336, event_retention_size=5),
    graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
    wheels_dir=DATA_DIR / 'wheels', task_ids=task_ids, concurrency=int(CONFIG.get('concurrency', 8)),
    display_mode='quiet', verbose=False)
json.dump({'config': CONFIG, 'eval_config_sample': ev,
           'budget': {'max_time_minutes': cfg.budget.time_minutes, 'max_tool_calls': cfg.budget.tool_calls,
                      'max_turns': cfg.budget.turns}}, open(WORK / 'run_meta.json', 'w'), indent=1, default=str)
t0 = time.time()
result = asyncio.run(Evaluator(cfg).run())
STATES.close()
rows = [{'iid': r.instance_id, 'resolved': bool(r.resolved), 'error': r.error,
         'tool_calls': r.tool_calls, 'llm_calls': r.total_llm_calls,
         'duration_s': round(r.duration_seconds or 0, 1), 'patch': r.agent_patch or ''}
        for r in result.task_results]
with open(WORK / 'final.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(f'done in {(time.time()-t0)/60:.1f} min: resolved {sum(r["resolved"] for r in rows)}/{len(rows)}', flush=True)
server.stop()
