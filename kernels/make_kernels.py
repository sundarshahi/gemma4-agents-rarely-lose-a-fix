"""Build Kaggle kernel folders (script kernels) from the shared bodies. Usage: python make_kernels.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent
USER = 'sundarshahi'
SETUP = (HERE / 'common_setup.py').read_text()
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
MODELS = ['gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct',
          'gemma-4-e4b-it-qat-w4a16-ct', 'gemma-4-e2b-it-qat-w4a16-ct']

BUDGET = {'max_time_minutes': 10, 'max_tool_calls': 100, 'max_turns': 100, 'command_timeout_s': 120, 'concurrency': 8}
RUNS = {  # slug -> config
    'g4-regret-run-e4b-s1': {'model': 'gemma-4-e4b-it-qat-w4a16-ct', 'run_id': 'e4b-s1', **BUDGET},
    'g4-regret-run-31b-s1': {'model': 'gemma-4-31b-it-qat-w4a16-ct', 'run_id': '31b-s1', **BUDGET},
}
REPLAY = {'g4-regret-replay-tn': {'repo_sha': '0758ce1e5cbc35fb919bfcfc73b1dcffea3a27fb'}}
VERIFY = {  # slug -> (config, kernel sources)
    'g4-regret-audit': ({'mode': 'audit', 'concurrency': 4}, []),
    **{f'g4-regret-verify-tn-{i}': ({'mode': 'states', 'num_shards': 3, 'shard': i, 'cap': 10, 'concurrency': 4},
                                    list(REPLAY)) for i in range(3)},
    # verification turned out to take ~6 s/state, so score every replayed state (no cap); overlaps give a flakiness check
    **{f'g4-regret-verify-tnall-{i}': ({'mode': 'states', 'num_shards': 2, 'shard': i, 'cap': 100000, 'concurrency': 4},
                                       list(REPLAY)) for i in range(2)},
    # 31B live run was stopped (GPU quota); E4B has ~200 distinct states, so one shard suffices
    'g4-regret-verify-live-0': ({'mode': 'states', 'num_shards': 1, 'shard': 0, 'cap': 100000, 'concurrency': 4},
                                ['g4-regret-run-e4b-s1']),
}


def write(slug, body, config, gpu, kernel_sources=(), models=(), internet=False):
    d = HERE / 'build' / slug
    d.mkdir(parents=True, exist_ok=True)
    code = f'CONFIG = {json.dumps(config)}\n\n{SETUP}\n\n{(HERE / body).read_text()}'
    (d / f'{slug}.py').write_text(code)
    meta = {
        'id': f'{USER}/{slug}', 'title': slug, 'code_file': f'{slug}.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': gpu, 'enable_tpu': False,
        'enable_internet': internet, 'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
        'competition_sources': ['gemma-4-developer-agent'], 'kernel_sources': list(kernel_sources),
        'model_sources': [f'google/gemma-4/Other/{m}/2' for m in models], 'docker_image': IMAGE,
    }
    if gpu:
        meta['machine_shape'] = 'NvidiaL4'
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    return d


SMOKE = {'g4-regret-smoke': {'model': 'gemma-4-e4b-it-qat-w4a16-ct', 'run_id': 'smoke', **BUDGET,
                              'task_ids': ['rich_3454', 'fastapi_11194', 'requests_7205', 'rich_2725']}}
for slug, c in {**SMOKE, **RUNS}.items():
    print(write(slug, 'run_body.py', c, True, models=[c['model']]))
for slug, c in REPLAY.items():
    print(write(slug, 'replay_body.py', c, False, internet=True))
for slug, (c, src) in VERIFY.items():
    print(write(slug, 'verify_body.py', c, False, kernel_sources=[f'{USER}/{s}' for s in src]))
