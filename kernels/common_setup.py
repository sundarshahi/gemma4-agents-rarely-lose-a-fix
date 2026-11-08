# Shared setup, inlined into each Kaggle kernel by make_kernels.py (kernels can't import local files).
# Mirrors cell 1 of the official "Getting Started - Gemma 4 Developer Agent" notebook (ryanholbrook).
import glob, importlib, json, logging, os, shutil, subprocess, sys
from pathlib import Path

for k, v in {
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False',
    'LITELLM_LOG': 'WARNING', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OTEL_SDK_DISABLED': 'true',
    'OTEL_PYTHON_DISABLED': 'true',
}.items():
    os.environ.setdefault(k, v)

def quiet_logs():
    for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore',
                 'openai', 'urllib3', 'opentelemetry', 'opentelemetry.context'):
        lg = logging.getLogger(name); lg.setLevel(logging.WARNING)
        if 'litellm' in name.lower():
            lg.propagate = False; lg.handlers.clear()

WHEELHOUSE = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
if not WHEELHOUSE.exists():  # older mount layout
    WHEELHOUSE = Path('/kaggle/input/gemma-4-developer-agent-wheelhouse')
for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for p in glob.glob(pat):
        try: os.unlink(p)
        except OSError: pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    t = tmp_whl / name
    if not t.exists():
        try: os.symlink(w, t)
        except OSError: shutil.copyfile(w, t)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches(); quiet_logs()

DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
if not DATA_DIR.exists():
    DATA_DIR = Path('/kaggle/input/gemma-4-developer-agent')
WORK = Path('/kaggle/working'); WORK.mkdir(parents=True, exist_ok=True)
print('data dir', DATA_DIR, 'wheelhouse', WHEELHOUSE, flush=True)
