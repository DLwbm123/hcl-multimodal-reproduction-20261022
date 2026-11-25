"""Start once on the Linux GPU host. Python/runtime and paths come from the pilot."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["RUN_ROOT"]).resolve()
base = Path(os.environ["PILOT_ROOT"]).resolve()
design = json.loads((root / "design.json").read_text())
limits = design["budget"]
entries = json.loads((base / "ENTRYPOINTS.json").read_text())
python = Path(entries["python"])
parent = Path(entries["worker"]).parent
protocol = json.loads((base / "protocol.json").read_text())
gpu = int(os.environ.get("RUN_GPU", 0))
available = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True))
if available < protocol["budget"]["min_gpu_free_mib"] or shutil.disk_usage(root).free < limits["storage_stop_bytes"]:
    raise RuntimeError("budget preflight: insufficient available resources")
if root == base or (root / "budget.json").exists():
    raise RuntimeError("refusing to reuse a previous run")
for subdir in ("tmp", "cache", "run"):
    (root / subdir).mkdir(exist_ok=True)
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HOME": str(root/"cache"), "TMPDIR": str(root/"tmp"), "CUDA_CACHE_PATH": str(root/"cache"),
       "TRITON_CACHE_DIR": str(root/"cache"), "TORCH_HOME": str(root/"cache"), "XDG_CACHE_HOME": str(root/"cache"),
       "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
check = subprocess.check_output([str(python), "-"], input=(root / "check.py").read_text(), cwd=root, env=env, text=True)
receipt = json.loads(check)
if not receipt["passed"]:
    raise RuntimeError("CPU preflight failed")
(root / "CPU_CHECK.json").write_text(check)
supervisor = (base / "code" / "supervisor.py").resolve()
started = time.time()
nonce = str(time.time_ns())
worker_entry = parent / ("r" + nonce + ".py")
super_entry = parent / ("s" + nonce + ".py")
for path, target in ((worker_entry, root / "run.py"), (super_entry, supervisor)):
    with path.open("x") as f:
        f.write("import runpy\nrunpy.run_path(" + repr(str(target)) + ", run_name='__main__')\n")
env.update(WORKER_PYTHON=str(python), WORKER_ENTRY=str(worker_entry))
protocol["budget"].update(gpu=gpu, max_total_model_calls=limits["max_total_model_calls"],
                          storage_stop_bytes=limits["storage_stop_bytes"], watch_seconds=3)
(root / "protocol.json").write_text(json.dumps(protocol, indent=2))
(root / "budget.json").write_text(json.dumps({"started_epoch": started,
    "hard_deadline_epoch": started + limits["max_seconds"], "soft_deadline_epoch": started + limits["max_seconds"] - 60}))
with (root / "supervisor.log").open("x") as log:
    proc = subprocess.Popen([str(python), str(super_entry)], cwd=root, env=env, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
print(json.dumps({"supervisor_pid": proc.pid, "available_gpu_mib": available, "gpu": gpu}))
