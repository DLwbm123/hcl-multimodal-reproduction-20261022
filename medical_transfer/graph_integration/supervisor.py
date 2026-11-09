"""Independent deadline/storage guard. RUN_ROOT and WORKER_ENTRY are environment-only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


root = Path(os.environ["RUN_ROOT"])
budget = json.loads((root / "budget.json").read_text())
if (root / "STARTED.json").exists():
    raise RuntimeError("refuse duplicate launch; inspect the existing run")
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
available = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                     "--format=csv,noheader,nounits"], text=True)
free = {x.split(",")[0].strip(): int(x.split(",")[1]) for x in available.splitlines()}
if free[gpu] < 24000:
    raise RuntimeError("insufficient GPU memory headroom")
with (root / "worker.log").open("x") as output:
    child = subprocess.Popen([sys.executable, os.environ["WORKER_ENTRY"]],
                              stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    (root / "STARTED.json").write_text(json.dumps({"time": time.time(), "pid": child.pid,
        "supervisor_pid": os.getpid(), "gpu": gpu, "free_mib": free[gpu], **budget}))
    stopped = None
    while child.poll() is None:
        used = 0
        for path in Path(budget.get("storage_root", root)).rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass  # Atomic log/checkpoint replacements may remove temporary files.
        if time.time() >= budget["deadline_epoch"] or used >= budget["stop_new_bytes"]:
            stopped = "deadline_or_storage"
            (root / "STOPPED.json").write_text(json.dumps({"time": time.time(), "bytes": used}))
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            break
        time.sleep(5)
    exit_code = child.wait()
(root / "EXIT.json").write_text(json.dumps({"exit_code": exit_code, "time": time.time(), "stopped": stopped}))
