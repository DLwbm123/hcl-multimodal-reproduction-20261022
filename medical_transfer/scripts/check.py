"""CPU-only integration checks against pinned HCL; no backend imports."""
import json,os,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
r=unittest.TextTestRunner(verbosity=2).run(suite)
report={'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'real_model_calls':0,'gpu_runs':0,'training_runs':0,'synthetic_only':True,'completed_epoch':time.time()}
(ROOT/'artifacts').mkdir(exist_ok=True)
(ROOT/'artifacts/cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(not r.wasSuccessful())
