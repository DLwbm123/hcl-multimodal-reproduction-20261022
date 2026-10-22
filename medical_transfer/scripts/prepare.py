"""Call from a neutral Python stdin/worker; environment carries actual paths."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_hcl.data import prepare
if __name__=='__main__':
    report=prepare(os.environ['SOURCE_DATA_ROOT'],os.environ['PREPARED_DATA_ROOT'])
    print(__import__('json').dumps(report,ensure_ascii=False))
