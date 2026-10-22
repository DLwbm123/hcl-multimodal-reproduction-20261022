import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_hcl.audit import audit
from medical_hcl.data import write_json
if __name__=='__main__':
    result=audit(os.environ['PREPARED_DATA_ROOT'],os.environ['SOURCE_DATA_ROOT'])
    write_json(os.environ['AUDIT_REPORT_PATH'],result)
    print(json.dumps(result))
