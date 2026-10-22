"""Pure evaluation of existing predictions. Never generates or learns."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_hcl.data import StrictMedicalLoader,read_rows,write_json
from medical_hcl.evaluation import MedicalEvaluator,TASKS
if __name__=='__main__':
    root=Path(os.environ['PREPARED_DATA_ROOT']);task=os.environ['EVAL_TASK'];role=os.environ['EVAL_ROLE']
    if task not in TASKS:raise ValueError('unknown task')
    if role not in ('validation','test','retention_anchor'):raise ValueError('evaluation role required')
    loader=StrictMedicalLoader(root)
    data=loader.load(root/f'visible/{task}/{role}.jsonl')
    evaluator=MedicalEvaluator(json.loads((root/'class_vocabulary.json').read_text()))
    result=evaluator.evaluate(data,read_rows(os.environ['PREDICTIONS_PATH']),split=role)
    destination=Path(os.environ['METRICS_PATH']).resolve()
    if destination.is_relative_to(root):raise ValueError('evaluation cannot write frozen prepared data')
    if destination.exists():raise FileExistsError('do not overwrite historical evaluation')
    write_json(destination,result)
