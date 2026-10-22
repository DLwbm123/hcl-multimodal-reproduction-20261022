"""Summarize an existing stage matrix, never select a checkpoint from test."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_hcl.evaluation import stage_summary
from medical_hcl.data import write_json
if __name__=='__main__':
    source=Path(os.environ['STAGE_MATRIX_PATH']);destination=Path(os.environ['SUMMARY_PATH'])
    if destination.exists():raise FileExistsError('preserve prior summary')
    matrix=json.loads(source.read_text())['primary_score_rows']
    write_json(destination,stage_summary(matrix))
