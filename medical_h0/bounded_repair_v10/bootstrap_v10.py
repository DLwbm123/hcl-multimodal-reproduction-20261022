import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(BASE/n) for n in ("operand_binding_v9","input_ablation_v7","component_diagnostic_v6","executable_gate_v5")]
