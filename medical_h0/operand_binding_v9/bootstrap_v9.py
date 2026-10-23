"""Reuse frozen v5-v7 implementations without modifying historical runs."""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(BASE/name) for name in ('input_ablation_v7','component_diagnostic_v6','executable_gate_v5')]
