"""Reuse the frozen v5/v6 runtime, graph and inference transport."""
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(BASE/'component_diagnostic_v6'), str(BASE/'executable_gate_v5')]
