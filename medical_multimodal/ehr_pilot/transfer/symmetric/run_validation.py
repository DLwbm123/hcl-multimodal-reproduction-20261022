"""Run only the missing development arm using the frozen V1 runner."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare

if __name__ == '__main__':
    compare.setup_model()
    compare.compare('demo', 'valid', int(os.environ['PART']))
