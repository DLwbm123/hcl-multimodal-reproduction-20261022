"""Allocated bytes without following shared-asset links; tolerate atomic writes."""
import os
import stat


def allocated_bytes(roots):
    total = 0
    for root in roots:
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                try:
                    info = os.stat(os.path.join(directory, name), follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += info.st_blocks * 512
    return total
