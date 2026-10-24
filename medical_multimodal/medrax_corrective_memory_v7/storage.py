"""Count the chosen root, but never traverse model/data symlinks inside it."""
import os
from pathlib import Path
import stat


def tree_bytes(path, *, allocated=False):
    total=0;seen=set();pending=[Path(path).resolve(strict=True)]
    while pending:
        item=pending.pop()
        try:s=item.lstat()
        except FileNotFoundError:continue  # Concurrent atomic log replacement.
        identity=(s.st_dev,s.st_ino)
        if identity in seen:continue
        seen.add(identity);total+=s.st_blocks*512 if allocated else s.st_size
        if stat.S_ISDIR(s.st_mode):
            with os.scandir(item) as entries:pending.extend(Path(e.path) for e in entries)
    return total
