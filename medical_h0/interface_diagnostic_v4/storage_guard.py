"""Bounded fresh rescans for du races; never accept partial size output."""
import re,subprocess,time
from pathlib import Path

VANISHED=re.compile(r"^du: cannot access '(.+)': No such file or directory$")
def measure_usage(root,reserve=10_000_000,run=subprocess.run):
    root=Path(root)
    values=[]
    for command in [['du','-sb','.'],['du','-s','-B1','.']]:
        for attempt in range(3):
            p=run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                  timeout=15,env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
            if p.returncode==0:
                value=int(p.stdout.split()[0]);assert value>=0
                values.append(value);break
            # Only a vanished atomic temporary file may trigger a fresh scan.
            lines=p.stderr.strip().splitlines()
            matches=[VANISHED.fullmatch(line) for line in lines]
            retryable=p.returncode==1 and bool(lines) and all(m and m[1].startswith('./') and
                '..' not in Path(m[1]).parts and m[1].endswith('.tmp') for m in matches)
            if not retryable or attempt==2:
                raise RuntimeError('storage accounting failed: '+p.stderr[:2000])
            time.sleep(.05)
        else:raise RuntimeError('no complete storage measurement')
    return max(values)+reserve
