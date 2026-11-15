"""Retry a disappearing atomic temporary file; all other storage errors remain fatal."""
import os,re,subprocess,time

def measure(root):
 sizes=[]
 for args in [['du','-sb','.'],['du','-s','-B1','.']]:
  for attempt in range(3):
   p=subprocess.run(args,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,env={**os.environ,'LC_ALL':'C'})
   if p.returncode==0:
    sizes.append(int(p.stdout.split()[0]));break
   lines=p.stderr.strip().splitlines()
   transient=p.returncode==1 and bool(lines) and all(re.fullmatch(r"du: cannot access '.+\.tmp': No such file or directory",s) for s in lines)
   if not transient or attempt==2:raise subprocess.CalledProcessError(p.returncode,args,output=p.stdout,stderr=p.stderr)
   time.sleep(.05)
 return max(sizes)+1048576
