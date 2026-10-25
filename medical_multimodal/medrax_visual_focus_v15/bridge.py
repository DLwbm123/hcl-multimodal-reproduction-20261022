"""Reuse the verified local subscription transport; no new authentication mechanism."""
import json
import os
from pathlib import Path
import subprocess
import time
import subscription
from storage import tree_bytes


def rpc(action,payload=None):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pro5000','/data/bmw/envs/p0/bin/python','/data/bmw/workers/r1021.py'],input=json.dumps({'action':action,'payload':payload}),text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError('RPC failed: '+p.stderr[-500:])
    return json.loads(p.stdout)


def main():
    root=Path(os.environ['R996_LOCAL']);bases=[root.parent/'medrax_subscription_v4',root.parent/'medrax_hcl_v5',root.parent/'medrax_memory_interface_v6',root.parent/'medrax_corrective_memory_v7',root.parent/'medrax_error_abstraction_v8',root.parent/'medrax_learning_qualification_v9',root.parent/'medrax_evidence_memory_v10',root.parent/'medrax_visual_memory_v11',root.parent/'medrax_postdraft_memory_v12',root.parent/'medrax_tool_query_v13',root.parent/'medrax_joint_query_v14',root.parent/'medrax_autonomy_12h_20261025'];budget=json.loads((root/'budget.json').read_text())
    while time.time()<budget['hard_deadline_epoch']:
        try:
            own=tree_bytes(root,allocated=True)
            aggregate=own+sum(tree_bytes(base,allocated=True) for base in bases)
            if own>=budget['own_stop_new_bytes'] or aggregate>=budget['aggregate_stop_new_bytes']:raise RuntimeError('Local storage write guard')
            job=rpc('pull')
            if job['state']=='finished':return
            if job['state']!='job':time.sleep(3);continue
            p=root/'private'/('response_'+job['key']+'.json')
            if p.exists():response=json.loads(p.read_text())
            else:response=subscription.execute(job['job'],root);p.write_text(json.dumps(response))
            rpc('respond',{'key':job['key'],'response':response});print(json.dumps({'key':job['key'],'status':response['status'],'seconds':response.get('seconds')}),flush=True)
        except Exception as exc:
            print(json.dumps({'time':time.time(),'error':repr(exc)}),flush=True);time.sleep(10)

if __name__=='__main__':main()
