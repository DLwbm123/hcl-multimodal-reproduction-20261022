"""Execute upstream pure-domain tests and record calculators actually exercised."""
import json,sys
from pathlib import Path
import pytest
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'vendor/medical_calc'))
covered={}
def profile(frame,event,arg):
    if event=='return' and frame.f_code.co_name=='calculate' and '/services/calculators/' in frame.f_code.co_filename and hasattr(arg,'value'):
        obj=frame.f_locals.get('self')
        if obj:covered[obj.tool_id]=covered.get(obj.tool_id,0)+1
class Coverage:
    def pytest_runtest_call(self,item):sys.setprofile(profile)
    def pytest_runtest_teardown(self,item):sys.setprofile(None)
files=json.loads(Path(sys.argv[1]).read_text());code=pytest.main(['-q','--tb=short','-p','no:cacheprovider',*files],plugins=[Coverage()])
sys.setprofile(None)
Path(sys.argv[2]).write_text(json.dumps({'exit_code':int(code),'tested_calculators':covered,'test_files':[Path(p).name for p in files],'scope':'Upstream computational regression tests, not clinical certification'},indent=2))
raise SystemExit(code)
