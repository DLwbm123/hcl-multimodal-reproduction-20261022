import subprocess,tempfile,unittest
from unittest.mock import Mock
from storage_guard import measure_usage

def result(code=0,out='8192 .\n',err=''):return subprocess.CompletedProcess([],code,out,err)
class Tests(unittest.TestCase):
 def test_atomic_rename_fresh_retry(self):
  run=Mock(side_effect=[result(),result(1,'1 .\n',"du: cannot access './attempts.tmp': No such file or directory\n"),result(0,'16384 .\n')])
  self.assertEqual(measure_usage('.',10,run),16394);self.assertEqual(run.call_count,3)
 def test_real_failure_not_retried(self):
  for err in ["du: cannot read directory './data': Permission denied\n","du: cannot access './record.json': No such file or directory\n","du: cannot access './a.tmp': Input/output error\n"]:
   run=Mock(return_value=result(1,err=err))
   with self.assertRaises(RuntimeError):measure_usage('.',run=run)
   self.assertEqual(run.call_count,1)
 def test_repeated_race_fails_closed(self):
  run=Mock(return_value=result(1,err="du: cannot access './attempts.tmp': No such file or directory\n"))
  with self.assertRaises(RuntimeError):measure_usage('.',run=run)
  self.assertEqual(run.call_count,3)
 def test_timeout_not_hidden(self):
  with self.assertRaises(subprocess.TimeoutExpired):measure_usage('.',run=Mock(side_effect=subprocess.TimeoutExpired('du',15)))
 def test_counts_live_temporary_files(self):
  from pathlib import Path
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'live.tmp').write_bytes(b'x'*20000)
   self.assertGreaterEqual(measure_usage(d,0),20000)
if __name__=='__main__':unittest.main()
