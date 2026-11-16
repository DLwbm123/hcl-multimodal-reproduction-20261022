import unittest,subprocess
from unittest.mock import patch
from storage_guard import measure

def result(code,err='',out='1234 .\n'):return subprocess.CompletedProcess([],code,out,err)
class Checks(unittest.TestCase):
 def test_transient_then_complete(self):
  with patch('storage_guard.subprocess.run',side_effect=[result(1,"du: cannot access './x.tmp': No such file or directory\n"),result(0),result(0,out='2048 .\n')]):self.assertEqual(measure('.'),2048+1048576)
 def test_real_error_fatal(self):
  for error in ["du: cannot access './x': Permission denied", "du: cannot access './x': No such file or directory"]:
   with patch('storage_guard.subprocess.run',return_value=result(1,error)),self.assertRaises(subprocess.CalledProcessError):measure('.')
 def test_persistent_race_stops(self):
  with patch('storage_guard.subprocess.run',return_value=result(1,"du: cannot access './x.tmp': No such file or directory")),self.assertRaises(subprocess.CalledProcessError):measure('.')
if __name__=='__main__':unittest.main()
