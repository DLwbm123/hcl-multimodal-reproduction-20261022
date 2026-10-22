import unittest,copy
from types import SimpleNamespace
from diagnostic_support import memory_pool,lock_learning,pair_changes

class DiagnosticTests(unittest.TestCase):
    def test_pool_masks_preserve_source_and_exclude_anchors(self):
        source={'raw':[{'id':'r'}],'abstract':[{'id':'a'}],'anchor':[{'answer':'PRIVATE_ANCHOR'}],'skills':[{}]}
        original=copy.deepcopy(source)
        for name,counts in [('direct',(0,0)),('static_no_memory',(0,0)),('raw_only',(1,0)),('abstract_only',(0,1)),('full_memory',(1,1))]:
            pool=memory_pool(source,name);self.assertEqual((len(pool['raw']),len(pool['abstract'])),counts)
            self.assertEqual(pool['anchor'],[]);self.assertEqual(pool['skills'],[])
            pool['raw'].append('x');self.assertEqual(source,original)
    def test_forbidden_execution_paths(self):
        pipe=lock_learning(SimpleNamespace())
        for method in ['train','continual_train','test']:
            with self.assertRaises(PermissionError):getattr(pipe,method)([])
    def test_success_losses_not_net_cancelled(self):
        a={'1':{'success':True,'answer':'a'},'2':{'success':False,'answer':'b'}}
        b={'1':{'success':False,'answer':'b'},'2':{'success':True,'answer':'a'}}
        self.assertEqual(pair_changes(a,b),{'successful_to_failed':1,'failed_to_successful':1,'answer_changed':2,'total':2})
        with self.assertRaises(ValueError):pair_changes(a,{})

if __name__=='__main__':unittest.main()
