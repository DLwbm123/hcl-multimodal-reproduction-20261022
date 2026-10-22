import copy, unittest
from types import SimpleNamespace
from semantic_boundary import bounded_context, install_semantic_boundary, MEMORY_BOUNDARY

class BoundaryTests(unittest.TestCase):
    def test_no_memory_unchanged_and_records_not_modified(self):
        records=[{'correct_feedback':'historical','id':'fixed'}];before=copy.deepcopy(records)
        self.assertEqual(bounded_context('prompt',[]),'prompt')
        self.assertEqual(bounded_context('prompt',records),'prompt\n\n'+MEMORY_BOUNDARY)
        self.assertEqual(records,before)
    def test_real_call_boundary_and_exception_restore(self):
        records=[{'id':'fixed'}]
        def build(*args,**kwargs):
            self.assertIs(kwargs['selected_memory'],records)
            return 'original context'
        router=SimpleNamespace(build_context=build)
        def predict(examples,r,**kwargs):
            text=r.build_context({},selected_memory=records)
            self.assertEqual(text,bounded_context('original context',records))
            raise RuntimeError('unrelated backend failure')
        pipe=SimpleNamespace(_predict_with_router=predict,harness_version=0,global_batch_index=0,checkpoint=SimpleNamespace(has_state=False))
        install_semantic_boundary(pipe)
        with self.assertRaises(RuntimeError):pipe._predict_with_router([],router)
        self.assertIs(router.build_context,build)
        with self.assertRaises(ValueError):install_semantic_boundary(pipe)

if __name__=='__main__':unittest.main()
