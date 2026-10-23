import unittest
from tools import RestrictedTools,calculate,COLUMNS


def row(patient='P',unit='kg',sample='Chart',value='70'):
    return dict(zip(COLUMNS,(patient,'weight',value,'2026-01-01','episode_000',unit,sample,'synthetic')))

class BoundaryTests(unittest.TestCase):
    def setUp(self): self.t=RestrictedTools([row(),row('OTHER'),row(unit='lb',value='154')])
    def test_distractors_and_filters(self):
        self.assertEqual(self.t.call('read_facts',{})['total'],3)
        self.assertEqual(len(self.t.call('read_facts',{'offset':1,'limit':1})['rows']),1)
        with self.assertRaises(ValueError):self.t.call('read_facts',{'filters':{'patient_id':"P' OR 1=1--"}})
    def test_no_external_capabilities(self):
        for name,args in [('bash',{'command':'id'}),('read_file',{'path':'../gold.jsonl'}),('read_facts',{'path':'/etc/passwd'}),('read_facts',{'filters':{'sql':'ATTACH DATABASE'}}),('submit_answer',{'calculators':[],'path':'../out'})]:
            with self.subTest(name=name,args=args),self.assertRaises(ValueError): self.t.call(name,args)
    def test_expression_boundary(self):
        for x in ["__import__('os').system('id')",'(1).__class__','[x for x in (1,2)]',"open('/etc/passwd').read()",'10**1000000','nan',"'x'*10000000"]:
            with self.subTest(x=x),self.assertRaises((ValueError,SyntaxError)): calculate(x,{})
    def test_math_and_chain(self):
        result=calculate('w/(h**2)',{'w':70,'h':1.75})
        self.assertAlmostEqual(result,22.857142857)
        self.assertEqual(calculate('1 if x>20 else 0',{'x':result}),1)
        with self.assertRaises(ValueError):calculate('x+1',{})
    def test_submission_once_and_abstention(self):
        answer={'calculators':[{'name':'Synthetic score','episode_id':'episode_000','value':None}]}
        self.t.call('submit_answer',answer); answer['calculators'][0]['value']=999
        self.assertIsNone(self.t.answer['calculators'][0]['value'])
        with self.assertRaises(ValueError):self.t.call('read_facts',{})
    def test_nonfinite_and_bool(self):
        for v in [float('nan'),float('inf'),True]:
            with self.subTest(v=v),self.assertRaises(ValueError):
                self.t.call('submit_answer',{'calculators':[{'name':'x','episode_id':'e','value':v}]})
    def test_call_budget(self):
        for _ in range(80):self.t.call('read_facts',{})
        with self.assertRaises(ValueError):self.t.call('read_facts',{})



class FixtureTests(unittest.TestCase):
    def test_fixed_readonly_fixture_and_isolation(self):
        import tempfile,sqlite3
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fixture.sqlite'
            with sqlite3.connect(p) as db:
                db.execute('CREATE TABLE patient_facts ('+','.join(k+' TEXT' for k in COLUMNS)+')')
                db.execute('INSERT INTO patient_facts VALUES ('+','.join('?' for _ in COLUMNS)+')',tuple(row().values()))
            tools=RestrictedTools.from_fixture(p)
            self.assertEqual(tools.call('read_facts',{})['total'],1)
            other=RestrictedTools([row('SECOND')])
            self.assertEqual(other.call('read_facts',{})['rows'][0]['patient_id'],'SECOND')
            link=Path(directory)/'link.sqlite';link.symlink_to(p)
            with self.assertRaises(ValueError):RestrictedTools.from_fixture(link)

if __name__=='__main__':unittest.main()
