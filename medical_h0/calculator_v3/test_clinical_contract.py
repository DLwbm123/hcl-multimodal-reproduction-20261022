"""Independent boundary/sentinel checks, synthetic only; no benchmark answers."""
import itertools,math,unittest
from clinical_catalog import CALCS,execute,identity,details,coverage
from strict_scoring import value_match
from runtime_state import ExecutionState
from tools import calculate

class ClinicalContracts(unittest.TestCase):
    def test_wells_all_combinations(self):
        keys=['clinical_signs_dvt','pe_most_likely_diagnosis','heart_rate_gt_100','immobilization_or_surgery','previous_dvt_pe','hemoptysis','malignancy']
        weights=[3,3,1.5,1.5,1.5,1,1]
        for bits in itertools.product([False,True],repeat=7):
            self.assertEqual(execute('wells_pe',dict(zip(keys,bits)))['value'],sum(w for b,w in zip(bits,weights) if b))
    def test_gcs_components(self):
        for e,v,m in itertools.product(range(1,5),range(1,6),range(1,7)):
            self.assertEqual(execute('glasgow_coma_scale',dict(eye_response=e,verbal_response=v,motor_response=m,is_intubated=False))['value'],e+v+m)
    def test_shock(self):
        for hr,sbp in [(60,120),(100,100),(150,75),(75,150)]:
            self.assertAlmostEqual(execute('shock_index',dict(heart_rate=hr,systolic_bp=sbp,diastolic_bp=None,patient_type='adult'))['value'],hr/sbp)
        with self.assertRaises(ValueError):execute('shock_index',dict(heart_rate=100,systolic_bp=0,diastolic_bp=None,patient_type='adult'))
    def test_qsofa_boundaries(self):
        for rr,bp,altered in itertools.product([21,22,23],[99,100,101],[False,True]):
            r=execute('qsofa_score',dict(respiratory_rate=rr,systolic_bp=bp,altered_mentation=altered,gcs_score=14 if altered else 15))
            self.assertEqual(r['value'],int(rr>=22)+int(bp<=100)+int(altered))
    def test_news_rcp_chart(self):
        # RCP NEWS2 chart 1, 2017, scale 1. Scale 2 checked separately.
        base=dict(respiratory_rate=16,spo2=98,on_supplemental_o2=False,temperature=37.,systolic_bp=120,heart_rate=70,consciousness='A',use_scale_2=False)
        checks={'respiratory_rate':[(8,3),(9,1),(11,1),(12,0),(20,0),(21,2),(24,2),(25,3)],'spo2':[(91,3),(92,2),(93,2),(94,1),(95,1),(96,0)],'systolic_bp':[(90,3),(91,2),(100,2),(101,1),(110,1),(111,0),(219,0),(220,3)],'heart_rate':[(40,3),(41,1),(50,1),(51,0),(90,0),(91,1),(110,1),(111,2),(130,2),(131,3)],'temperature':[(35.,3),(35.1,1),(36.,1),(36.1,0),(38.,0),(38.1,1),(39.,1),(39.1,2)]}
        for k,items in checks.items():
            for val,score in items:self.assertEqual(execute('news2_score',{**base,k:val})['value'],score,(k,val))
        self.assertEqual(execute('news2_score',{**base,'consciousness':'C','on_supplemental_o2':True})['value'],5)
        for spo2,oxy,expected in [(83,False,3),(84,False,2),(86,False,1),(88,False,0),(93,False,0),(93,True,3),(95,True,4),(97,True,5)]:
            self.assertEqual(execute('news2_score',{**base,'use_scale_2':True,'spo2':spo2,'on_supplemental_o2':oxy})['value'],expected)
    def test_strict_names(self):
        self.assertNotEqual(identity('Alvarez Score for Pulmonary Embolism'),identity("Wells' Criteria for Pulmonary Embolism"))
        self.assertNotEqual(identity('Pulmonary Embolism'),identity('wells_pe'))
        self.assertNotEqual(identity('wells_dvt'),identity('wells_pe'))
        self.assertNotEqual(identity('National Early Warning Score (NEWS)'),identity('NEWS2'))
        self.assertNotEqual(identity('SIRS, Sepsis, and Septic Shock Criteria'),identity('sirs_criteria'))
        self.assertEqual(identity('Heart Rate/Systolic Blood Pressure Ratio'),identity('Shock Index'))
    def test_values_not_substrings(self):
        self.assertFalse(value_match('not 5, actually 7',5));self.assertFalse(value_match('5 mg','5 g'))
        self.assertFalse(value_match(float('nan'),5));self.assertFalse(value_match(True,1))
        self.assertFalse(value_match('class AB','class BA'));self.assertTrue(value_match('5','5'))
    def test_strict_full_scoring(self):
        from strict_scoring import score_prediction
        gold={'required_calculators':['Shock Index'], 'calculator_answers':[{'name':'Shock Index','final_answer':1.,'inputs':[]}]}
        row={'name':'Heart Rate/Systolic Blood Pressure Ratio','episode_id':'episode_000','value':1.}
        self.assertEqual(score_prediction({'calculators':[row]},gold)['n_correct'],1)
        self.assertEqual(score_prediction({'calculators':[row,row]},gold)['n_correct'],0)
        self.assertEqual(score_prediction({'calculators':[{**row,'episode_id':'episode_001'}]},gold)['n_correct'],0)
        self.assertEqual(score_prediction({'calculators':[]},gold)['value_accuracy'],0)
        self.assertLess(score_prediction({'calculators':[row,{'name':'invented','episode_id':'episode_000','value':1}]},gold)['utility'],1)
    def test_explicit_inputs(self):
        with self.assertRaises(ValueError):execute('wells_pe',{})
        with self.assertRaises(ValueError):execute('wells_pe',{k:'false' for k in details('wells_pe')['parameters']['properties']})
    def test_syntax_and_no_forced_submission(self):
        self.assertEqual(calculate('1 if x > 2 else 0',{'x':3}),1)
        state=ExecutionState([]);state.call('read_facts',{})
        for _ in range(2):self.assertTrue(state.call('calculate',{'expression':'if(x>2,1,0)','variables':{'x':3}})[1])
        state.prepare(20);self.assertIn('execute_calculator',state.allowed());self.assertIn('calculate',state.allowed());self.assertIsNone(state.tools.answer)
    def test_all_directory_details(self):
        for cid in CALCS:
            d=details(cid);self.assertTrue(d['references']);self.assertEqual(d['execution_enabled'],cid in coverage())
if __name__=='__main__':unittest.main()
