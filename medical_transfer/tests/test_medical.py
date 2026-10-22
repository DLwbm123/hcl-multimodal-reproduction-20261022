import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
UPSTREAM=Path(os.environ.get('HCL_UPSTREAM_ROOT',str(ROOT.parent)))
sys.path.insert(0,str(UPSTREAM))
from medical_hcl.evaluation import MedicalEvaluator,HCLMedicalEvaluator,parse_bbox,stage_summary
from medical_hcl.data import prepare,write_json,write_rows,StrictMedicalLoader,model_visible,assign_subroles,read_rows
from medical_hcl.bridge import construct_pipeline,NoInferenceModel

VOC=['normal','polyp']
def example(task='classification',id='x',role='validation',answer=None):
    if answer is None:answer={'classification':'polyp','polyp_localization':{'bbox':[0,0,10,10]},'visual_qa':'red; pink'}[task]
    return {'id':id,'task_id':id,'task_name':task,'task_type':'medical_'+task,'question':'synthetic question','answer':answer,'image':None,'split':role,'visible_context':{'width':20,'height':20},'metadata':{'width':20,'height':20,'role':role,'group_id':'g_'+id,'pixel_id':'p_'+id,'answer_kind':'set'}}

def fixture(root):
    rows=[];assets=[];counts={}
    for ti,task in enumerate(('classification','polyp_localization','visual_qa')):
        counts[task]={}
        for role,n in [('adaptation',150),('validation',3),('test',3)]:
            vis=[];gold=[];counts[task][role]=n
            for k in range(n):
                ident=f'{task}_{role}_{k}';idx=len(assets);image=f'images/im_{idx}.jpg'
                # Existence check only: synthetic bytes are never decoded as an image.
                p=root/image;p.parent.mkdir(exist_ok=True);p.write_bytes(b'synthetic-not-a-real-image')
                group=f'g_{idx}';pixel=f'p_{idx}'
                assets.append({'image':image,'width':20,'height':20,'group_id':group,'pixel_id':pixel})
                rows.append({'case_id':ident,'task':task,'role':role,'group_id':group,'pixel_id':pixel,'asset':idx,'question_type':'color_set' if task=='visual_qa' else None})
                vis.append({'case_id':ident,'image':image,'question':'synthetic question','output_contract':'synthetic','image_size':{'width':20,'height':20}})
                target={'class_label':VOC[k%2]} if task=='classification' else {'bbox':[0,0,10,10]} if task=='polyp_localization' else {'answer':'red; pink'}
                gold.append({'case_id':ident,'target':target})
            write_rows(root/f'inputs/{task}/{role}.jsonl',vis);write_rows(root/f'private/targets/{task}/{role}.jsonl',gold)
    write_rows(root/'audit/unified_manifest.jsonl',rows);write_rows(root/'audit/image_assets.jsonl',assets)
    write_json(root/'audit/class_vocabulary.json',VOC);write_json(root/'audit/SPLIT_FROZEN.json',{'version':3,'counts':counts})

class Metrics(unittest.TestCase):
    def setUp(self):self.e=MedicalEvaluator(VOC)
    def test_missing_keeps_denominator(self):
        m=self.e.evaluate([example(),example(id='y')],[{'task_id':'x','answer':'polyp'}])['metrics']
        self.assertEqual((m['total'],m['correct'],m['missing_count'],m['primary_score']),(2,1,1,.5))
    def test_timeout_stale_answer_is_failure(self):
        for status in ['timeout','error']:
            m=self.e.evaluate([example()],[{'task_id':'x','answer':'polyp','status':status}])['metrics'];self.assertEqual(m['correct'],0)
    def test_timeout_to_success_is_a_behavior_change(self):
        m=self.e.evaluate([example()],[{'task_id':'x','answer':'polyp','status':'ok'}],reference_predictions=[{'task_id':'x','answer':'polyp','status':'timeout'}])['metrics']
        self.assertEqual((m['behavior_change_count'],m['recovered_count']),(1,1))
    def test_classification_fixed_vocab_macro_f1(self):
        m=self.e.evaluate([example()],[{'task_id':'x','answer':'polyp'}])['metrics']
        self.assertEqual(m['macro_f1'],.5);self.assertEqual(m['macro_f1_classes'],2)
    def test_unknown_class_and_json_are_invalid(self):
        for answer in ['other','{"class":"polyp"}',float('nan')]:
            self.assertEqual(self.e.evaluate([example()],[{'task_id':'x','answer':answer}])['metrics']['format_compliant_count'],0)
    def test_bbox_half_open_iou(self):
        m=self.e.evaluate([example('polyp_localization')],[{'task_id':'x','answer':'{"bbox":[0,0,10,10]}'}])['metrics']
        self.assertEqual((m['mean_iou'],m['primary_score']),(1,1))
    def test_invalid_bbox_no_clipping_or_guessing(self):
        for a in ['{"bbox":[0,0,21,10]}',{'bbox':[0,0,True,10]},{'bbox':[0,0,float('nan'),10]},{'bbox':[10,0,0,10]},[0,0,10,10],{'bbox':[0,0,10,10],'extra':'x'},'```json\n{"bbox":[0,0,10,10]}\n```']:
            self.assertIsNone(parse_bbox(a,20,20))
    def test_bbox_partial_iou_and_threshold(self):
        m=self.e.evaluate([example('polyp_localization')],[{'task_id':'x','answer':{'bbox':[0,0,5,10]}}])['metrics']
        self.assertEqual((m['mean_iou'],m['primary_score']),(.5,1))
    def test_set_order_case_whitespace(self):
        m=self.e.evaluate([example('visual_qa')],[{'task_id':'x','answer':' Pink ; RED '}])['metrics'];self.assertEqual(m['primary_score'],1)
    def test_partial_set_not_anchor_success(self):
        m=self.e.evaluate([example('visual_qa')],[{'task_id':'x','answer':'red'}])['metrics']
        self.assertEqual(m['correct'],0);self.assertAlmostEqual(m['answer_f1_diagnostic'],2/3)
    def test_count_zero_not_missing(self):
        e=example('visual_qa',answer='0');e['metadata']['answer_kind']='integer'
        self.assertEqual(self.e.evaluate([e],[{'task_id':'x','answer':'０'}])['metrics']['correct'],1)
    def test_duplicate_unknown_prediction_ids_rejected(self):
        for p in [[{'task_id':'x'},{'task_id':'x'}],[{'task_id':'other'}]]:
            with self.assertRaises(ValueError):self.e.evaluate([example()],p)
    def test_gold_missing_and_duplicate_denominator_rejected(self):
        with self.assertRaises(ValueError):self.e.evaluate([example(),example()],[])
        with self.assertRaises(ValueError):self.e.evaluate([example(answer='not a class')],[])
    def test_recovery_cannot_cancel_anchor_loss(self):
        data=[example(id='x'),example(id='y')]
        result=self.e.evaluate(data,[{'task_id':'y','answer':'polyp'}],reference_predictions=[{'task_id':'x','answer':'polyp'}])['metrics']
        self.assertEqual((result['forget_count'],result['recovered_count']),(1,1))
    def test_hcl_shared_prediction_pool_subset(self):
        e=HCLMedicalEvaluator(VOC)
        self.assertEqual(e.evaluate([example()],[{'task_id':'x','answer':'polyp'},{'task_id':'y','answer':'normal'}])['metrics']['total'],1)
    def test_mixed_tasks_not_averaged_as_primary(self):
        self.assertIsNone(self.e.evaluate([example(),example('visual_qa',id='q')],[])['metrics']['primary_score'])
    def test_stage_matrix_forgetting_and_bwt_separate(self):
        r=stage_summary([[.7,None,None],[.9,.6,None],[.8,.7,.5]])
        self.assertAlmostEqual(r['average_forgetting_pp'],0);self.assertAlmostEqual(r['bwt_pp'],10)
        with self.assertRaises(ValueError):stage_summary([[float('nan')]])

class DataAndBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name);cls.source=cls.base/'source';cls.source.mkdir();fixture(cls.source)
        cls.prepared=cls.base/'prepared';cls.report=prepare(cls.source,cls.prepared)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.storage=Path(self.tmp.name)/'runtime'
        self.loader=StrictMedicalLoader(self.prepared)
    def test_frozen_roles_unchanged_and_no_overlap(self):
        self.assertTrue(self.report['original_validation_test_unchanged']);self.assertEqual(self.report['cross_group_roles'],0)
        for task,counts in self.report['counts'].items():self.assertEqual((counts['validation'],counts['test']),(3,3))
    def test_deterministic_group_assignment(self):
        m=read_rows(self.source/'audit/unified_manifest.jsonl');targets={}
        for task in ('classification','polyp_localization','visual_qa'):
            for t in read_rows(self.source/f'private/targets/{task}/adaptation.jsonl'):targets[t['case_id']]=t['target']
        self.assertEqual(assign_subroles(m,targets,1),assign_subroles(list(reversed(m)),targets,1))
    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):prepare(self.source,self.prepared)
    def test_private_join_and_zero_labels(self):
        data=self.loader.load(self.prepared/'visible/polyp_localization/validation.jsonl');self.assertEqual(data[0]['answer']['bbox'][0],0)
        self.assertNotIn('answer',model_visible(data[0]));self.assertNotIn('metadata',model_visible(data[0]))
    def test_strict_loader_no_val_alias_for_anchor(self):
        spec={'task_name':'classification'}
        rows=self.loader.load_task_spec(spec,split='anchor');self.assertTrue(all(x['metadata']['role']=='retention_anchor' for x in rows))
        with self.assertRaises(ValueError):self.loader.load_task_spec(spec,split='val')
    def test_hcl_pipeline_construct_no_generation(self):
        p=construct_pipeline(self.prepared,self.storage)
        self.assertIsInstance(p.model,NoInferenceModel);self.assertEqual(p.model.calls,0);self.assertFalse(p.forgetting_probe.enabled)
        with self.assertRaises(PermissionError):p.continual_train(self.prepared/'taskstream.json')
    def test_no_real_model_can_be_injected_without_authorization(self):
        with self.assertRaises(PermissionError):construct_pipeline(self.prepared,self.storage,model=object())
    def test_actual_hcl_gate_uses_medical_fixed_denominator(self):
        p=construct_pipeline(self.prepared,self.storage)
        cur=example();anc=example('visual_qa',id='anchor',role='retention_anchor')
        gate=p._evaluate_gate([cur,anc],[{'task_id':'x','answer':'polyp'}],current_val_examples=[cur],historical_anchor_examples=[anc],split='val',reference_predictions=[{'task_id':'anchor','answer':'pink; red'}])
        self.assertEqual(gate['historical_anchor_metrics']['forget_count'],1);self.assertEqual(gate['historical_anchor_metrics']['total'],1)
    def test_hcl_retrieval_never_contains_reserved_anchors(self):
        p=construct_pipeline(self.prepared,self.storage)
        rows=self.loader.load_task_spec({'task_name':'classification'},split='anchor',limit=20)
        p.memory.finalize_task('classification',rows)
        self.assertGreaterEqual(len(p.memory.historical_anchors()),20);self.assertEqual(p.memory.retrieval_candidates(),[])
    def test_memory_records_medical_correctness_and_blocks_wrong_roles(self):
        p=construct_pipeline(self.prepared,self.storage)
        e=example('polyp_localization',role='h_search')
        p.memory.record_training_interaction('polyp_localization',e,{'task_id':'x','answer':'{"bbox":[0,0,10,10]}'})
        self.assertTrue(p.memory.raw.records[0]['response_is_correct'])
        e['metadata']['role']='retention_anchor'
        with self.assertRaises(ValueError):p.memory.record_training_interaction('polyp_localization',e,{'task_id':'x'})
    def test_test_readonly_with_saved_synthetic_predictions(self):
        p=construct_pipeline(self.prepared,self.storage)
        e=example(role='test');s=copy.deepcopy(p.memory.snapshot())
        with patch.object(p,'_predict',return_value=[{'task_id':'x','answer':'polyp'}]):
            result=p.test([e],record_task_test=False)
        self.assertEqual(result.evaluation['metrics']['correct'],1);self.assertEqual(s,p.memory.snapshot());self.assertEqual(p.model.calls,0)
    def test_train_rejects_test_and_anchors(self):
        p=construct_pipeline(self.prepared,self.storage)
        with self.assertRaises(ValueError):p.train([example(role='test')],val_data=[example()])
    def test_model_visible_prompt_excludes_gold_and_source_path(self):
        from hcl.task_interface import TaskInterface,TaskInterfaceConfig
        e=example();e['answer']='CANARY_PRIVATE_GOLD';e['metadata']['source_record']='PRIVATE_CLASS_PATH';e['metadata']['mask']='PRIVATE_MASK'
        ti=TaskInterface(TaskInterfaceConfig(mode='rule'),model=NoInferenceModel())
        text=ti._render_structuring_prompt(e)
        for token in ['CANARY_PRIVATE_GOLD','PRIVATE_CLASS_PATH','PRIVATE_MASK']:self.assertNotIn(token,text)

    def test_independent_artifact_audit(self):
        from medical_hcl.audit import audit
        r=audit(self.prepared,self.source)
        self.assertTrue(r['passed']);self.assertEqual(r['cases'],468)
    def test_anchor_limit_covers_classes_before_repeated_examples(self):
        rows=self.loader.load_task_spec({'task_name':'classification'},split='anchor',limit=20)
        self.assertEqual({x['answer'] for x in rows},set(VOC))
        self.assertEqual(len({x['metadata']['group_id'] for x in rows}),20)
    def test_static_cannot_update_harness(self):
        p=construct_pipeline(self.prepared,self.storage,arm='static_harness')
        with self.assertRaises(PermissionError):p.train([example(role='h_search')],val_data=[example()])
        self.assertFalse(p.memory.config.enabled)
    def test_actual_optimizer_rejects_forgetting_despite_current_gain(self):
        p=construct_pipeline(self.prepared,self.storage)
        cur=example();anc=example(id='a',role='retention_anchor')
        baseline=[{'task_id':'x','answer':'normal'},{'task_id':'a','answer':'polyp'}]
        stable=p._evaluate_gate([cur,anc],baseline,current_val_examples=[cur],historical_anchor_examples=[anc],split='val')
        candidate=p._evaluate_gate([cur,anc],[{'task_id':'x','answer':'polyp'},{'task_id':'a','answer':'normal'}],current_val_examples=[cur],historical_anchor_examples=[anc],split='val',reference_predictions=baseline)
        c={'candidate_id':'synthetic','component':'router_context'}
        self.assertIsNone(p.optimizer.select_best_candidate([c],{'synthetic':candidate},stable))
        self.assertIn('historical_loss_budget_exceeded',str(p.optimizer.history[-1]))
    def test_actual_optimizer_accepts_gain_without_anchor_loss(self):
        p=construct_pipeline(self.prepared,self.storage)
        cur=example();anc=example(id='a',role='retention_anchor')
        baseline=[{'task_id':'x','answer':'normal'},{'task_id':'a','answer':'polyp'}]
        stable=p._evaluate_gate([cur,anc],baseline,current_val_examples=[cur],historical_anchor_examples=[anc],split='val')
        candidate=p._evaluate_gate([cur,anc],[{'task_id':'x','answer':'polyp'},{'task_id':'a','answer':'polyp'}],current_val_examples=[cur],historical_anchor_examples=[anc],split='val',reference_predictions=baseline)
        c={'candidate_id':'synthetic','component':'router_context'}
        self.assertEqual(p.optimizer.select_best_candidate([c],{'synthetic':candidate},stable)['candidate_id'],'synthetic')

    def test_readonly_violation_restores_memory_and_checkpoint(self):
        p=construct_pipeline(self.prepared,self.storage);before=copy.deepcopy(p.memory.snapshot())
        def corrupt(*args,**kwargs):
            p.memory.raw.records.append({'malicious':'test-write'})
            return [{'task_id':'x','answer':'polyp'}]
        with patch.object(p,'_predict',side_effect=corrupt):
            with self.assertRaises(AssertionError):p.test([example(role='test')],record_task_test=False)
        self.assertEqual(p.memory.snapshot(),before)
        self.assertFalse(p.checkpoint.path.exists())
    def test_heldout_individuals_absent_from_candidate_prompts(self):
        p=construct_pipeline(self.prepared,self.storage)
        anchors=[example(id=f'a{i}',role='retention_anchor') for i in range(20)]
        for a in anchors:a['question']='HELDOUT_CANARY_'+a['id']
        p.memory.anchor.records=anchors
        prompts=[]
        class SyntheticJudge:
            def generate(self,prompt,**kwargs):prompts.append(prompt);return '{"candidates":[]}'
        p.judge_model=SyntheticJudge()
        tr=example('polyp_localization',id='h',role='h_search');val=example('polyp_localization',id='v')
        def predictions(examples,**kwargs):
            return [{'task_id':e['task_id'],'answer':{'bbox':[0,0,10,10]} if e['task_name']=='polyp_localization' else 'polyp'} for e in examples]
        with patch.object(p,'_predict',side_effect=predictions),patch.object(p.memory,'consolidate_task',return_value={}):
            p.train([tr],val_data=[val])
        self.assertGreater(len(prompts),0)
        self.assertNotIn('HELDOUT_CANARY',str(prompts))
        self.assertEqual(p.model.calls,0)

    def test_test_subsample_is_blind_to_gold(self):
        spec={'task_name':'classification'}
        before=self.loader.load_task_spec(spec,split='test',limit=2)
        source=self.loader.load(self.prepared/'visible/classification/test.jsonl')
        for x in source:x['answer']='normal' if x['answer']=='polyp' else 'polyp'
        with patch.object(self.loader,'load',return_value=source):
            after=self.loader.load_task_spec(spec,split='test',limit=2)
        self.assertEqual([x['task_id'] for x in before],[x['task_id'] for x in after])
