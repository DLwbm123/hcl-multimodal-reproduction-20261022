"""Episode binding and recoverable compact observations. No evaluator access."""
import json
from tools import RestrictedTools,SCHEMAS

class ExecutionState:
    def __init__(self,rows):
        self.tools=RestrictedTools(rows);self.cache={};self.observations={};self.results=[];self.errors=[]
        self.episodes={r['episode_id'] for r in rows};self.facts_read=False;self.phase='facts';self.turn=0;self.duplicates=0;self.no_progress=0
        self.fact_rows={f'fact_{i+1:04d}':dict(row) for i,row in enumerate(rows)};self.fact_observations=[];self.read_row_ids=set();self.request_count=0
    def allowed(self):
        if self.phase=='facts':return {'read_facts'}
        if self.phase=='done':return set()
        return {s['function']['name'] for s in SCHEMAS}
    def prepare(self,turn):
        self.turn=turn;return [s for s in SCHEMAS if s['function']['name'] in self.allowed()]
    def validate_episode(self,ep):
        if not isinstance(ep,str) or ep not in self.episodes:raise ValueError('episode_id is not in the visible facts')
    def submission(self,args):
        rows=[]
        for item in args['calculators']:
            if 'result_id' in item:
                found=[x for x in self.results if x['result_id']==item['result_id']]
                if len(found)!=1:raise ValueError('unknown result_id')
                bound=found[0]
                name=bound.get('name',item.get('name'))
                if not name:raise ValueError('arithmetic result needs an explicit calculator name')
                if ('episode_id' in item and item['episode_id']!=bound['episode_id']) or (bound.get('name') and 'name' in item and item['name']!=bound['name']):
                    raise ValueError('submission conflicts with bound result identity or episode')
                row={'name':name,'episode_id':bound['episode_id'],'value':item.get('value',bound['value'])}
            else:row=dict(item)
            self.validate_episode(row['episode_id']);rows.append(row)
        return {'calculators':rows}
    def bind_sources(self,args):
        patient=args.get('patient_id');sources=args.get('input_sources')
        if patient not in {r['patient_id'] for r in self.tools.rows}:raise ValueError('unknown patient_id')
        if not any(r['patient_id']==patient and r['episode_id']==args['episode_id'] for r in self.tools.rows):raise ValueError('output episode does not belong to selected patient')
        if not isinstance(sources,dict) or set(sources)!=set(args['parameters']):raise ValueError('input_sources must cover every parameter exactly')
        saved={};dependencies=set()
        for param,source in sources.items():
            if not isinstance(source,dict) or set(source)-{'fact_ids','result_ids','transformation'}:raise ValueError('invalid source binding')
            fs=source.get('fact_ids',[]);rs=source.get('result_ids',[])
            if not isinstance(fs,list) or not isinstance(rs,list) or not fs+rs:raise ValueError('explicit evidence required for '+param)
            facts=[];results=[]
            for fid in fs:
                if fid not in self.read_row_ids or fid not in self.fact_rows:raise ValueError('source fact must have been read')
                row=self.fact_rows[fid]
                if row['patient_id']!=patient:raise ValueError('cross-patient input is forbidden')
                facts.append({'fact_id':fid,**row})
            for rid in rs:
                found=[r for r in self.results if r['result_id']==rid and r.get('patient_id')==patient]
                if len(found)!=1:raise ValueError('unknown or cross-patient dependency')
                results.append(rid);dependencies.add(rid)
            explanation=source.get('transformation','')
            if not isinstance(explanation,str) or not 0<len(explanation)<=500:raise ValueError('explain extraction or conversion in transformation')
            saved[param]={'facts':facts,'result_ids':results,'transformation':explanation,'normalized_value':args['parameters'][param]}
        return {'patient_id':patient,'input_bindings':saved,'dependency_result_ids':sorted(dependencies),'provenance_semantics':'source existence/patient checked; value extraction, units and clinical relevance remain claims to audit'}
    def call(self,name,args):
        self.request_count+=1
        if self.request_count>80:return {'error':'tool_call_limit'},True
        if name not in self.allowed():return {'error':'tool_unavailable_in_phase','phase':self.phase},True
        key=name+':'+json.dumps(args,sort_keys=True,allow_nan=False)
        if name!='submit_answer' and name!='get_observation' and key in self.cache:
            oid,error=self.cache[key];self.duplicates+=1;self.no_progress+=1
            return {'observation_id':oid,'cached':True,'status':'error' if error else 'success','note':'Identical request adds no evidence. Use get_observation to reread; change invalid inputs or continue another subtask.'},error
        try:
            if name=='get_observation':
                if set(args)!={'observation_id'} or args['observation_id'] not in self.observations:raise ValueError('unknown observation_id')
                self.no_progress+=1
                saved=self.observations[args['observation_id']]
                return {**saved,'retrieved':True},bool(saved.get('error'))
            if name in ('calculate','execute_calculator'):self.validate_episode(args.get('episode_id'))
            if name=='submit_answer':args=self.submission(args)
            binding=self.bind_sources(args) if name=='execute_calculator' else {}
            dispatch={k:v for k,v in args.items() if k not in ('patient_id','input_sources')}
            out=self.tools.call(name,dispatch);error=False;self.no_progress+=1
            if name=='read_facts':
                self.facts_read=True;self.phase='work'
                offset=args.get('offset',0)
                out['rows']=[{**row,'fact_id':f'fact_{offset+i+1:04d}'} for i,row in enumerate(out['rows'])]
                new={row['fact_id'] for row in out['rows']}-self.read_row_ids
                self.read_row_ids.update(row['fact_id'] for row in out['rows'])
                if new:self.no_progress=0
            if name in ('calculate','execute_calculator'):
                bound={'result_id':f'result_{len(self.results)+1:03d}','episode_id':args['episode_id'],'value':out['value'] if name=='execute_calculator' else out['result']}
                if name=='execute_calculator':bound.update(**binding,formula_version='medical-calc-mcp-45f648d9-plus-preserved-news2-patch',calculator_id=out['calculator_id'],name=out['name'],unit=out['unit'],parameters=args['parameters'])
                else:bound.update(expression=args['expression'],variables=args['variables'])
                self.no_progress=0;self.results.append(bound);out={**out,'result_id':bound['result_id']}
            if name=='submit_answer':self.phase='done'
        except (ValueError,TypeError,SyntaxError,OverflowError,ZeroDivisionError,KeyError,RecursionError) as exc:
            out={'error':type(exc).__name__,'detail':str(exc)[:300]};error=True;self.no_progress+=1
            self.errors.append({'tool':name,**out})
        oid=f'observation_{len(self.observations)+1:03d}';out={**out,'observation_id':oid}
        self.observations[oid]=out;self.cache[key]=(oid,error)
        if name=='read_facts' and not error:self.fact_observations.append(oid)
        return out,error
    def snapshot(self):
        return {'phase':self.phase,'remaining_turns':24-self.turn,'facts_read':self.facts_read,
                'available_tools':sorted(self.allowed()),'computed_results':[{k:v for k,v in r.items() if k!='input_bindings'} for r in self.results],'recent_errors':self.errors[-4:],
                'known_episodes':sorted(self.episodes),'fact_observations':self.fact_observations,
                'submitted':self.tools.answer is not None,'no_progress_events':self.no_progress,'tool_requests':self.request_count,
                'note':'Results retain their chosen episode. Submit by result_id to preserve it; no gold-based correction. Older observations remain available via get_observation.'}
