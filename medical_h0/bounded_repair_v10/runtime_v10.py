"""Read-only local diagnostics; strict execution and one corrected call."""
import copy
import json
import math
import unicodedata
import bootstrap_v10
from jsonschema import Draft202012Validator
from runtime_v9 import OperandState, BINDING, source_matches, numeric, same
from runtime_v5 import ExecutionState as FormulaState


def in_scope(target,key,fact):
    stable=target['calculator_id']=='corrected_qt' and key=='sex' and unicodedata.normalize('NFKC',fact['field_name']).casefold().strip()=='sex'
    return fact['patient_id']==target['patient_id'] and (fact['episode_id']==target['episode_id'] or stable)


class RepairState(OperandState):
    def __init__(self,tools,precise=False,generation_limit=12):
        super().__init__(tools)
        self.precise=precise;self.generation_limit=generation_limit
        self.execute_attempts=0;self.repair_exhausted=False;self.diagnostics=[]
        self.binding_validator=Draft202012Validator(BINDING)

    def diagnose(self,args):
        errors=[];checks=[]
        def add(code,path,message,parameter=None,**extra):
            errors.append(dict(code=code,path=path,parameter=parameter,message=message,**extra))
        for e in self.validators['execute_calculator'].iter_errors(args):
            path='/'+ '/'.join(map(str,e.absolute_path))
            param=None
            if isinstance(args,dict) and len(e.absolute_path)>=2 and list(e.absolute_path)[0]=='bindings':
                idx=list(e.absolute_path)[1]
                if isinstance(args.get('bindings'),list) and isinstance(idx,int) and idx<len(args['bindings']) and isinstance(args['bindings'][idx],dict):param=args['bindings'][idx].get('parameter')
            add('SCHEMA_ERROR',path,e.message,param,keyword=e.validator)
        if not isinstance(args,dict):return {'errors':errors,'binding_checks':checks}
        for k in ('calculator_id','patient_id','episode_id'):
            if k in args and args[k]!=self.target[k]:add('TARGET_MISMATCH','/'+k,'Invocation conflicts with explicit target '+k)
        bindings=args.get('bindings');seen=set()
        if not isinstance(bindings,list):return {'errors':errors,'binding_checks':checks}
        for i,b in enumerate(bindings):
            path=f'/bindings/{i}';key=b.get('parameter') if isinstance(b,dict) else None
            core={k:v for k,v in b.items() if k in BINDING['properties']} if isinstance(b,dict) else b
            if list(self.binding_validator.iter_errors(core)):
                checks.append({'parameter':key,'path':path,'status':'not_checked','reason':'required structure or types invalid'});continue
            before=len(errors)
            if key in seen:add('DUPLICATE_PARAMETER',path+'/parameter','Duplicate parameter binding: '+key,key)
            seen.add(key);values=[];sources_ok=True
            for j,o in enumerate(b['operands']):
                opath=path+f'/operands/{j}';fid=o['fact_id'];values.append(o['value'])
                if fid not in self.read_row_ids:
                    add('SOURCE_NOT_READ',opath+'/fact_id','Operand source must exist and have been read',key);sources_ok=False;continue
                f=self.fact_rows[fid]
                if not in_scope(self.target,key,f):add('SOURCE_SCOPE_MISMATCH',opath+'/fact_id','Operand source is outside permitted target scope',key);sources_ok=False
                if not source_matches(o['value'],f['value_text']):
                    add('SOURCE_VALUE_MISMATCH',opath+'/value','Operand value differs from its cited raw source',key,declared_value=o['value'],source_value=f['value_text']);sources_ok=False
            operation=b['operation'];computed=None;operation_checked=True
            if operation in ('identity','lowercase'):
                if len(values)!=1:add('OPERATION_ARITY',path+'/operands',operation+' requires exactly one operand',key);operation_checked=False
                elif operation=='lowercase' and not isinstance(values[0],str):add('OPERATION_TYPE',path+'/operands','lowercase requires a text operand',key);operation_checked=False
                else:computed=values[0].lower() if operation=='lowercase' else values[0]
            elif len(values)<2 or not all(numeric(v) for v in values):
                add('OPERATION_ARITY_OR_TYPE',path+'/operands','multiply requires at least two finite numeric operands',key);operation_checked=False
            else:
                computed=math.prod(values)
                if not math.isfinite(computed):add('NONFINITE_OPERATION',path+'/operands','Nonfinite operation result',key);operation_checked=False
            if operation_checked and not same(b['value'],computed):
                add('OPERATION_RESULT_MISMATCH',path+'/value','Declared parameter value disagrees with operation result',key,operation=operation,declared_value=b['value'],operation_result=computed)
            checks.append({'parameter':key,'path':path,'status':'locally_consistent' if len(errors)==before else 'invalid',
                'source_checks_complete':True,'operation_checked':operation_checked,'schema_valid':not any(e['code']=='SCHEMA_ERROR' and (e['path']==path or e['path'].startswith(path+'/')) for e in errors),
                'note':'Local consistency does not establish correct fields, complete clinical inputs or conflict resolution.'})
        return {'errors':errors,'binding_checks':checks}

    def reject(self,diagnosis):
        self.request_count+=1;self.no_progress+=1
        if self.execute_attempts>=2:self.repair_exhausted=True
        self.diagnostics.append(copy.deepcopy(diagnosis))
        if self.precise:
            out={'status':'validation_error','executed':False,**copy.deepcopy(diagnosis),'repair_attempts_remaining':max(0,2-self.execute_attempts)}
        else:
            e=diagnosis['errors'][0]
            out={'error':'ValidationError' if e['code']=='SCHEMA_ERROR' else 'ValueError','detail':e['message'][:800]}
        oid=f'observation_{len(self.observations)+1:03d}';out['observation_id']=oid
        self.observations[oid]=(copy.deepcopy(out),True);self.observation_tools[oid]='execute_calculator'
        # Keep only the actually exposed error in the model-visible state.
        self.errors.append({'tool':'execute_calculator',**copy.deepcopy(out)})
        return out,True

    def malformed(self,message):
        self.execute_attempts+=1
        return self.reject({'errors':[{'code':'JSON_PARSE_ERROR','path':'/','parameter':None,'message':message}],
                            'binding_checks':[{'status':'not_checked','reason':'invalid JSON'}]})

    def call(self,name,args):
        if name!='execute_calculator':return super().call(name,args)
        self.execute_attempts+=1
        if self.execute_attempts>2:
            return self.reject({'errors':[{'code':'REPAIR_LIMIT','path':'/','parameter':None,'message':'At most one corrected calculation call is allowed'}],'binding_checks':[]})
        diagnosis=self.diagnose(args)
        if diagnosis['errors']:return self.reject(diagnosis)
        out,error=super().call(name,args)
        self.diagnostics.append(copy.deepcopy(diagnosis))
        if error:
            if self.execute_attempts>=2:self.repair_exhausted=True
            if self.precise:
                exposed={'status':'validation_error','executed':False,'errors':[{'code':'EXECUTION_CONTRACT_ERROR','path':'/','parameter':None,'message':out.get('detail',out.get('error','execution error'))}],
                         'binding_checks':diagnosis['binding_checks'],'repair_attempts_remaining':max(0,2-self.execute_attempts),'observation_id':out['observation_id']}
                self.observations[out['observation_id']]=(copy.deepcopy(exposed),True);self.errors[-1]={'tool':name,**copy.deepcopy(exposed)};out=exposed
        return out,error

    def execute(self,args):
        translated=copy.deepcopy(args)
        translated['bindings']=[{'parameter':b['parameter'],'value':b['value'],'fact_ids':list(dict.fromkeys(o['fact_id'] for o in b['operands'])),
            'transformation':b['operation']+'('+', '.join(str(o['value'])+' from '+o['fact_id'] for o in b['operands'])+')'} for b in args['bindings']]
        out=FormulaState.execute(self,translated)
        self.results[-1]['operand_bindings']=copy.deepcopy(args['bindings']);return out

    def snapshot(self):
        out=super().snapshot();out['remaining_turns']=max(0,self.generation_limit-self.turn)
        out['calculation_attempts_used']=self.execute_attempts;out['calculation_attempts_remaining']=max(0,2-self.execute_attempts)
        return out
