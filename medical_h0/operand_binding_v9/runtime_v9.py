"""Declared operand-source consistency, with no field-selection oracle."""
import copy
import math
import re
import bootstrap_v9
from jsonschema import Draft202012Validator
from component_state import ComponentState

BINDING = {'type':'object','properties':{
    'parameter':{'type':'string'},'value':{'type':['number','string']},
    'operation':{'enum':['identity','lowercase','multiply']},
    'operands':{'type':'array','minItems':1,'maxItems':8,'items':{
        'type':'object','properties':{'fact_id':{'type':'string'},'value':{'type':['number','string']}},
        'required':['fact_id','value'],'additionalProperties':False}}},
    'required':['parameter','value','operation','operands'],'additionalProperties':False}


def numeric(value):
    return type(value) in (int,float) and math.isfinite(value)


def same(a,b):
    return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9) if numeric(a) and numeric(b) else type(a)==type(b) and a==b


def source_matches(value,text):
    if isinstance(value,str):return value==text
    match=re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)\s*(?:[A-Za-z][A-Za-z0-9/]*)?\s*',text)
    return bool(match) and numeric(value) and same(value,float(match[1]))


class OperandState(ComponentState):
    def __init__(self,tools):
        super().__init__(tools,'mapping')
        spec=next(s['function'] for s in self.schemas if s['function']['name']=='execute_calculator')
        spec['parameters']['properties']['bindings']['items']=copy.deepcopy(BINDING)
        spec['description']=('Execute only supported parameters using explicit operand-source bindings. Each operand has its source fact_id and the exact value extracted from that fact (a numeric scalar may omit a trailing unit). '
            'identity uses exactly one operand unchanged; lowercase uses exactly one text operand; multiply uses two or more numeric operands. '
            'Supply the computed parameter value and every operand. Sources must have been read and match target patient/episode. The tool checks declared source values and arithmetic, but cannot choose the medically correct fields or supply missing operands. '
            'Do not add fact_ids or transformation: the immutable ledger records all declared operands automatically. No value is corrected for you.')
        self.validators={s['function']['name']:Draft202012Validator(s['function']['parameters']) for s in self.schemas}

    def execute(self,args):
        translated=copy.deepcopy(args)
        bindings=[]
        for b in args['bindings']:
            values=[]
            for operand in b['operands']:
                fid=operand['fact_id']
                if fid not in self.read_row_ids:raise ValueError('Operand source must exist and have been read')
                fact=self.fact_rows[fid]
                if any(fact[k]!=self.target[k] for k in ('patient_id','episode_id')):
                    raise ValueError('Operand source is outside target scope')
                if not source_matches(operand['value'],fact['value_text']):
                    raise ValueError('Operand value differs from its cited raw source: '+fid)
                values.append(operand['value'])
            op=b['operation']
            if op in ('identity','lowercase'):
                if len(values)!=1:raise ValueError(op+' requires exactly one operand')
                value=values[0]
                if op=='lowercase':
                    if not isinstance(value,str):raise ValueError('lowercase requires a text operand')
                    value=value.lower()
            elif op=='multiply':
                if len(values)<2 or not all(numeric(v) for v in values):raise ValueError('multiply requires at least two finite numeric operands')
                value=math.prod(values)
                if not math.isfinite(value):raise ValueError('Nonfinite operation result')
            else:raise ValueError('Unsupported operation')
            if not same(b['value'],value):raise ValueError('Declared parameter value disagrees with operation result')
            bindings.append({'parameter':b['parameter'],'value':b['value'],
                'fact_ids':list(dict.fromkeys(o['fact_id'] for o in b['operands'])),
                'transformation':op+'('+', '.join(str(o['value'])+' from '+o['fact_id'] for o in b['operands'])+')'})
        translated['bindings']=bindings
        out=super().execute(translated)
        self.results[-1]['operand_bindings']=copy.deepcopy(args['bindings'])
        return out
