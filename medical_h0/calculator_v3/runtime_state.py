"""Episode-local execution state. No gold, scorer, or learned-state access."""
import json
from tools import RestrictedTools, SCHEMAS

class ExecutionState:
    def __init__(self, rows):
        self.tools=RestrictedTools(rows)
        self.cache={};self.results=[];self.facts_read=False;self.no_progress=0
        self.phase='facts';self.turn=0;self.duplicates=0

    def prepare(self, turn):
        self.turn=turn
        return [s for s in SCHEMAS if s['function']['name'] in self.allowed()]

    def allowed(self):
        if self.phase=='facts':return {'read_facts'}
        if self.phase=='submit':return {'submit_answer'}
        return {'read_facts','calculate','search_calculators','get_calculator','execute_calculator','submit_answer'}

    def call(self,name,args):
        if name not in self.allowed():
            self.no_progress+=1
            return {'error':'tool_unavailable_in_phase','phase':self.phase,'available':sorted(self.allowed())},True
        key=name+':'+json.dumps(args,sort_keys=True,allow_nan=False)
        if name!='submit_answer' and key in self.cache:
            self.duplicates+=1;self.no_progress+=1
            return {**self.cache[key],'cached':True},False
        try:
            result=self.tools.call(name,args)
        except (ValueError,TypeError,SyntaxError,OverflowError,ZeroDivisionError,KeyError,RecursionError) as e:
            self.no_progress+=1
            return {'error':type(e).__name__,'detail':str(e)[:160]},True
        self.no_progress=0
        if name=='read_facts':
            self.facts_read=True;self.phase='work'
        elif name=='calculate':
            self.results.append({'id':len(self.results)+1,'expression':args['expression'],'variables':args['variables'],'result':result['result']})
        elif name=='execute_calculator':
            self.results.append({'id':len(self.results)+1,'calculator_id':args['calculator_id'],'parameters':args['parameters'],'result':result['value'],'unit':result['unit']})
        elif name=='submit_answer':self.phase='done'
        self.cache[key]=result
        return result,False

    def snapshot(self):
        return {'phase':self.phase,'remaining_turns':24-self.turn,'facts_read':self.facts_read,
                'available_tools':sorted(self.allowed()),'computed_results':self.results,
                'submitted':self.tools.answer is not None,'no_progress_events':self.no_progress,'note':'Repeated calls return cached results. Errors do not close tools; change invalid inputs or report abstention. Submission remains your explicit action.'}
