"""Model-facing capabilities: no paths, SQL, Python, shell, imports or network."""
import ast
import json
import math
import operator
import sqlite3
from pathlib import Path

COLUMNS = ('patient_id','field_name','value_text','observed_at','episode_id','unit','sample_type','source')
OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.Pow:operator.pow, ast.Mod:operator.mod}
COMPARE = {ast.Lt:operator.lt, ast.LtE:operator.le, ast.Gt:operator.gt,
           ast.GtE:operator.ge, ast.Eq:operator.eq, ast.NotEq:operator.ne}
FUNCTIONS = {'sqrt':math.sqrt,'log':math.log,'log10':math.log10,'exp':math.exp,
             'abs':abs,'min':min,'max':max,'round':round}


def number(x):
    if type(x) not in (int,float,bool) or not math.isfinite(x) or abs(x)>1e15:
        raise ValueError('finite numeric value with magnitude <= 1e15 required')
    return x


def calculate(expression, variables):
    if not isinstance(expression,str) or len(expression)>1024:
        raise ValueError('expression must be a string of at most 1024 characters')
    if type(variables) is not dict or len(variables)>64:
        raise ValueError('variables must be an object with at most 64 numbers')
    for k,v in variables.items():
        if not k.isidentifier() or k.startswith('_') or k in FUNCTIONS:
            raise ValueError('invalid variable name')
        number(v)
    tree=ast.parse(expression,mode='eval')
    if sum(1 for _ in ast.walk(tree))>128:
        raise ValueError('expression too complex')
    def evaluate(n, depth=0):
        if depth>20: raise ValueError('expression too deep')
        ev=lambda x:evaluate(x,depth+1)
        if isinstance(n,ast.Constant): return number(n.value)
        if isinstance(n,ast.Name) and n.id in variables: return variables[n.id]
        if isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UAdd,ast.USub):
            return number(ev(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1))
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=ev(n.left),ev(n.right)
            if isinstance(n.op,ast.Pow) and abs(b)>16: raise ValueError('exponent exceeds 16')
            return number(OPS[type(n.op)](a,b))
        if isinstance(n,ast.Compare) and len(n.ops)==1 and type(n.ops[0]) in COMPARE:
            return COMPARE[type(n.ops[0])](ev(n.left),ev(n.comparators[0]))
        if isinstance(n,ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCTIONS and not n.keywords and 1<=len(n.args)<=8:
            args=[ev(a) for a in n.args]
            if n.func.id=='round' and len(args)==2 and (type(args[1]) is not int or abs(args[1])>15):
                raise ValueError('round precision outside bounds')
            return number(FUNCTIONS[n.func.id](*args))
        raise ValueError('unsupported mathematical syntax')
    return evaluate(tree.body)


def schema(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{
        'type':'object','properties':properties,'required':required,'additionalProperties':False}}}

SCHEMAS=[
 schema('search_calculators','Search the fixed medical calculator directory by clinical terms or calculator name. Empty query lists all, paginated. Results are NOT case-specific recommendations.',{'query':{'type':'string'},'offset':{'type':'integer','minimum':0}},[]),
 schema('get_calculator','Read a calculator definition, purpose, parameter schema, units and formula references. Read this before execution; calculator versions are distinct.',{'calculator_id':{'type':'string'}},['calculator_id']),
 schema('execute_calculator','Execute a tested medical formula by calculator_id with explicit typed parameters. Use get_calculator to obtain the exact parameter names, units and options; never guess missing facts.',{'calculator_id':{'type':'string'},'parameters':{'type':'object'}},['calculator_id','parameters']),
 schema('read_facts','Read all current-task facts in pages. No filters or SQL arguments. Distractor patients, units and specimens remain; select clinically relevant rows from the returned table yourself.',
        {'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':100}},[]),
 schema('calculate','Numeric-only expression grammar: + - * / ** %, comparisons, and Python conditional A if CONDITION else B. Example: 1 if x > 2 else 0 with variables {"x":3}. if(...) and ?: are invalid; strings, lists, imports, sum and attribute access are unsupported. Functions: sqrt/log/log10/exp/abs/min/max/round. Every variable must be supplied as a number in variables; values do not persist.',
        {'expression':{'type':'string'},'variables':{'type':'object','additionalProperties':{'type':'number'}}},['expression','variables']),
 schema('submit_answer','Submit final calculator results once. This is the only submission interface; no output filename is needed. Use full standard published calculator names including acronyms, not bare acronyms. Return one row per calculator and episode; null means abstention.',
        {'calculators':{'type':'array','items':{'type':'object','properties':{'name':{'type':'string'},'episode_id':{'type':'string'},'value':{'type':['number','string','null']}},'required':['name','episode_id','value'],'additionalProperties':False}}},['calculators'])]


class RestrictedTools:
    def __init__(self, rows):
        if len(rows)>10000: raise ValueError('fixture row cap exceeded')
        self.rows=[dict(row) for row in rows]
        if any(set(row)!=set(COLUMNS) or any(type(v) is not str or len(v)>4096 for v in row.values()) for row in self.rows):
            raise ValueError('unexpected fixture schema')
        self.answer=None
        self.calls=0

    @classmethod
    def from_fixture(cls, path):
        # The trusted host supplies this path; it is never a model argument.
        path=Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size>4_000_000:
            raise ValueError('invalid fixture file')
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
            db.row_factory=sqlite3.Row
            rows=db.execute('SELECT '+','.join(COLUMNS)+' FROM patient_facts LIMIT 10001').fetchall()
        return cls(rows)

    def call(self,name,args):
        self.calls+=1
        if self.calls>80: raise ValueError('tool call limit exceeded')
        if self.answer is not None: raise ValueError('episode already submitted')
        if type(args) is not dict or len(json.dumps(args,allow_nan=False))>32000:
            raise ValueError('invalid argument object')
        allowed={'search_calculators':{'query','offset'},'get_calculator':{'calculator_id'},'execute_calculator':{'calculator_id','parameters'},'read_facts':{'offset','limit'},'calculate':{'expression','variables'},'submit_answer':{'calculators'}}
        if name not in allowed or set(args)-allowed[name]: raise ValueError('unknown tool or argument')
        if name in ('search_calculators','get_calculator','execute_calculator'):
            from clinical_catalog import search,details,execute
            if name=='search_calculators':return search(**args)
            if name=='get_calculator':return details(args['calculator_id'])
            return execute(args['calculator_id'],args['parameters'])
        if name=='read_facts':
            offset,limit=args.get('offset',0),args.get('limit',100)
            if type(offset) is not int or not 0<=offset<=10000 or type(limit) is not int or not 1<=limit<=100:
                raise ValueError('invalid pagination')
            found=self.rows
            return {'rows':found[offset:offset+limit],'total':len(found),'next_offset':offset+limit if offset+limit<len(found) else None,'note':'Select the required patient, episode, unit and specimen from the returned rows yourself.'}
        if name=='calculate':
            if set(args)!={'expression','variables'}: raise ValueError('expression and variables required')
            return {'result':calculate(**args)}
        if set(args)!={'calculators'} or type(args['calculators']) is not list or len(args['calculators'])>100:
            raise ValueError('expected at most 100 calculator rows')
        for row in args['calculators']:
            if type(row) is not dict or set(row)!={'name','episode_id','value'}:
                raise ValueError('invalid submission row')
            for k in ('name','episode_id'):
                if type(row[k]) is not str or not row[k].strip() or len(row[k])>256:
                    raise ValueError('invalid name or episode')
            v=row['value']
            if v is not None:
                if type(v) is str:
                    if not v or len(v)>256: raise ValueError('invalid class label')
                elif type(v) is bool: raise ValueError('boolean is not a submitted numeric value')
                else: number(v)
        self.answer=json.loads(json.dumps(args,allow_nan=False))
        return {'submitted':True,'count':len(self.answer['calculators'])}
