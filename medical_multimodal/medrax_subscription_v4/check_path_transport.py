"""Attachment aliases map one-to-one; unknown paths remain rejectable by remote tools."""
import json
from subscription import normalize_image_arguments
aliases={'/tmp/job/i0.png':'/remote/case/a.png','/tmp/job/i1.png':'/remote/case/b.png'}
raw={'content':'','tool_calls':[{'name':'expert','arguments_json':json.dumps({'image_paths':['/tmp/job/i1.png','/tmp/job/i0.png'],'prompt':'Do not rewrite /tmp/job/i0.png in question text'})},{'name':'classifier','arguments_json':json.dumps({'image_path':'/unknown/image.png'})},{'name':'segment','arguments_json':json.dumps({'image_path':'/tmp/job/i0.png'})}]}
copy=json.loads(json.dumps(raw));out,changes=normalize_image_arguments(raw,aliases)
assert raw==copy and len(changes)==3
args=json.loads(out['tool_calls'][0]['arguments_json']);assert args['image_paths']==['/remote/case/b.png','/remote/case/a.png'] and args['prompt']==json.loads(raw['tool_calls'][0]['arguments_json'])['prompt']
assert json.loads(out['tool_calls'][1]['arguments_json'])['image_path']=='/unknown/image.png'
assert json.loads(out['tool_calls'][2]['arguments_json'])['image_path']=='/remote/case/a.png'
assert normalize_image_arguments({'content':'A','tool_calls':[]},aliases)==({'content':'A','tool_calls':[]},[])
print('PASS: exact alias mapping, image order, unknown-path boundary, raw evidence, final messages')
