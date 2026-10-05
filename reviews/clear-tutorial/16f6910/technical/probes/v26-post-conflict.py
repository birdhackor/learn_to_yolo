import ast,pathlib,json,hashlib,types
import torch
p=pathlib.Path(__file__).resolve().parent/'fixed-sources/u-tal.py';tree=ast.parse(p.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaskAlignedAssigner');fun=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='select_highest_overlaps');fun.decorator_list=[]
ns={'torch':torch};exec(compile(ast.Module(body=[fun],type_ignores=[]),str(p),'exec'),ns)
# Same conflict mask and CIoU as the v10 counterexample, padded to seven candidate columns.
mask=torch.tensor([[[1.,0,0,0,0,0,0],[1.,0,0,0,0,0,0],[0.,1,0,0,0,0,0]]])
overlap=torch.tensor([[[.89873755,.89873755,0,0,0,0,0],[.89873755,.89873755,0,0,0,0,0],[1.,1,0,0,0,0,0]]])
metric=torch.tensor([[[.49994075,.00526984,0,0,0,0,0],[.47134867,.00526984,0,0,0,0,0],[.001,.94868326,0,0,0,0,0]]])
owner,fg,after=ns['select_highest_overlaps'](types.SimpleNamespace(topk=7,topk2=1),mask,overlap,3,metric)
assert after.sum(-1).tolist()==[[0.,0.,1.]] and fg.tolist()==[[0.,1,0,0,0,0,0]]
result={'method':'Original AST select_highest_overlaps only, input mask before conflict; verifies the order and row bound without substituting official implementation','commit':'441632cdfd19e22e60a4b1b1999d46326ca51ec4','url':'https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_used':False,'topk':7,'topk2':1,'before':mask.tolist(),'after':after.tolist(),'gt_positive_counts':after.sum(-1).tolist(),'fg':fg.tolist(),'owner':owner.tolist(),'proof':'After the column-wise conflict rewrite, topk_idx has exactly topk2 entries per GT row. mask_pos *= topk_idx can never increase a row count above topk2; masked zero ties cannot create new positives.'}
pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
