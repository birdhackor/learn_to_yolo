import ast,math,json,pathlib,hashlib
import torch
from torch import nn
base=pathlib.Path(__file__).resolve().parent/'fixed-sources'
ns=dict(torch=torch,nn=nn,math=math)
for file,symbol in [('v10-metrics.py','bbox_iou'),('v10-tal.py','TaskAlignedAssigner')]:
    tree=ast.parse((base/file).read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==symbol)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(base/file),'exec'),ns)
a=ns['TaskAlignedAssigner'](topk=1,num_classes=3,alpha=.5,beta=6)
scores=torch.tensor([[[.9,.8,1e-6],[1e-4,1e-4,.9]]])
pred=torch.tensor([[[0.,0.,10.,10.],[0.,0.,10.,10.]]])
points=torch.tensor([[4.,5.],[6.,5.]])
gt=torch.tensor([[[0.,0.,9.,10.],[1.,0.,10.,10.],[0.,0.,10.,10.]]])
labels=torch.tensor([[[0],[1],[2]]]); valid=torch.ones(1,3,1)
a.bs=1;a.n_max_boxes=3
before,metric,overlap=a.get_pos_mask(scores,pred,labels,gt,points,valid)
owner,fg,after=a.select_highest_overlaps(before,overlap,3)
result=a(scores,pred,points,labels,gt,valid)
record={'source':[{ 'file':f,'sha256':hashlib.sha256((base/f).read_bytes()).hexdigest()} for f in ['v10-tal.py','v10-metrics.py']], 'gpu_used':False, 'torch':str(torch.__version__), 'assigner':{'topk':1,'num_classes':3,'alpha':.5,'beta':6,'eps':1e-9},'gt_labels':labels.tolist(),'mask_gt':valid.tolist(),'scores':scores.tolist(),'pred_xyxy':pred.tolist(),'points_xy':points.tolist(),'gt_xyxy':gt.tolist(),'inside':a.select_candidates_in_gts(points,gt).tolist(),'quality':metric.tolist(),'ciou':overlap.tolist(),'before':before.tolist(),'after':after.tolist(),'gt_positive_counts':after.sum(-1).tolist(),'labels':result[0].tolist(),'target_scores':result[2].tolist(),'fg':result[3].tolist(),'owner':result[4].tolist()}
assert before.sum(-1).tolist()==[[1.,1.,1.]]
assert result[4].tolist()==[[2,2]] and result[3].all()
assert after.sum(-1).tolist()==[[0.,0.,2.]]
assert result[2][0,:,2].min()>0
out=pathlib.Path(__file__).with_suffix('.json');out.write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
