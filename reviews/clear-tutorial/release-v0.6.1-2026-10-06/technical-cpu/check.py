import hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from checkpoint_copy import save_checkpoint, load_checkpoint

torch.set_num_threads(2)
z = torch.tensor([[2.,1.],[0.,0.],[-2.,3.]],dtype=torch.float64,requires_grad=True)
y = torch.tensor([0,1,1],dtype=torch.long)
p = F.softmax(z,dim=1)
ce = F.cross_entropy(z,y)
manual = -p[torch.arange(len(y)),y].log().mean()
grad_ce, = torch.autograd.grad(ce,z,retain_graph=True)
grad_manual, = torch.autograd.grad(manual,z)
assert torch.allclose(p.sum(1),torch.ones(3,dtype=torch.float64))
assert torch.allclose(ce,manual,atol=1e-14)
assert torch.allclose(grad_ce,grad_manual,atol=1e-14)
wrong_prob = torch.tensor([[.9,.1]],dtype=torch.float64)
wrong_second = F.softmax(wrong_prob,dim=1).flatten().tolist()
wrong_loss = F.cross_entropy(wrong_prob,torch.tensor([0])).item()
lower_bound = F.cross_entropy(torch.tensor([[1.,0.]],dtype=torch.float64),torch.tensor([0])).item()
assert abs(wrong_second[0]-.6899744811276125)<1e-14
assert abs(lower_bound-math.log1p(math.exp(-1)))<1e-14

random.seed(31); np.random.seed(32); torch.manual_seed(33)
model=torch.nn.Linear(2,1); optimizer=torch.optim.Adam(model.parameters(),lr=.01)
checkpoint=Path('rng-roundtrip.pt')
saved=save_checkpoint(checkpoint,model,optimizer,{},['demo'],0)
def probe():return {'python':[random.random() for _ in range(3)],'numpy':np.random.random(3).tolist(),'torch':torch.rand(3).tolist()}
expected=probe()
restored=torch.nn.Linear(2,1); restored_optimizer=torch.optim.Adam(restored.parameters(),lr=.01)
load_checkpoint(checkpoint,restored,restored_optimizer)
observed=probe()
assert expected==observed
result={'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads(),'ce_from_logits':ce.item(),'negative_log_correct_probability_mean':manual.item(),'ce_gradient_matches_probability_formula':True,'double_softmax_example':wrong_second,'double_softmax_ce':wrong_loss,'two_class_probability_input_ce_lower_bound':lower_bound,'rng_restored_next_draws_exact':expected==observed,'rng_probe':observed,'copied_checkpoint_sha256':hashlib.sha256(Path('checkpoint_copy.py').read_bytes()).hexdigest(),'limitations':['Only scalar CPU probes for the three global RNGs; no CUDA, local NumPy Generator, DataLoader worker, or resumed optimization parity check.','No lesson training was rerun by this reviewer.']}
Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
