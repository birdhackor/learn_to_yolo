import torch
import torch.nn.functional as F


def size_iou(sizes, anchors):
    intersection = torch.minimum(sizes[:,None,:],anchors[None,:,:]).prod(-1)
    return intersection/(sizes.prod(-1)[:,None]+anchors.prod(-1)[None,:]-intersection)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    anchors = torch.tensor([[16.,16.],[8.,8.]])
    box = torch.tensor([8.,12.,24.,28.])
    center = (box[:2]+box[2:])/2
    wh = box[2:]-box[:2]
    grid_xy = torch.floor(center/16).long()
    offsets = center/16-grid_xy
    ious = size_iou(wh[None],anchors)[0]
    best = int(ious.argmax())
    encoded = torch.cat([torch.logit(offsets.clamp(1e-4,1-1e-4)),torch.log(wh/anchors[best])])
    decoded_center = (grid_xy+encoded[:2].sigmoid())*16
    decoded_wh = anchors[best]*encoded[2:].exp()
    decoded = torch.cat([decoded_center-decoded_wh/2,decoded_center+decoded_wh/2])
    assert torch.allclose(decoded,box,atol=.02)
    assert torch.allclose(ious,torch.tensor([1.,.25]))
    assert torch.allclose(encoded[:2],torch.tensor([-9.210240,-1.098612]),atol=1e-5)
    assert torch.allclose(encoded[:2].sigmoid(),torch.tensor([1e-4,.25]),atol=1e-7)
    raw = torch.nn.Parameter(torch.zeros(1,4,4,2,7))
    pos = torch.zeros(1,4,4,2,dtype=torch.bool)
    ignore = torch.zeros_like(pos)
    x,y = grid_xy.tolist()
    pos[0,y,x,best] = True
    for a in range(2):
        if a != best and ious[a] > .2:
            ignore[0,y,x,a] = True
    obj_target = pos.float()
    valid = ~ignore
    optimizer = torch.optim.SGD([raw],lr=.1)
    before = raw.detach().clone()
    # Only the responsible slot regresses; ignore does not enter objectness BCE.
    regression = F.mse_loss(raw[pos][:,:2].sigmoid(), offsets[None])
    regression = regression + F.mse_loss(raw[pos][:,2:4],torch.log(wh/anchors[best])[None])
    objectness = F.binary_cross_entropy_with_logits(raw[...,4][valid],obj_target[valid])
    classification = F.cross_entropy(raw[pos][:,5:],torch.tensor([0]))
    loss = regression+objectness+classification
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert raw.grad[ignore].abs().sum() == 0
    assert raw.grad[~pos][...,:4].abs().sum() == 0
    optimizer.step()
    assert not torch.equal(raw,before)
    print('size IoU',ious.tolist(),'best anchor',best,'log wh',encoded[2:].tolist())
    print('ratio offsets',offsets.tolist(),'encoded logits',encoded[:2].tolist())
    print('positive/ignore/negative',int(pos.sum()),int(ignore.sum()),int((~pos&~ignore).sum()))
    print('decoded',decoded.tolist(),'one slot update completed')


if __name__ == '__main__':
    main()
