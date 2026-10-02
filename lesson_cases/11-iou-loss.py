import math
import torch


def losses(predicted, target):
    pwh = (predicted[2:]-predicted[:2]).clamp(min=1e-6)
    twh = target[2:]-target[:2]
    inter_wh = (torch.minimum(predicted[2:],target[2:])-torch.maximum(predicted[:2],target[:2])).clamp(min=0)
    intersection = inter_wh.prod()
    union = pwh.prod()+twh.prod()-intersection
    iou = intersection/union.clamp(min=1e-6)
    outer_wh = torch.maximum(predicted[2:],target[2:])-torch.minimum(predicted[:2],target[:2])
    outer_area = outer_wh.prod()
    giou = iou-(outer_area-union)/outer_area.clamp(min=1e-6)
    pc = (predicted[:2]+predicted[2:])/2
    tc = (target[:2]+target[2:])/2
    center_penalty = (pc-tc).square().sum()/outer_wh.square().sum().clamp(min=1e-6)
    v = 4/math.pi**2*(torch.atan(twh[0]/twh[1])-torch.atan(pwh[0]/pwh[1])).square()
    alpha = (v/(1-iou+v).clamp(min=1e-6)).detach()
    return {'iou':1-iou,'giou':1-giou,'diou':1-iou+center_penalty,
            'ciou':1-iou+center_penalty+alpha*v}


def box_from_center(center):
    half_size = torch.tensor([8.,8.],dtype=center.dtype)
    return torch.cat([center-half_size,center+half_size])


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    target = torch.tensor([8.,12.,24.,28.])
    center = torch.nn.Parameter(torch.tensor([40.,20.]))
    initial = losses(box_from_center(center),target)
    assert abs(initial['iou'].item()-1) < 1e-6
    assert abs(initial['giou'].item()-1.2) < 1e-6
    assert abs(initial['diou'].item()-(1+576/1856)) < 1e-6
    assert torch.allclose(initial['diou'],initial['ciou'])
    iou_gradient = torch.autograd.grad(initial['iou'],center,retain_graph=True)[0]
    assert torch.equal(iou_gradient,torch.zeros(2))
    giou_gradient = torch.autograd.grad(initial['giou'],center,retain_graph=True)[0]
    assert torch.allclose(giou_gradient,torch.tensor([.02,0.]),atol=1e-6)
    giou_center = torch.nn.Parameter(torch.tensor([40.,20.]))
    giou_optimizer = torch.optim.SGD([giou_center],lr=50)  # .02 * 50 = 1 pixel left
    giou_initial = losses(box_from_center(giou_center),target)['giou']
    giou_optimizer.zero_grad(set_to_none=True)
    giou_initial.backward()
    assert torch.isfinite(giou_center.grad).all()
    giou_optimizer.step()
    giou_after = losses(box_from_center(giou_center),target)['giou']
    assert torch.allclose(giou_center,torch.tensor([39.,20.]),atol=1e-5)
    assert abs(giou_after.item()-(2-512/624)) < 1e-6 and giou_after < giou_initial
    print('GIoU gradient',giou_gradient.tolist(),'after 1-pixel real SGD move',round(giou_after.item(),6))
    shifted_center = torch.tensor([36.,20.],requires_grad=True)  # 4 pixels left, gap remains 4.
    shifted = losses(box_from_center(shifted_center),target)
    assert shifted['iou'].item() == 1
    assert torch.equal(torch.autograd.grad(shifted['iou'],shifted_center)[0],torch.zeros(2))
    assert abs(shifted['diou'].item()-(1+400/(36**2+16**2))) < 1e-6
    optimizer = torch.optim.SGD([center],lr=100)
    optimizer.zero_grad(set_to_none=True)
    initial['diou'].backward()
    assert torch.isfinite(center.grad).all() and center.grad[0] > 0
    before = center.detach().clone()
    optimizer.step()
    after = losses(box_from_center(center),target)
    assert center[0] < before[0] and after['diou'] < initial['diou']
    print('initial losses',{k:round(v.item(),6) for k,v in initial.items()})
    print('non-overlap plain IoU center gradient',iou_gradient.tolist())
    print('DIoU center after one real step',center.detach().tolist(),
          'loss',round(after['diou'].item(),6))
    exact = losses(target,target)
    assert all(abs(value.item()) < 1e-6 for value in exact.values())
    print('exact boxes: all four losses zero')


if __name__ == '__main__':
    main()
