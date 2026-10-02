import torch
from torch import nn
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from miniyolo.geometry import nms


class TwoScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(nn.Conv2d(3,8,3,2,1),nn.ReLU(),
                                   nn.Conv2d(8,16,3,2,1),nn.ReLU(),
                                   nn.Conv2d(16,16,3,2,1),nn.ReLU())
        self.deep = nn.Sequential(nn.Conv2d(16,32,3,2,1),nn.ReLU())
        self.fine_head = nn.Conv2d(16,7,1)
        self.coarse_head = nn.Conv2d(32,7,1)

    def forward(self,x):
        fine = self.early(x)
        coarse = self.deep(fine)
        return (self.fine_head(fine).permute(0,2,3,1),
                self.coarse_head(coarse).permute(0,2,3,1))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    small = {'boxes':torch.tensor([[5.,5.,13.,13.]]),'labels':torch.tensor([0])}
    large = {'boxes':torch.tensor([[32.,32.,56.,56.]]),'labels':torch.tensor([1])}
    coarse_small = build_targets([small],4,64,2)
    fine_small = build_targets([small],8,64,2)
    assert torch.allclose(coarse_small['box'][0,0,0],torch.tensor([.5625,.5625,.125,.125]))
    assert torch.allclose(fine_small['box'][0,1,1],torch.tensor([.125,.125,.125,.125]))
    image = torch.zeros(1,3,64,64)
    image[0,0,5:13,5:13] = 1
    image[0,2,32:56,32:56] = 1
    model = TwoScale()
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    fine,coarse = model(image)
    assert fine.shape == (1,8,8,7) and coarse.shape == (1,4,4,7)
    # Disjoint scale assignment: small box to fine, large to coarse.
    coarse_large = build_targets([large],4,64,2)
    assert fine_small['objectness'][0,5,5] == 0  # large GT is unassigned to fine
    assert coarse_large['objectness'][0,0,0] == 0  # small GT is unassigned to coarse
    swapped_small = {'boxes':small['boxes'],'labels':torch.tensor([1])}
    assert torch.equal(build_targets([swapped_small],8,64,2)['positive'],fine_small['positive'])
    loss = grid_loss(fine,fine_small)['total'] + grid_loss(coarse,coarse_large)['total']
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert model.fine_head.weight.grad.abs().sum() > 0
    assert model.coarse_head.weight.grad.abs().sum() > 0
    optimizer.step()
    # Encode the SAME small GT to each scale and check both decode paths.
    decoded_scales = []
    for size,target in [(4,coarse_small),(8,fine_small)]:
        raw = torch.zeros(1,size,size,7)
        raw[...,4] = -20  # channel 4 is objectness logit
        b,y,x = target['positive'].nonzero()[0].tolist()
        raw[b,y,x,:4] = torch.logit(target['box'][b,y,x])
        raw[b,y,x,4:] = torch.tensor([10.,10.,-10.])  # obj, class0, class1
        decoded = decode_grid(raw,64,.25,.5)[0]
        assert torch.allclose(decoded['boxes'],small['boxes'],atol=1e-4)
        decoded_scales.append(decoded)
    boxes = torch.cat([p['boxes'] for p in decoded_scales],dim=0)
    scores = torch.cat([p['scores'] for p in decoded_scales],dim=0)
    labels = torch.cat([p['labels'] for p in decoded_scales],dim=0)
    assert boxes.shape == (2,4) and scores.shape == labels.shape == (2,)
    selected = []
    for label in labels.unique():
        indices = torch.where(labels==label)[0]
        selected.append(indices[nms(boxes[indices],scores[indices],iou_threshold=.5)])
    selected = torch.cat(selected)
    assert len(selected) == 1
    assert torch.allclose(boxes[selected],small['boxes'],atol=1e-4)
    candidates = fine.shape[1]*fine.shape[2] + coarse.shape[1]*coarse.shape[2]
    assert candidates == 80
    print('cross-scale duplicate boxes before/after class-wise NMS',len(boxes),'->',len(selected))
    print('stride16 small target',coarse_small['box'][0,0,0].tolist())
    print('stride8 small target',fine_small['box'][0,1,1].tolist())
    print('head shapes',tuple(fine.shape),tuple(coarse.shape),'candidates before score/NMS',candidates)
    print('both heads received gradients; one update; scale encode/decode checked')


if __name__ == '__main__':
    main()
