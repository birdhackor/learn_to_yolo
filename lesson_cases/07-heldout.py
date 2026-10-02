import torch
from miniyolo.data import ShapeDataset, collate
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from miniyolo.metrics import evaluate_ap


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    targets = [{'boxes': torch.tensor([[8.,12.,24.,28.]]), 'labels': torch.tensor([0])},
               {'boxes': torch.tensor([[40.,36.,56.,52.]]), 'labels': torch.tensor([0])}]
    predictions = [
        {'boxes': torch.tensor([[8.,12.,24.,28.],[0.,0.,4.,4.],[8.,12.,24.,28.]]),
         'scores': torch.tensor([.9,.8,.7]), 'labels': torch.tensor([0,0,0])},
        {'boxes': torch.empty(0,4), 'scores': torch.empty(0),
         'labels': torch.empty(0,dtype=torch.long)}]
    metrics = evaluate_ap(predictions,targets,num_classes=2,iou_threshold=.5)
    assert abs(metrics['map']-.5) < 1e-6
    assert abs(metrics['precision']-1/3) < 1e-6
    assert abs(metrics['recall']-.5) < 1e-6
    assert metrics['ap_per_class'][1] is None
    print('artificial evaluation fixture', metrics)
    train = ShapeDataset(n=4,size=64,seed=7,max_objects=1,allow_empty=False)
    heldout = ShapeDataset(n=4,size=64,seed=901,max_objects=1,allow_empty=False)
    train_images, train_anns = collate([train[i] for i in range(4)])
    heldout_images, heldout_anns = collate([heldout[i] for i in range(4)])
    model = GridDetector(num_classes=2,grid_size=4,width=8)
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    train_target = build_targets(train_anns,4,64,2)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = grid_loss(model(train_images),train_target)['total']
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predicted = decode_grid(model(heldout_images), image_size=64, score_threshold=.01, nms_iou=.5)
    result = evaluate_ap(predicted,heldout_anns,num_classes=2,iou_threshold=.5)
    assert 0 <= result['map'] <= 1
    print('3-step held-out PIPELINE SMOKE, not trained detector evidence', result)


if __name__ == '__main__':
    main()
