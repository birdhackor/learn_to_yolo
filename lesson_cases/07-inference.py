import torch
from miniyolo.data import collate
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.inference import decode_grid


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    image = torch.zeros(3, 64, 64)
    image[0, 12:28, 8:24] = 1
    image[2, 36:52, 40:56] = 1
    two = {'boxes': torch.tensor([[8.,12.,24.,28.], [40.,36.,56.,52.]]),
           'labels': torch.tensor([0,1])}
    one = {'boxes': two['boxes'][:1], 'labels': two['labels'][:1]}
    empty = {'boxes': torch.empty(0,4), 'labels': torch.empty(0,dtype=torch.long)}
    one_image = image.clone()
    one_image[2,36:52,40:56] = 0  # The one-object image really has no blue pixels.
    assert one_image[2].sum() == 0 and one_image[0].sum() == 256
    images, anns = collate([(image,two), (one_image,one), (torch.zeros_like(image),empty)])
    model = GridDetector(num_classes=2, grid_size=4, width=8).eval()
    with torch.inference_mode():
        raw = model(images)
        smoke = decode_grid(raw, image_size=64, score_threshold=.25, nms_iou=.5)
    assert len(smoke) == 3
    assert all(p['boxes'].shape == (len(p['scores']),4) for p in smoke)
    print('untrained model candidate counts (no quality claim)', [len(p['boxes']) for p in smoke])
    target = build_targets(anns,4,64,2)
    fixture = torch.zeros_like(raw)
    fixture[...,4] = -20
    for b,y,x in target['positive'].nonzero().tolist():
        fixture[b,y,x,:4] = torch.logit(target['box'][b,y,x].clamp(1e-4,1-1e-4))
        fixture[b,y,x,4] = 10
        fixture[b,y,x,5:] = -10
        fixture[b,y,x,5+target['class_ids'][b,y,x].item()] = 10
    decoded = decode_grid(fixture,64,.25,.5)
    assert [len(p['boxes']) for p in decoded] == [2,1,0]
    for predicted, ann in zip(decoded, anns):
        # Tied scores may change ordering; compare each class separately.
        for cls in ann['labels'].unique().tolist():
            assert torch.allclose(predicted['boxes'][predicted['labels']==cls],
                                  ann['boxes'][ann['labels']==cls], atol=.02)
    assert len(decode_grid(fixture[:1],64,.25,.5)) == 1
    print('artificial known-logit fixture counts', [len(p['boxes']) for p in decoded])
    print('fixture red box', decoded[1]['boxes'][0].tolist())
    duplicate = fixture[1:2].clone()
    duplicate[0,1,0,:4] = torch.logit(torch.tensor([.999,.25,.25,.25]))
    duplicate[0,1,0,4:] = torch.tensor([9.,10.,-10.])
    # The neighbor cell predicts almost the same red box, with a lower score.
    before_nms = decode_grid(duplicate,64,.25,1.0)[0]
    after_nms = decode_grid(duplicate,64,.25,.5)[0]
    assert len(before_nms['boxes']) == 2 and len(after_nms['boxes']) == 1
    print('same-class overlapping fixture NMS counts',2,'->',1)


if __name__ == '__main__':
    main()
