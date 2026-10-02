import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from miniyolo.custom_data import JsonDetectionDataset, validate_annotations
from miniyolo.data import collate
from miniyolo.targets import build_targets
from miniyolo.models import GridDetector
from miniyolo.losses import grid_loss


def expect_rejection(records, classes, reason):
    try:
        validate_annotations(records,classes)
    except ValueError as error:
        print('rejected',reason,':',str(error))
    else:
        raise AssertionError(reason+' was accepted')


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    classes = ['red_rectangle','blue_rectangle','yellow_rectangle']
    records = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for source,split in [('video_A','train'),('video_B','validation'),('video_C','test')]:
            for index in range(2):
                relative = f'{source}/{index}.png'
                pixels = np.zeros((80,120,3),dtype=np.uint8)
                if index == 0:
                    pixels[10:30,20:60,:2] = 255
                path = root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(pixels).save(path)
                records.append({'path':relative,'width':120,'height':80,'source_id':source,
                                'split':split,'boxes':[[20,10,60,30]] if index==0 else [],
                                'labels':[2] if index==0 else []})
        manifest = root/'annotations.json'
        manifest.write_text(json.dumps({'classes':classes,'images':records}))
        validate_annotations(records,classes)
        bad = copy.deepcopy(records); bad[1]['split'] = 'test'
        expect_rejection(bad,classes,'source leakage')
        bad = copy.deepcopy(records); bad[0]['boxes'] = [[20,10],[60,30]]
        expect_rejection(bad,classes,'malformed box rows')
        bad[0]['labels'] = [2,2]
        expect_rejection(bad,classes,'box row length with matching label count')
        bad = copy.deepcopy(records); bad[0]['labels'] = [True]
        expect_rejection(bad,classes,'boolean class id')
        bad = copy.deepcopy(records); bad[1]['width'] = 0; bad[1]['height'] = -1
        expect_rejection(bad,classes,'empty-image dimensions')
        bad = copy.deepcopy(records); bad[0]['boxes'][0][0] = float('nan')
        expect_rejection(bad,classes,'nonfinite coordinate')
        dataset = JsonDetectionDataset(manifest,root=root,split='train',image_size=64)
        assert len(dataset) == 2 and dataset.classes == classes
        images, anns = collate([dataset[i] for i in range(len(dataset))])
        assert images.shape == (2,3,64,64) and images.dtype == torch.float32
        assert torch.equal(images[0,0],images[0,1]) and images[0,2].sum() == 0
        assert images[1].sum() == 0 and anns[1]['boxes'].shape == (0,4)
        assert torch.allclose(anns[0]['boxes'],torch.tensor([[10.666667,15.375,32.,26.125]]),atol=1e-4)
        assert anns[0]['labels'].tolist() == [2]
        wrong = copy.deepcopy(records); wrong[0]['width'] = 121
        wrong_manifest = root/'wrong-size.json'
        wrong_manifest.write_text(json.dumps({'classes':classes,'images':wrong}))
        try:
            JsonDetectionDataset(wrong_manifest,root=root,split='train')
        except ValueError:
            print('rejected actual image/annotation size mismatch')
        else:
            raise AssertionError('actual image size mismatch accepted')
        target = build_targets(anns,grid_size=4,image_size=64,num_classes=len(classes))
        model = GridDetector(num_classes=len(classes),grid_size=4,width=8)
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        before = next(model.parameters()).detach().clone()
        prediction = model(images)
        assert prediction.shape == (2,4,4,8)
        loss = grid_loss(prediction,target)['total']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before,next(model.parameters()).detach())
        print('6 real PNG files loaded/verified; counts by split',
              {s:sum(r['split']==s for r in records) for s in ['train','validation','test']})
        print('synchronized input box',anns[0]['boxes'].tolist())
        print('new class 2; head',tuple(prediction.shape),'one training step',round(loss.item(),4))


if __name__ == '__main__':
    main()
