from pathlib import Path
import numpy as np
from PIL import Image
import torch
from miniyolo.data import ShapeDataset, collate
from miniyolo.geometry import letterbox, undo_letterbox
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from scripts.detect_image import load_grid_checkpoint, detect_image


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    folder = Path("artifacts/lesson-08-own-images")
    folder.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((80,120,3),dtype=np.uint8)
    pixels[10:30,20:60,0] = 255
    path = folder/'my_image.png'
    Image.fromarray(pixels).save(path)
    with Image.open(path) as opened:
        array = np.asarray(opened.convert('RGB')).copy()
    image = torch.from_numpy(array).permute(2,0,1).float()/255
    original_box = torch.tensor([[20.,10.,60.,30.]])
    prepared, input_boxes, meta = letterbox(image,original_box,size=64)
    assert prepared.shape == (3,64,64)
    restored = undo_letterbox(input_boxes,meta)
    assert torch.allclose(restored,original_box,atol=1e-4)
    # Round in float64 to 4 decimals: float32 math returns 60 as 59.999996..., and a float32
    # 10.6667 would still print as 10.66670036...
    print('original CHW',tuple(image.shape),'input box',input_boxes.double().round(decimals=4).tolist())
    print('metadata',meta,'roundtrip',restored.double().round(decimals=4).tolist())
    # Actually update parameters, save them, and apply the saved state_dict.
    dataset = ShapeDataset(n=4,size=64,seed=7,max_objects=1,allow_empty=False)
    train_images, train_anns = collate([dataset[i] for i in range(4)])
    model = GridDetector(num_classes=2,grid_size=4,width=8)
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    before = next(model.parameters()).detach().clone()
    target = build_targets(train_anns,4,64,2)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = grid_loss(model(train_images),target)['total']
        loss.backward()
        optimizer.step()
    assert not torch.equal(before,next(model.parameters()).detach())
    config = {'image_size':64,'grid_size':4,'width':8,'num_classes':2,
              'score_threshold':.05,'nms_iou':.5}
    checkpoint_path = folder/'checkpoint.pt'
    torch.save({'model_state_dict':model.state_dict(),'config':config,
                'class_names':['red rectangle','blue rectangle'],'steps_completed':3},checkpoint_path)
    reloaded, loaded_config, classes = load_grid_checkpoint(checkpoint_path)
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(prepared[None]),reloaded(prepared[None]))
    assert loaded_config == config and len(classes) == 2
    output_path = folder/'prediction.png'
    report = detect_image(path,checkpoint_path,output_path)
    assert output_path.is_file() and output_path.with_suffix('.json').is_file()
    with Image.open(output_path) as saved:
        assert saved.size == (120,80)
    # Boxes are drawn at the display threshold .25, not at config's AP-evaluation cutoff .05.
    assert report['score_threshold'] == .25
    assert all(score >= report['score_threshold'] for score in report['scores'])
    print('3-step checkpoint safely reloaded: exact logits; original-space box count',len(report['boxes']),
          'at score threshold',report['score_threshold'])
    print('prediction PNG and JSON saved:', output_path, output_path.with_suffix('.json'))
    print('pipeline evidence only, not photo detection quality')
    bad_path = folder/'wrong-class-count.pt'
    torch.save({'model_state_dict':model.state_dict(),'config':config,
                'class_names':['red','blue','yellow']},bad_path)
    try:
        load_grid_checkpoint(bad_path)
    except ValueError:
        print('class/config mismatch rejected')
    else:
        raise AssertionError('wrong class count accepted')
    target = build_targets([{'boxes':input_boxes,'labels':torch.tensor([0])}],4,64,2)
    fixture = torch.zeros(1,4,4,7)
    fixture[...,4] = -20
    b,y,x = target['positive'].nonzero()[0].tolist()
    fixture[b,y,x,:4] = torch.logit(target['box'][b,y,x])
    fixture[b,y,x,4] = 10
    fixture[b,y,x,5:] = torch.tensor([10.,-10.])
    known = decode_grid(fixture,64,.25,.5)[0]
    known_original = undo_letterbox(known['boxes'],meta)
    assert torch.allclose(known_original,original_box,atol=1e-3)
    print('artificial known box restored',known_original.double().round(decimals=4).tolist())

if __name__ == '__main__':
    main()
