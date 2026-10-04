"""CPU image inference for this book's GridDetector checkpoint format.

python scripts/detect_image.py --image my.png --checkpoint checkpoint.pt

Draws the boxes scoring at least the book's display threshold 0.25 (--score-threshold changes it).
Without --output, the PNG and its same-name JSON go to the repo's git-ignored artifacts/runs/predictions/.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniyolo.geometry import letterbox, undo_letterbox
from miniyolo.models import GridDetector
from miniyolo.inference import decode_grid

# The book's display threshold. A checkpoint's score_threshold (.05 from miniyolo.train) is the low
# candidate cutoff kept for AP evaluation, not a drawing choice: every grid cell of even the untrained
# two-class model passes .05, since each cell scores about .06.
DISPLAY_SCORE_THRESHOLD = .25


def load_grid_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or any(k not in checkpoint for k in ('model_state_dict', 'config', 'class_names')):
        raise ValueError('checkpoint needs model_state_dict, config and ordered class_names')
    config, classes = checkpoint['config'], checkpoint['class_names']
    if not isinstance(config, dict):
        raise ValueError('checkpoint config must be a dictionary')
    for key in ('image_size', 'grid_size', 'width'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'checkpoint config {key} must be a positive integer')
    if not isinstance(classes, list) or not classes or any(type(c) is not str or not c for c in classes):
        raise ValueError('class_names must be a nonempty ordered list of names')
    if len(set(classes)) != len(classes):
        raise ValueError('class_names must be unique')
    if 'num_classes' in config and (type(config['num_classes']) is not int or config['num_classes'] != len(classes)):
        raise ValueError('num_classes and class_names disagree')
    model = GridDetector(num_classes=len(classes),grid_size=config['grid_size'],width=config['width'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('checkpoint parameters contain nonfinite values')
    model.eval()
    return model, config, classes


def detect_image(image_path, checkpoint_path, output_path, score_threshold=DISPLAY_SCORE_THRESHOLD, nms_iou=None):
    model, config, classes = load_grid_checkpoint(checkpoint_path)
    suppression = config.get('nms_iou', .5) if nms_iou is None else nms_iou
    if not 0 <= score_threshold <= 1 or not 0 <= suppression <= 1:
        raise ValueError('score/NMS thresholds must be in [0,1]')
    with Image.open(image_path) as opened:
        original = opened.convert('RGB')
    rgb = np.asarray(original).copy()
    tensor = torch.from_numpy(rgb).permute(2,0,1).float()/255
    prepared, _, metadata = letterbox(tensor,torch.empty(0,4),size=config['image_size'])
    with torch.inference_mode():
        raw = model(prepared.unsqueeze(0))
        if not torch.isfinite(raw).all():
            raise ValueError('model produced nonfinite outputs')
        prediction = decode_grid(raw,image_size=config['image_size'],
                                 score_threshold=score_threshold,nms_iou=suppression)[0]
    boxes = undo_letterbox(prediction['boxes'],metadata)
    # A box lying entirely inside padding can become zero area after clipping.
    keep = (boxes[:,2:] > boxes[:,:2]).all(dim=1)
    boxes, scores, labels = boxes[keep], prediction['scores'][keep], prediction['labels'][keep]
    report = {'image_path':str(image_path),'checkpoint_path':str(checkpoint_path),
              'original_size':{'width':original.width,'height':original.height},
              'class_names':classes,'score_threshold':score_threshold,'nms_iou':suppression,
              'boxes':boxes.tolist(),'scores':scores.tolist(),'labels':labels.tolist()}
    canvas = original.copy()
    drawing = ImageDraw.Draw(canvas)
    for box,score,label in zip(report['boxes'],report['scores'],report['labels']):
        drawing.rectangle(box,outline=(255,165,0),width=2)
        drawing.text((box[0],max(0,box[1]-12)),f'{classes[label]} {score:.2f}',fill=(255,165,0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output_path)
    output_path.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',default=str(ROOT/'artifacts/runs/predictions/my-image.png'),
                        help='annotated PNG; the JSON is saved next to it with the same name')
    parser.add_argument('--score-threshold',type=float,default=DISPLAY_SCORE_THRESHOLD,
                        help="lowest score to draw (default 0.25, the book's display threshold; "
                             "not the checkpoint's score_threshold, the candidate cutoff kept for AP evaluation)")
    parser.add_argument('--nms-iou',type=float,help="NMS IoU threshold (default: the checkpoint config's nms_iou, else 0.5)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = detect_image(args.image,args.checkpoint,args.output,args.score_threshold,args.nms_iou)
    print(json.dumps({'boxes':len(report['boxes']),'score_threshold':report['score_threshold'],
                      'nms_iou':report['nms_iou'],'output':args.output,
                      'class_names':report['class_names']},ensure_ascii=False))


if __name__ == '__main__':
    main()
