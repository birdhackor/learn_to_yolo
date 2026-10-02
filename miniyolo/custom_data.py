"""JSON pixel-xyxy annotations -> RGB letterboxed dataset, without downloads."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import letterbox


def validate_annotations(records, classes):
    """Validate raw rows before tensor conversion, including unselected splits."""
    if not isinstance(classes, list) or not classes or any(type(c) is not str or not c for c in classes):
        raise ValueError('classes must be a nonempty list of names')
    if len(set(classes)) != len(classes):
        raise ValueError('class names must be unique')
    if not isinstance(records, list):
        raise ValueError('images must be a list')
    group_split, paths = {}, set()
    for r in records:
        if not isinstance(r, dict):
            raise ValueError('each image record must be an object')
        for key in ('path', 'width', 'height', 'source_id', 'split', 'boxes', 'labels'):
            if key not in r:
                raise ValueError(f'missing {key}')
        if type(r['path']) is not str or not r['path']:
            raise ValueError('path must be a nonempty relative path')
        path = Path(r['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('path must stay relative to the data root')
        normalized = path.as_posix()
        if normalized in paths:
            raise ValueError('duplicate image')
        paths.add(normalized)
        if type(r['source_id']) is not str or not r['source_id']:
            raise ValueError('source_id must be a nonempty string')
        if r['split'] not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        old = group_split.setdefault(r['source_id'], r['split'])
        if old != r['split']:
            raise ValueError('source leakage')
        if any(type(r[k]) is not int or r[k] <= 0 for k in ('width', 'height')):
            raise ValueError('width/height must be positive integers, including empty images')
        if type(r['boxes']) is not list or type(r['labels']) is not list:
            raise ValueError('boxes/labels must be arrays')
        if len(r['boxes']) != len(r['labels']):
            raise ValueError('box/label row counts differ')
        if any(type(label) is not int or not 0 <= label < len(classes) for label in r['labels']):
            raise ValueError('class ids must be integer indices; bool is invalid')
        for row in r['boxes']:
            if type(row) is not list or len(row) != 4:
                raise ValueError('each box row must have exactly four coordinates')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
                raise ValueError('coordinates must be finite numbers')
            x1, y1, x2, y2 = row
            if not (0 <= x1 < x2 <= r['width'] and 0 <= y1 < y2 <= r['height']):
                raise ValueError('box outside image or nonpositive area')
    return True


class JsonDetectionDataset(Dataset):
    """All records are validated before selecting a split; output follows collate."""
    def __init__(self, annotation_path, root=None, split='train', image_size=64):
        annotation_path = Path(annotation_path)
        contents = json.loads(annotation_path.read_text(encoding='utf-8'))
        if not isinstance(contents, dict) or 'classes' not in contents or 'images' not in contents:
            raise ValueError('manifest needs classes and images')
        validate_annotations(contents['images'], contents['classes'])
        if split not in (None, 'train', 'validation', 'test'):
            raise ValueError('unknown requested split')
        if type(image_size) is not int or image_size <= 0:
            raise ValueError('image_size must be a positive integer')
        self.root = Path(root) if root is not None else annotation_path.parent
        self.classes = contents['classes']
        self.image_size = image_size
        # Verify real files and original dimensions even in the other splits.
        for record in contents['images']:
            path = self.root / record['path']
            with Image.open(path) as image:
                if image.size != (record['width'], record['height']):
                    raise ValueError(f'image size differs from annotation: {record["path"]}')
                image.verify()
        self.records = [r for r in contents['images'] if split is None or r['split'] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['path']) as opened:
            rgb = np.asarray(opened.convert('RGB')).copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        boxes = (torch.tensor(record['boxes'], dtype=torch.float32)
                 if record['boxes'] else torch.empty(0, 4))
        labels = torch.tensor(record['labels'], dtype=torch.long)
        prepared, transformed, _ = letterbox(image, boxes, size=self.image_size)
        return prepared, {'boxes': transformed, 'labels': labels}
