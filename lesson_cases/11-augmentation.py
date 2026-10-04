import torch


def horizontal_flip(image,boxes):
    width = image.shape[-1]
    result = boxes.clone()
    result[:,0] = width-boxes[:,2]
    result[:,2] = width-boxes[:,0]
    return image.flip(-1),result


def crop(image,boxes,left,top,width,height,min_visibility=.5):
    shifted = boxes-torch.tensor([left,top,left,top],dtype=boxes.dtype)
    clipped = shifted.clone()
    clipped[:,[0,2]] = clipped[:,[0,2]].clamp(0,width)
    clipped[:,[1,3]] = clipped[:,[1,3]].clamp(0,height)
    area = (boxes[:,2:]-boxes[:,:2]).prod(-1)
    visible = (clipped[:,2:]-clipped[:,:2]).clamp(min=0).prod(-1)
    keep = (visible > 0) & (visible/area >= min_visibility)
    return image[:,top:top+height,left:left+width],clipped[keep],keep


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    image = torch.zeros(3,64,64)
    image[0,12:28,8:24] = 1
    boxes = torch.tensor([[8.,12.,24.,28.]])
    labels = torch.tensor([0],dtype=torch.long)
    flipped, transformed = horizontal_flip(image,boxes)
    assert torch.equal(transformed,torch.tensor([[40.,12.,56.,28.]]))
    expected = torch.zeros_like(image)
    expected[0,12:28,40:56] = 1
    assert torch.equal(flipped,expected)
    twice, original_boxes = horizontal_flip(flipped,transformed)
    assert torch.equal(twice,image) and torch.equal(original_boxes,boxes)
    cropped, clipped, keep = crop(image,boxes,16,8,32,32,.5)
    assert torch.equal(clipped,torch.tensor([[0.,4.,8.,20.]]))
    cropped_labels = labels[keep]
    assert cropped_labels.tolist() == [0] and cropped_labels.dtype == torch.long
    assert keep.tolist() == [True]
    expected_crop = torch.zeros(3,32,32)
    expected_crop[0,4:20,0:8] = 1
    assert torch.equal(cropped,expected_crop)
    _,removed,keep_strict = crop(image,boxes,16,8,32,32,.6)
    strict_labels = labels[keep_strict]
    assert removed.shape == (0,4) and keep_strict.tolist() == [False]
    assert strict_labels.shape == (0,) and strict_labels.dtype == torch.long
    # A no-object image keeps empty boxes AND Long[0] labels through both operations.
    empty_image = torch.zeros_like(image)
    empty_labels = torch.empty(0,dtype=torch.long)
    flipped_empty,empty = horizontal_flip(empty_image,torch.empty(0,4))
    cropped_empty,empty,empty_keep = crop(flipped_empty,empty,16,8,32,32,.5)
    assert empty.shape == (0,4) and cropped_empty.sum() == 0
    assert empty_labels[empty_keep].shape == (0,) and empty_labels.dtype == torch.long
    print('flip box',transformed.tolist(),'double flip is identity')
    print('crop box',clipped.tolist(),'visible area 128/256=.5')
    print('labels after visibility .5/.6',cropped_labels.tolist(),strict_labels.tolist())
    print('no-object image: boxes [0,4], labels Long[0], pixels stay zero')


if __name__ == '__main__':
    main()
