import torch


def size_iou(sizes, anchors):
    intersection = torch.minimum(sizes[:,None,:],anchors[None,:,:]).prod(-1)
    return intersection/(sizes.prod(-1)[:,None]+anchors.prod(-1)[None,:]-intersection)


def cluster(sizes, initial, steps=10):
    anchors = initial.clone()
    for _ in range(steps):
        assignment = (1-size_iou(sizes,anchors)).argmin(1)
        updated = anchors.clone()
        for k in range(len(anchors)):
            selected = sizes[assignment==k]
            if len(selected):
                updated[k] = selected.mean(0)
        if torch.allclose(updated,anchors):
            return updated,assignment
        anchors = updated
    return anchors,assignment


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    sizes = torch.tensor([[8.,8.],[9.,8.],[8.,9.],[32.,16.],[30.,16.],[32.,18.]])
    # These six shapes stand for TRAIN annotations after the same input resize.
    anchors, groups = cluster(sizes,sizes[[0,-1]])
    expected = torch.tensor([[25/3,25/3],[94/3,50/3]])
    assert torch.allclose(anchors,expected)
    assert groups.tolist() == [0,0,0,1,1,1]
    baseline = torch.tensor([[16.,16.],[16.,16.]])
    old = size_iou(sizes,baseline).max(1).values.mean()
    new = size_iou(sizes,anchors).max(1).values.mean()
    assert new > old
    median = torch.stack([sizes[groups==k].median(dim=0).values for k in range(2)])
    assert torch.equal(median,torch.tensor([[8.,8.],[32.,16.]]))
    median_fit = size_iou(sizes,median).max(1).values.mean()
    assert abs(median_fit.item()-.9340278) < 1e-6
    assert median_fit > new  # This fixture only; not a general optimal-update theorem.
    # Held-out aspect ratios: neither cluster necessarily fits the new source.
    shifted = torch.tensor([[4.,40.],[40.,4.]])
    shifted_fit = size_iou(shifted,anchors).max(1).values.mean()
    print('groups',groups.tolist(),'anchors pixel',anchors.tolist())
    print('train mean best size IoU',round(old.item(),4),'->',round(new.item(),4))
    print('median anchors',median.tolist(),'train coverage',round(median_fit.item(),4))
    print('new-source shape coverage',round(shifted_fit.item(),4),'not detector AP')
    # Co-scaling the data and anchors leaves size IoU unchanged.
    assert torch.allclose(size_iou(sizes,anchors),size_iou(sizes*2,anchors*2))


if __name__ == '__main__':
    main()
