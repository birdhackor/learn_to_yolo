"""A hand-specified Gram example, not DINOv3 training or a pretrained-model benchmark."""
import json

import torch
import torch.nn.functional as F


def main():
    torch.set_num_threads(2)
    anchor = F.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), dim=-1)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    rotated = anchor @ rotation
    constant = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    anchor_gram = anchor @ anchor.T
    rotated_gram = rotated @ rotated.T
    constant_gram = constant @ constant.T
    rotated_gram_mse = (rotated_gram - anchor_gram).square().mean()
    constant_gram_mse = (constant_gram - anchor_gram).square().mean()
    feature_mse = (rotated - anchor).square().mean()
    assert torch.allclose(anchor_gram, rotated_gram, atol=1e-7)
    assert rotated_gram_mse == 0 and feature_mse > 0 and constant_gram_mse > 0
    assert torch.allclose(anchor.norm(dim=-1), torch.ones(3))
    assert torch.allclose(torch.diag(anchor_gram), torch.ones(3))
    print(json.dumps({
        "mode": "hand-specified normalized patch relations; no optimizer, no downloads",
        "anchor_features": anchor.tolist(), "anchor_gram": anchor_gram.tolist(),
        "rotated_feature_mse": float(feature_mse),
        "rotated_gram_mse": float(rotated_gram_mse),
        "constant_gram": constant_gram.tolist(),
        "constant_gram_mse": float(constant_gram_mse),
        "scope": "Mean squared Gram difference illustrates relation preservation; not the complete DINOv3 loss or schedule.",
    }, indent=2))


if __name__ == "__main__":
    main()
