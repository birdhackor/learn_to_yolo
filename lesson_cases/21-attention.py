"""由15.1的QK-softmax-V走到ViT多頭attention，顯示可核對的權重列。"""

import json
import math

import torch
from torch.nn import functional as F

from miniyolo.vision_data import ColorRectangles
from miniyolo.vision_transformer import SelfAttention, TinyViT


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    tokens = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [0., 0.]]])
    attention = SelfAttention(embed_dim=2, num_heads=1)
    with torch.no_grad():
        attention.qkv.weight.copy_(torch.eye(2).repeat(3, 1))
        attention.qkv.bias.zero_()
        attention.projection.weight.copy_(torch.eye(2))
        attention.projection.bias.zero_()
    output, weights = attention(tokens, return_attention=True)
    expected_row = (torch.tensor([1., 0., 1., 0.]) / math.sqrt(2)).softmax(0)
    assert torch.allclose(weights[0, 0, 0], expected_row)
    assert torch.allclose(output[0, 0], torch.tensor([expected_row[0] + expected_row[2], expected_row[1] + expected_row[2]]))
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 4))
    before = attention.qkv.weight.detach().clone()
    optimizer = torch.optim.SGD(attention.parameters(), lr=.1)
    loss = F.mse_loss(output, tokens * torch.tensor([1., .5]))
    optimizer.zero_grad()
    loss.backward()
    gradient_sums = [float(part.abs().sum()) for part in attention.qkv.weight.grad.chunk(3, dim=0)]
    assert all(value > 0 for value in gradient_sums)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in attention.parameters())
    optimizer.step()
    assert not torch.equal(before, attention.qkv.weight)
    model = TinyViT().eval()
    image = ColorRectangles(samples=128).images[3:4]
    embedded = model.embed_tokens(image)
    # 第一個block的attention先看LayerNorm後的tokens；沒有預訓練。
    mixed, multi_weights = model.blocks[0].attention(model.blocks[0].norm1(embedded), return_attention=True)
    assert multi_weights.shape == (1, 4, 17, 17)
    assert torch.allclose(multi_weights.sum(-1), torch.ones(1, 4, 17), atol=1e-6)
    with torch.no_grad():
        # 交換第1與第16個patch，CLS仍在第0位；關閉dropout後檢查排列關係。
        permutation = torch.arange(17)
        permutation[1], permutation[16] = 16, 1
        contents = model.embed_tokens(image, use_position=False)
        without_pos = model.encode_tokens(contents)
        swapped_without_pos = model.encode_tokens(contents[:, permutation])
        cls_without_pos_change = float((without_pos[:, 0] - swapped_without_pos[:, 0]).abs().max())
        patches_equivariance_error = float((swapped_without_pos[:, 1:] - without_pos[:, permutation][:, 1:]).abs().max())
        assert torch.allclose(swapped_without_pos, without_pos[:, permutation], atol=2e-6)
        # 位置保持原槽，只換patch內容。
        fixed_pos_change = float((model.encode_tokens(contents + model.pos_embed)[:, 0]
                                  - model.encode_tokens(contents[:, permutation] + model.pos_embed)[:, 0]).abs().max())
        assert fixed_pos_change > 1e-6
    result = {
        "event": "attention", "manual_tokens": tokens[0].tolist(),
        "first_query_scores_before_scale": [1., 0., 1., 0.], "scale": math.sqrt(2),
        "first_attention_row": weights[0, 0, 0].detach().tolist(),
        "first_weighted_value": output[0, 0].detach().tolist(),
        "manual_weight_shape": list(weights.shape), "manual_loss": float(loss.detach()),
        "q_k_v_gradient_abs_sums": gradient_sums, "qkv_weights_changed": True,
        "vit_tokens_shape": list(embedded.shape), "vit_heads": 4, "vit_head_dim": 8,
        "vit_attention_shape": list(multi_weights.shape), "vit_attention_scores": multi_weights.numel(),
        "vit_output_shape": list(mixed.shape),
        "vit_row_sum_max_error": float((multi_weights.sum(-1) - 1).abs().max().detach()),
        "swap_patch_sequence_indices": [1, 16],
        "without_position_cls_max_change": cls_without_pos_change,
        "without_position_patch_equivariance_max_error": patches_equivariance_error,
        "with_fixed_position_cls_max_change": fixed_pos_change,
        "limitation": "One SGD update validates the mechanism; random ViT attention weights are not explanations of a trained decision.",
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
