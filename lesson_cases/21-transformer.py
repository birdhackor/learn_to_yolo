"""追蹤Pre-LN兩次residual，再觀察位置固定與token整體重排的差異。"""

import json

import torch

from miniyolo.vision_data import ColorRectangles
from miniyolo.vision_transformer import TinyViT


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = TinyViT().eval()
    image = ColorRectangles(samples=128).images[3:4]
    with torch.no_grad():
        embedded = model.embed_tokens(image)
        block = model.blocks[0]
        attention_update = block.attention(block.norm1(embedded))
        after_attention = embedded + attention_update
        mlp_update = block.mlp(block.norm2(after_attention))
        after_mlp = after_attention + mlp_update
        assert torch.equal(after_mlp, block(embedded))
        features = model.forward_features(image)
        logits = model(image)
        # CLS維持第0位，其餘patch逆序；先不給位置，encoder是排列等變的。
        permutation = torch.cat((torch.tensor([0]), torch.arange(16, 0, -1)))
        no_position = model.embed_tokens(image, use_position=False)
        encoded = model.encode_tokens(no_position)
        reordered_encoded = model.encode_tokens(no_position[:, permutation])
        equivariance_error = float((reordered_encoded - encoded[:, permutation]).abs().max())
        assert torch.allclose(reordered_encoded, encoded[:, permutation], atol=2e-6)
        # 這次只重排patch內容，位置向量仍在原座位；CLS輸出會改變。
        contents_reordered = no_position[:, permutation] + model.pos_embed
        original_with_position = no_position + model.pos_embed
        position_change = float((model.encode_tokens(contents_reordered)[:, 0]
                                 - model.encode_tokens(original_with_position)[:, 0]).abs().max())
        assert position_change > 1e-6
    result = {
        "event": "transformer", "block_count": model.depth,
        "input_tokens_shape": list(embedded.shape), "pre_ln": True,
        "first_attention_update_norm": float(attention_update.norm()),
        "first_mlp_update_norm": float(mlp_update.norm()),
        "manual_residual_matches_block": bool(torch.equal(after_mlp, block(embedded))),
        "cls_feature_shape": list(features["cls"].shape),
        "patch_feature_shape": list(features["patches"].shape), "logit_shape": list(logits.shape),
        "permutation": permutation.tolist(), "without_position_equivariance_max_error": equivariance_error,
        "with_fixed_position_cls_max_change": position_change,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "limitation": "Untrained mechanism check; changed CLS values do not show successful color classification.",
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
