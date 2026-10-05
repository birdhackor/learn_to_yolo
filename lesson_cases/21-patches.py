"""追蹤 32×32 RGB 圖的 patch 順序、共用投影、位置向量與 CLS。"""

import json

import torch

from miniyolo.vision_transformer import PatchEmbedding, TinyViT, patchify


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    # 示意材料：每個8×8 patch 的 R 是編號/16，G=.25，B=.75；不是訓練資料。
    image = torch.empty(1, 3, 32, 32)
    for index in range(16):
        row, col = divmod(index, 4)
        image[0, :, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = torch.tensor(
            [index / 16, .25, .75])[:, None, None]
    raw = patchify(image, 8)
    assert raw.shape == (1, 16, 192)
    assert torch.equal(raw[0, :, 0], torch.arange(16) / 16)
    projection = PatchEmbedding(image_size=32, patch_size=8, embed_dim=3)
    with torch.no_grad():
        projection.projection.weight.zero_()
        projection.projection.bias.zero_()
        for channel in range(3):
            projection.projection.weight[channel, channel] = 1 / 64
    projected = projection(image)
    assert torch.allclose(projected[0, :, 0], torch.arange(16) / 16)
    assert torch.allclose(projected[0, :, 1:], torch.tensor([.25, .75]).expand(16, 2))
    model = TinyViT()
    tokens = model.embed_tokens(image)
    patch_tokens = model.patch_embed(image)
    assert torch.allclose(tokens[:, 0], model.cls_token[:, 0] + model.pos_embed[:, 0])
    assert torch.allclose(tokens[:, 1:], patch_tokens + model.pos_embed[:, 1:])
    # 相同patch內容仍能帶不同位置資訊；pos向量是待訓練參數，並非座標答案。
    uniform = torch.ones_like(image) * .25
    uniform_without_position = model.embed_tokens(uniform, use_position=False)
    uniform_with_position = model.embed_tokens(uniform)
    assert torch.equal(uniform_without_position[:, 1], uniform_without_position[:, 2])
    assert not torch.equal(uniform_with_position[:, 1], uniform_with_position[:, 2])
    result = {
        "event": "patches", "image_shape": list(image.shape), "patch_size": 8,
        "raw_patch_shape": list(raw.shape), "raw_vector_order": "channel,row,column",
        "patch_order": [[i, i // 4, i % 4] for i in range(16)],
        "patch_red_means": projected[0, :, 0].detach().tolist(),
        "projection_shape": list(projected.shape), "projection_demo": "RGB means; hand-set weights",
        "vit_patch_embedding_shape": list(patch_tokens.shape),
        "tokens_with_cls_shape": list(tokens.shape), "cls_index": 0,
        "first_patch_sequence_index": 1, "learned_position_shape": list(model.pos_embed.shape),
        "same_content_equal_without_position": bool(torch.equal(uniform_without_position[:, 1], uniform_without_position[:, 2])),
        "same_content_equal_with_position": bool(torch.equal(uniform_with_position[:, 1], uniform_with_position[:, 2])),
        "full_attention_scores_per_head": {"patch8_with_cls": 17 ** 2, "patch4_with_cls": 65 ** 2},
        "score_count_ratio_with_cls": 65 ** 2 / 17 ** 2,
        "patch_only_score_count_ratio": 64 ** 2 / 16 ** 2,
        "limitation": "Hand-set projection explains token construction; no optimizer update or learned classification here.",
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
