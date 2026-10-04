"""An actual QK-softmax-V calculation with the spatial round trip."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    feature = torch.tensor([[[[1., 0.], [1., 0.]], [[0., 1.], [1., 0.]]]])  # [1,2,2,2]
    tokens = feature.flatten(2).transpose(1, 2)  # [1,4,2] in row-major spatial order
    qkv = nn.Linear(2, 6, bias=False)
    with torch.no_grad():
        qkv.weight.copy_(torch.eye(2).repeat(3, 1))
    q, k, v = qkv(tokens).chunk(3, dim=-1)
    attention = (q @ k.transpose(-2, -1) / math.sqrt(2)).softmax(-1)
    output_tokens = attention @ v
    output = output_tokens.transpose(1, 2).reshape_as(feature)
    assert tokens[0].tolist() == [[1., 0.], [0., 1.], [1., 1.], [0., 0.]]
    assert torch.allclose(attention.sum(-1), torch.ones(1, 4))
    row = torch.tensor([1., 0., 1., 0.]).div(math.sqrt(2)).softmax(0)
    assert torch.allclose(attention[0, 0], row)
    assert torch.allclose(output_tokens[0, 0], torch.tensor([row[0] + row[2], row[1] + row[2]]))
    print('tokens:', tokens.tolist())
    # round in float64 so 0.3349 prints as 0.3349, not as float32's nearest value 0.3348999917...
    print('first attention row:', attention[0, 0].detach().double().round(decimals=4).tolist())
    print('first weighted value:', output_tokens[0, 0].detach().double().round(decimals=4).tolist())
    print('shape feature -> tokens -> affinity -> feature:', tuple(feature.shape), tuple(tokens.shape), tuple(attention.shape), tuple(output.shape))
    # Exercise runs before SGD, while the shared projection is still identity.
    # Keep all original-input assertions intact.
    with torch.no_grad():
        exercise_tokens = tokens.detach().clone()
        exercise_tokens[0, 3] = torch.tensor([2., 0.])
        eq, ek, ev = qkv(exercise_tokens).chunk(3, dim=-1)
        exercise_weights = (eq @ ek.transpose(-2, -1) / math.sqrt(2)).softmax(-1)
        exercise_output = exercise_weights @ ev
        expected_row = torch.tensor([1., 0., 1., 2.]).div(math.sqrt(2)).softmax(0)
        assert torch.allclose(exercise_weights[0, 0], expected_row)
        assert torch.allclose(exercise_output[0, 0], torch.tensor([1.3395, .3302]), atol=1e-4)
        print('exercise fourth-token [2,0], first weight row:', exercise_weights[0, 0].double().round(decimals=4).tolist())
        print('exercise first output:', exercise_output[0, 0].double().round(decimals=4).tolist())
    optimizer = torch.optim.SGD(qkv.parameters(), lr=.1)
    optimizer.zero_grad()
    # Halve channel 1: with target=feature the example is symmetric under a channel swap,
    # so from the identity start Q and K would get identical gradients.
    target = feature * torch.tensor([1., .5]).view(1, 2, 1, 1)
    loss = F.mse_loss(output, target)
    loss.backward()
    grad_q, grad_k, grad_v = qkv.weight.grad.chunk(3, dim=0)
    assert all(part.abs().sum() > 0 for part in (grad_q, grad_k, grad_v))
    # allclose, not equal: rounding alone can make mathematically equal gradients differ.
    assert not torch.allclose(grad_q, grad_k)
    assert not torch.allclose(grad_q, grad_v)
    assert not torch.allclose(grad_k, grad_v)
    old = qkv.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, qkv.weight)
    print(f'backward/step through Q,K,V verified; reconstruction loss={loss.item():.4f}')


if __name__ == '__main__':
    main()
