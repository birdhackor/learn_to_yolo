"""Verified progressive-weight schedule in a small two-head regression example."""
import json
import torch
from torch import nn
from torch.nn import functional as F


def weights(epoch, epochs, final_many=.1):
    many = max(1 - epoch / max(epochs - 1, 1), 0) * (.8 - final_many) + final_many
    return many, 1 - many


def train(progressive, epochs=30, final_many=.1, fixed_weights=(.8, .2)):
    torch.manual_seed(7)  # paired initialization and fixed data
    features = torch.tensor([[-1.], [0.], [1.], [2.]])
    target = 2 * features + 1
    many, one = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = torch.optim.SGD(list(many.parameters()) + list(one.parameters()), lr=.05)
    history = []
    first_step = {}
    for epoch in range(epochs):
        a, b = weights(epoch, epochs, final_many) if progressive else fixed_weights
        optimizer.zero_grad()
        lm = F.mse_loss(many(features), target)
        one_prediction = one(features)
        lo = F.mse_loss(one_prediction, target)
        if epoch == 0:
            first_step.update(weight=float(one.weight.detach()), bias=float(one.bias.detach()),
                              prediction=one_prediction.detach().flatten().tolist(), one_mse=float(lo.detach()), one_gain=b)
        loss = a * lm + b * lo
        loss.backward()
        assert many.weight.grad is not None and one.weight.grad is not None
        if epoch == 0:
            first_step.update(weighted_dw=float(one.weight.grad), weighted_dbias=float(one.bias.grad))
        optimizer.step()
        if epoch == 0:
            first_step.update(updated_weight=float(one.weight.detach()), updated_bias=float(one.bias.detach()))
        history.append((a, b, float(lo.detach())))
    return F.mse_loss(one(features), target).item(), history, first_step


def total_one_weight(history):
    """Sum of the one-head loss weight b over all epochs of one run."""
    return sum(b for _, b, _ in history)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    final_many = .1  # Change only this value to .3 for the schedule exercise.
    fixed, fixed_history, _ = train(False)
    progressive, history, first_step = train(True, final_many=final_many)
    assert abs(history[0][0] - .8) < 1e-7 and abs(history[-1][0] - final_many) < 1e-7
    # The heads are independent: a scales only the many head's gradient, b only the one head's.
    # Matched control: fixed weights with the schedule's sum of b. Against fixed .8/.2 the schedule
    # differs in both shape and sum of b; against this control only the shape differs.
    matched_one = total_one_weight(history) / len(history)
    matched_weights = (1 - matched_one, matched_one)
    matched, matched_history, _ = train(False, fixed_weights=matched_weights)
    assert abs(total_one_weight(matched_history) - total_one_weight(history)) < 1e-9
    actual = torch.tensor([first_step[k] for k in ('weight', 'bias', 'one_mse', 'weighted_dw', 'weighted_dbias', 'updated_weight', 'updated_bias')])
    expected = torch.tensor([.318423, .313781, 5.866377, -1.146190, -.610803, .375733, .344321])
    assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-6)
    print('many/one weight first:', tuple(round(v, 3) for v in history[0][:2]))
    print('many/one weight last:', tuple(round(v, 3) for v in history[-1][:2]))
    runs = ((f'fixed many/one {tuple(round(v, 3) for v in fixed_history[0][:2])}', fixed, fixed_history),
            ('progressive', progressive, history),
            (f'matched fixed many/one {tuple(round(v, 3) for v in matched_weights)}', matched, matched_history))
    for name, mse, run_history in runs:
        print(f'{name}: sum of b={total_one_weight(run_history):.3f}, one-head MSE={mse:.6f}')
    print('one-head first forward/backward/step:', json.dumps(first_step))
    # STAL-style eligibility only: retain the actual GT for regression.
    gt = torch.tensor([7., 7., 9., 9.])
    points = torch.tensor([[4., 4.], [12., 4.], [4., 12.], [12., 12.]])
    center = (gt[:2] + gt[2:]) / 2
    expanded_size = (gt[2:] - gt[:2]).clamp(min=16)
    pool_box = torch.cat((center - expanded_size / 2, center + expanded_size / 2))
    inside = lambda box: ((points > box[:2]) & (points < box[2:])).all(-1)
    assert inside(gt).sum() == 0 and inside(pool_box).sum() == 4
    assert gt.tolist() == [7., 7., 9., 9.]
    print('small-object eligible points original / expanded:', int(inside(gt).sum()), int(inside(pool_box).sum()))
    print('GT regression box remains:', gt.tolist())
    print('not a full YOLO26/STAL/MuSGD training reproduction')


if __name__ == '__main__':
    main()
