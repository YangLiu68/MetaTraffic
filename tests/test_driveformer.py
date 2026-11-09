import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metatraffic3d.driving.transformer import HIST, HORIZON, STRIDE, history_index, make_driveformer  # noqa: E402


def test_history_ages_are_real_time_gaps():
    idx, ages = history_index(0)
    assert list(idx) == [0] * HIST and np.allclose(ages, 0)  # episode start: repeat the first frame
    idx, ages = history_index(20)
    assert list(idx) == [20 - k * STRIDE for k in range(HIST - 1, -1, -1)]
    assert np.allclose(ages, [0.6, 0.4, 0.2, 0.0])


def test_driveformer_outputs_and_gradients():
    m = make_driveformer(n_cams=4)
    b = 3
    img = torch.randn(b, HIST, 4, 15, 512)
    ages = torch.tensor(np.stack([history_index(t)[1] for t in (0, 3, 30)]), dtype=torch.float32)
    states = torch.randn(b, HIST, 12)
    cmd = torch.zeros(b, 7)
    cmd[:, 1] = 1
    mean, gear, chunk, value = m(img, ages, states, cmd)
    assert mean.shape == (b, 3) and gear.shape == (b, 7) and chunk.shape == (b, HORIZON, 3) and value.shape == (b,)
    assert (mean[:, 0].abs() <= 1).all() and (mean[:, 1:] >= 0).all() and (mean[:, 1:] <= 1).all()
    (mean.sum() + chunk.sum() + value.sum() + gear.sum()).backward()
    assert m.time_proj.weight.grad is not None  # time embedding is used


def test_time_matters():
    torch.manual_seed(0)
    m = make_driveformer(n_cams=1).eval()
    img = torch.randn(1, HIST, 1, 15, 512)
    states, cmd = torch.randn(1, HIST, 12), torch.zeros(1, 7)
    a = m(img, torch.tensor([[0.6, 0.4, 0.2, 0.0]]), states, cmd)[0]
    b = m(img, torch.tensor([[0.0, 0.0, 0.0, 0.0]]), states, cmd)[0]
    assert not torch.allclose(a, b)  # the same frames at different ages give different outputs
