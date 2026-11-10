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


def test_pca_keeps_the_dominant_directions_and_matches_torch():
    from metatraffic3d.driving.transformer import PCA

    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(512, 8)))[0].T  # 8 orthonormal directions
    x = rng.normal(size=(2000, 8)) * np.arange(8, 0, -1) @ basis + 0.01 * rng.normal(size=(2000, 512)) + 3.0
    pca = PCA.fit(x, 8)
    assert pca.comp.shape == (8, 512) and pca.explained > 0.99
    z = pca(x[:5].astype(np.float32))
    back = z @ pca.comp + pca.mean
    assert np.abs(back - x[:5]).max() < 0.1  # 8 components reconstruct the data
    zt = pca(torch.as_tensor(x[:5], dtype=torch.float16))
    assert np.allclose(zt.numpy(), z, atol=0.05)
    m = make_driveformer(n_cams=2, feat_dim=8)  # the model takes the compressed width
    out = m(torch.randn(1, HIST, 2, 15, 8), torch.zeros(1, HIST), torch.zeros(1, HIST, 12), torch.zeros(1, 7))
    assert out[0].shape == (1, 3)


def test_checkpointed_training_pass_matches_plain_pass():
    torch.manual_seed(0)
    m = make_driveformer(n_cams=2, feat_dim=16).train()
    for mod in m.modules():  # no dropout, so both paths are deterministic
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.dropout = 0.0
    args = (torch.randn(2, HIST, 2, 15, 16), torch.rand(2, HIST), torch.randn(2, HIST, 12), torch.zeros(2, 7))
    out = m(*args)  # training + grad: checkpointed layers
    with torch.no_grad():
        ref = m(*args)  # plain encoder
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, atol=1e-5)
    out[0].sum().backward()
    assert m.encoder.layers[0].linear1.weight.grad is not None


def test_pedal_loss_recovers_from_a_saturated_brake():
    from metatraffic3d.driving.policy import bc_loss

    logits = torch.tensor([[0.0, -30.0]], requires_grad=True)  # brake squashed deep into sigmoid's flat end
    mean = torch.stack([torch.zeros(1), torch.sigmoid(logits[:, 0]), torch.sigmoid(logits[:, 1])], dim=1)
    target = torch.tensor([[0.0, 0.0, 1.0]])
    gear_logits, gear = torch.zeros(1, 7), torch.tensor([3])
    bc_loss(mean, gear_logits, target, gear)[0].backward(retain_graph=True)
    assert abs(float(logits.grad[0, 1])) < 1e-9  # L1 on the sigmoid output: no signal at all
    logits.grad = None
    bc_loss(mean, gear_logits, target, gear, pedal_logits=logits)[0].backward()
    assert float(logits.grad[0, 1]) < -0.5  # cross-entropy on the logit: full-strength push to brake


def test_one_pedal_rule_turns_half_pressed_pedals_into_a_decision():
    from metatraffic3d.driving.policy import pick_one_pedal

    assert pick_one_pedal(np.array([0.1, 0.09, 0.07])).tolist() == pytest.approx([0.1, 0.09, 0.0])
    assert pick_one_pedal(np.array([0.0, 0.06, 0.15])).tolist() == pytest.approx([0.0, 0.0, 0.15])
    raw = np.array([0.0, 0.3, 0.0], np.float32)
    assert pick_one_pedal(raw) is not raw  # the network output is not changed in place


def test_rl_explores_around_the_run_time_action_with_one_pedal():
    pytest.importorskip("mujoco")
    from metatraffic3d.driving.policy import pick_one_pedal
    from metatraffic3d.driving.rl import action_dist, to_control

    mean = torch.tensor([[0.1, 0.30, 0.02], [-0.2, 0.05, 0.40]])
    d = action_dist(mean, torch.full((3,), -2.3))  # std 0.1
    for i in range(2):  # without noise RL drives exactly as the policy does at run time
        assert to_control(d.mean[i].numpy()).tolist() == pytest.approx(pick_one_pedal(mean[i].numpy()).tolist())
    ctl = np.array([to_control(x) for x in d.sample((2000,))[:, 0].numpy()])
    assert not ((ctl[:, 1] > 0) & (ctl[:, 2] > 0)).any()  # never both pedals
    assert (ctl[:, 2] > 0).mean() < 0.01  # noise (almost) never taps a brake whose mean is near 0
