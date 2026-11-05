import torch

from vispeech.ema import EMA


def test_ema_tracks_and_restores():
    m = torch.nn.Linear(3, 2)
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    w0 = m.weight.clone()
    with ema.applied(m):
        assert not torch.allclose(m.weight, w0)  # đang dùng bản EMA
    torch.testing.assert_close(m.weight, w0)  # trả lại trọng số gốc
    for _ in range(200):
        ema.update(m)
    torch.testing.assert_close(ema.shadow["weight"], torch.ones(2, 3), atol=1e-4, rtol=0)


def test_ema_state_roundtrip():
    m = torch.nn.Linear(3, 2)
    a = EMA(m)
    a.update(m)
    b = EMA(torch.nn.Linear(3, 2))
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(b.shadow["weight"], a.shadow["weight"])
    assert b.num_updates == 1
