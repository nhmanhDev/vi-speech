import numpy as np
import pytest
import torch

from vispeech.tts.matcha import MatchaConfig, MatchaTTS, generate_path, intersperse, maximum_path, sequence_mask

TINY = MatchaConfig(vocab_size=50, enc_dim=32, enc_layers=1, enc_ff=64, dp_dim=32, dec_channels=(32, 32), dec_mid_blocks=1)


def _ref_mas(value, tx, ty):
    """Thuật toán MAS gốc của Glow-TTS (bản cython), viết lại bằng python thuần."""
    value = value.copy()
    path = np.zeros_like(value)
    neg = -1e9
    for y in range(ty):
        for x in range(max(0, tx + y - ty), min(tx, y + 1)):
            v_cur = neg if x == y else value[x, y - 1]
            v_prev = (0.0 if y == 0 and x == 0 else neg) if x == 0 else value[x - 1, y - 1]
            value[x, y] += max(v_prev, v_cur)
    idx = tx - 1
    for y in range(ty - 1, -1, -1):
        path[idx, y] = 1
        if idx != 0 and (idx == y or value[idx, y - 1] < value[idx - 1, y - 1]):
            idx -= 1
    return path


def test_mas_matches_reference():
    torch.manual_seed(0)
    logp = torch.randn(4, 9, 30)
    xl, yl = torch.tensor([9, 6, 3, 9]), torch.tensor([30, 14, 9, 9])
    mask = sequence_mask(xl, 9)[:, :, None] & sequence_mask(yl, 30)[:, None, :]
    p = maximum_path(logp, mask)
    for b in range(4):
        ref = _ref_mas(logp[b, : xl[b], : yl[b]].numpy(), int(xl[b]), int(yl[b]))
        np.testing.assert_array_equal(p[b, : xl[b], : yl[b]].numpy(), ref)
    assert (p.sum(1)[mask[:, 0, :]] == 1).all()  # mỗi frame đúng 1 ký tự
    assert (p.sum(2)[mask[:, :, 0]] >= 1).all()  # mỗi ký tự >= 1 frame


def test_generate_path():
    d = torch.tensor([[2, 1, 3]])
    p = generate_path(d, torch.ones(1, 3, 6, dtype=torch.bool))
    assert p[0].argmax(0).tolist() == [0, 0, 1, 2, 2, 2]


def test_intersperse():
    assert intersperse([5, 6], 0) == [0, 5, 0, 6, 0]


def test_forward_backward_and_synthesize():
    torch.manual_seed(0)
    m = MatchaTTS(TINY)
    x = torch.randint(1, 50, (2, 15))
    y = torch.randn(2, 100, 60)
    out = m(x, torch.tensor([15, 9]), y, torch.tensor([60, 33]))
    loss = out["dur"] + out["prior"] + out["fm"]
    loss.backward()
    assert all(torch.isfinite(v).all() for v in out.values())
    mel, lens = m.eval().synthesize(x[:1], torch.tensor([15]), n_steps=2)
    assert mel.shape[1] == 100 and mel.shape[2] % m.align == 0 and lens[0] <= mel.shape[2]


def test_decoder_handles_lengths_not_seen_in_tracing():
    m = MatchaTTS(TINY).eval()
    for t in (8, 64, 200):
        z = torch.randn(1, 100, t)
        v = m.decoder(z, torch.ones(1, 1, t, dtype=torch.bool), torch.randn(1, 100, t), torch.tensor([0.3]))
        assert v.shape == z.shape


def test_numpy_istft_matches_torch():
    from vispeech.tts.onnx_tts import istft

    torch.manual_seed(0)
    spec = torch.randn(513, 40, dtype=torch.complex64)
    win = torch.hann_window(1024)
    ref = torch.istft(spec, 1024, 256, 1024, win, center=True).numpy()
    out = istft(spec.abs().numpy(), spec.angle().numpy())
    np.testing.assert_allclose(out[: len(ref)], ref, atol=1e-4)


@pytest.mark.skipif(pytest.importorskip("onnxruntime") is None, reason="onnxruntime")
def test_tts_onnx_matches_torch(tmp_path):
    vocos = pytest.importorskip("vocos")  # noqa: F841 — cần để export vocoder
    from vispeech.tts.export import main as export_main
    from vispeech.tts.onnx_tts import OnnxTTS

    torch.manual_seed(0)
    cfg = MatchaConfig(**{**vars(TINY), "vocab_size": 101})
    m = MatchaTTS(cfg).eval()
    torch.nn.init.zeros_(m.encoder.dp[-1].weight)
    torch.nn.init.constant_(m.encoder.dp[-1].bias, 1.0)
    ck = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "model_config": vars(cfg)}, ck)
    try:
        export_main(["--ckpt", str(ck), "--out", str(tmp_path / "onnx")])
    except OSError:
        pytest.skip("không tải được Vocos (offline)")
    tts = OnnxTTS(str(tmp_path / "onnx"))
    ids = tts.ids("xin chào, việt nam!")
    with torch.no_grad():
        ref, _ = m.synthesize(torch.from_numpy(ids), torch.tensor([ids.shape[1]]), n_steps=3, temperature=0.0)
    out = tts.mel("xin chào, việt nam!", n_steps=3, temperature=0.0)
    np.testing.assert_allclose(out[0], ref[0, :, : out.shape[2]].numpy(), atol=1e-3)
    assert len(tts("xin chào")) > 0


# ------------------------------------------------------------------ prosody (điều khiển ngữ điệu)

PROS = MatchaConfig(**{**vars(TINY), "n_prosody": 3, "pros_dim": 8})


def test_prosody_features_on_real_voice():
    import os

    import soundfile as sf

    from vispeech.tts.prosody import features

    w, sr = sf.read(os.path.join(os.path.dirname(__file__), "fixtures", "infore_00002.flac"), dtype="float32")
    pitch, rng, rate = features(w, sr, n_syllables=22)
    assert 10 < pitch < 17  # giọng nữ InfoRe ~ 180-270 Hz = 10-17 semitone trên 100 Hz
    assert 2 < rng < 15
    assert 0.9 < np.exp(rate) < 6  # âm tiết / giây


def test_offsets_to_z_roundtrip():
    from vispeech.tts.prosody import offsets_to_z

    st = {"mean": [13.0, 7.0, 1.3], "std": [1.0, 2.0, 0.1]}
    assert offsets_to_z(st) == pytest.approx([0, 0, 0])
    z = offsets_to_z(st, pitch_st=2.0, range_scale=1.5, rate=float(np.exp(0.1)))
    assert z == pytest.approx([2.0, 1.75, 1.0])


def test_prosody_changes_output_and_dropout():
    torch.manual_seed(0)
    m = MatchaTTS(PROS).eval()
    torch.nn.init.zeros_(m.encoder.dp[-1].weight)
    torch.nn.init.constant_(m.encoder.dp[-1].bias, 1.0)
    x = torch.randint(1, 50, (1, 12))
    a, _ = m.synthesize(x, torch.tensor([12]), n_steps=2, temperature=0.0)
    b, _ = m.synthesize(x, torch.tensor([12]), n_steps=2, temperature=0.0, prosody=torch.tensor([[2.0, 1.0, -1.0]]))
    c, _ = m.synthesize(x, torch.tensor([12]), n_steps=2, temperature=0.0, prosody=torch.zeros(1, 3))
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, c)  # không truyền prosody = ngữ điệu trung bình (z = 0)
    m.train()
    out = m(x.repeat(4, 1), torch.full((4,), 12), torch.randn(4, 100, 40), torch.full((4,), 40), prosody=torch.ones(4, 3))
    (out["dur"] + out["prior"] + out["fm"]).backward()
    assert m.pros_proj[0].weight.grad is not None


def test_tts_onnx_with_prosody_matches_torch(tmp_path):
    pytest.importorskip("vocos")
    from vispeech.tts.export import main as export_main
    from vispeech.tts.onnx_tts import OnnxTTS

    torch.manual_seed(0)
    cfg = MatchaConfig(**{**vars(PROS), "vocab_size": 101})
    m = MatchaTTS(cfg).eval()
    torch.nn.init.zeros_(m.encoder.dp[-1].weight)
    torch.nn.init.constant_(m.encoder.dp[-1].bias, 1.0)
    stats = {"infore": {"mean": [13.0, 7.0, 1.3], "std": [1.0, 2.0, 0.1], "n": 10}}
    ck = tmp_path / "m.pt"
    torch.save({"model": m.state_dict(), "model_config": vars(cfg), "prosody_stats": stats}, ck)
    try:
        export_main(["--ckpt", str(ck), "--out", str(tmp_path / "onnx")])
    except OSError:
        pytest.skip("không tải được Vocos (offline)")
    tts = OnnxTTS(str(tmp_path / "onnx"))
    ids = tts.ids("xin chào")
    z = torch.tensor(tts.prosody(pitch_st=2.0, rate=1.1))
    with torch.no_grad():
        ref, _ = m.synthesize(torch.from_numpy(ids), torch.tensor([ids.shape[1]]), n_steps=3, temperature=0.0, prosody=z)
    out = tts.mel("xin chào", n_steps=3, temperature=0.0, pitch_st=2.0, rate=1.1)
    np.testing.assert_allclose(out[0], ref[0, :, : out.shape[2]].numpy(), atol=1e-3)
