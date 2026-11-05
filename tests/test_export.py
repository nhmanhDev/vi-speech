import os

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from vispeech.audio import LogMel  # noqa: E402
from vispeech.stt.export import main as export_main  # noqa: E402
from vispeech.stt.model import ConformerConfig, ConformerCTC  # noqa: E402
from vispeech.stt.onnx_asr import OfflineASR, StreamingASR  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("exp")
    torch.manual_seed(0)
    cfg = ConformerConfig(d_model=64, n_heads=4, n_layers=2, subsampling_channels=32)
    m = ConformerCTC(cfg).eval()
    ck = os.path.join(d, "m.pt")
    torch.save({"model": m.state_dict(), "model_config": vars(cfg)}, ck)
    export_main(["--ckpt", ck, "--out", str(d / "onnx"), "--chunk", "8", "--left-chunks", "2"])
    return m, str(d / "onnx")


def test_offline_onnx_matches_torch(exported):
    m, d = exported
    wav = (np.random.randn(16000 * 2) * 0.1).astype(np.float32)
    feats = LogMel()(wav)
    with torch.no_grad():
        ref = m(torch.from_numpy(feats)[None], torch.tensor([len(feats)]))[0].log_softmax(-1)[0].numpy()
    out = OfflineASR(d, int8=False).sess.run(None, {"feats": feats[None]})[0][0]
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_streaming_onnx_matches_torch_chunked(exported):
    m, d = exported
    wav = (np.random.randn(16000 * 3) * 0.1).astype(np.float32)
    feats = LogMel()(wav)
    with torch.no_grad():
        ref = m(torch.from_numpy(feats)[None], torch.tensor([len(feats)]), chunk=8, left_chunks=2)[0].log_softmax(-1)[0].numpy()
    st = StreamingASR(d, int8=False)
    outs = []
    while len(feats) >= st.win:
        lp, st.k, st.v, st.conv = st.sess.run(None, {"feats": feats[None, : st.win], "k_cache": st.k, "v_cache": st.v,
                                                     "conv_cache": st.conv, "offset": np.array(st.offset, dtype=np.int64)})
        st.offset += 8
        outs.append(lp[0])
        feats = feats[st.hop :]
    s = np.concatenate(outs)
    np.testing.assert_allclose(s, ref[: len(s)], atol=1e-3)


def test_int8_runs_and_streaming_api(exported):
    _, d = exported
    st = StreamingASR(d, int8=True)
    wav = (np.random.randn(16000) * 0.1).astype(np.float32)
    for i in range(0, len(wav), 320):
        st.accept(wav[i : i + 320])
    assert isinstance(st.finish(), str)
    assert st.offset == 0  # finish() reset trạng thái
