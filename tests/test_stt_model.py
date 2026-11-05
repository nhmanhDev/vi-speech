import numpy as np
import torch

from vispeech.audio import LogMel, StreamingLogMel
from vispeech.metrics import ErrorRate, edit_distance
from vispeech.stt.model import ConformerConfig, ConformerCTC, chunk_mask, subsampled_lengths

TINY = ConformerConfig(d_model=64, n_heads=4, n_layers=2, subsampling_channels=32)


def test_streaming_logmel_matches_offline():
    wav = np.random.randn(16000).astype(np.float32)
    fe = LogMel()
    st = StreamingLogMel(fe)
    parts = [st.accept(wav[i : i + 317]) for i in range(0, len(wav), 317)]
    np.testing.assert_allclose(np.concatenate(parts), fe(wav), atol=1e-5)


def test_chunk_mask():
    m = chunk_mask(6, chunk=2, left_chunks=0)
    assert m[3].tolist() == [False, False, True, True, False, False]
    assert chunk_mask(4, 0, -1).all()


def test_forward_shapes():
    m = ConformerCTC(TINY).eval()
    x = torch.randn(2, 203, 80)
    lens = torch.tensor([203, 150])
    logits, out_lens = m(x, lens)
    assert logits.shape[:2] == (2, int(subsampled_lengths(torch.tensor(203))))
    assert out_lens.tolist() == subsampled_lengths(lens).tolist()


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    m = ConformerCTC(TINY).eval()
    x = torch.randn(1, 150, 80)
    with torch.no_grad():
        a, _ = m(x, torch.tensor([150]))
        b, _ = m(torch.cat([x, torch.randn(1, 60, 80)], 1), torch.tensor([150]))
    n = a.shape[1]
    torch.testing.assert_close(a, b[:, :n], atol=1e-4, rtol=1e-4)


def test_streaming_matches_chunked_full_forward():
    torch.manual_seed(0)
    m = ConformerCTC(TINY).eval()
    T, C, L = 403, 4, 2
    x = torch.randn(1, T, 80)
    with torch.no_grad():
        full, _ = m(x, torch.tensor([T]), chunk=C, left_chunks=L)
        win, hop = m.window_frames(C)
        k, v, c = m.init_cache(C * L)
        outs = []
        for n in range((T - win) // hop + 1):
            o, k, v, c = m.forward_chunk(x[:, n * hop : n * hop + win], k, v, c, torch.tensor(n * C))
            outs.append(o)
    s = torch.cat(outs, 1)
    torch.testing.assert_close(s, full[:, : s.shape[1]], atol=1e-4, rtol=1e-4)


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = ConformerCTC(TINY)
    x = torch.randn(2, 120, 80)
    xl = torch.tensor([120, 120])
    y = torch.randint(2, 96, (2, 8))
    yl = torch.tensor([8, 8])
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for _ in range(60):
        logits, ol = m(x, xl)
        loss = torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0, 1), y, ol, yl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.3


def test_metrics():
    assert edit_distance("abc", "abd") == 1
    er = ErrorRate()
    er.update("xin chào việt nam", "xin chao việt nam")
    assert er.wer == 0.25
