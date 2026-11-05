"""Tiền xử lý dữ liệu TTS: parquet HF -> audio sạch 24 kHz + mel (khớp vocoder Vocos) + báo cáo QC.

Các bước cho mỗi câu (xem docs/DATA.md để biết lý do):
 1. decode -> mono, giữ sample rate gốc để đo QC
 2. QC thô: clipping, im lặng, độ dài
 3. Silero VAD: cắt im lặng đầu/cuối, chừa 150 ms
 4. chuẩn loudness về -20 LUFS (ITU-R BS.1770), giới hạn đỉnh -1 dBFS
 5. resample 24 kHz (soxr), tính mel 100 bin giống Vocos
 6. chuẩn hoá văn bản (giữ dấu câu để làm điểm ngắt nghỉ), kiểm bảng ký tự
 7. ghi chỉ số để lọc sau: SNR (đo bằng VAD), tốc độ nói (âm tiết/giây)

Bước lọc bằng ASR (CER khi nhận dạng lại) chạy riêng trên GPU: vispeech.tts.asr_filter.

    python -m vispeech.tts.prepare --parquet 'data/infore/*.parquet' --out data/tts/infore --speaker infore
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from ..audio import load_audio, resample
from ..text.normalize import normalize
from ..text.vocab import Vocab

SR = 24000
TARGET_LUFS = -20.0
PEAK_DB = -1.0
_W: dict = {}


class VocosMel:
    """Mel giống hệt `vocos.feature_extractors.MelSpectrogramFeatures` (charactr/vocos-mel-24khz)."""

    def __init__(self):
        import torchaudio

        self.t = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=256, n_mels=100, center=True, power=1
        )

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            mel = self.t(torch.from_numpy(wav)[None])[0]
        return torch.log(torch.clamp(mel, min=1e-7)).T.numpy()  # (frames, 100)


def wada_snr(wav: np.ndarray) -> float:
    """Ước lượng SNR không cần tham chiếu (Kim & Stern 2008), xấp xỉ bằng bảng tra."""
    db_vals = np.arange(-20, 101)
    g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627,
        0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886,
        0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088,
        0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148,
        0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349, 1.01047155, 1.0362095,
        1.06136425, 1.08579312, 1.1094819, 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898,
        1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727, 1.37345513,
        1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568,
        1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097, 1.528578,
        1.53389835, 1.5391211, 1.5439065, 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671,
        1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969, 1.59693155,
        1.599446, 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074,
        1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027,
        1.62842767, 1.62945532, 1.6303307, 1.63128026, 1.63204102])
    w = wav / (np.abs(wav).max() + 1e-9)
    a = np.abs(w) + 1e-10
    v = np.log(a.mean()) - np.log(a).mean()
    snr = db_vals[np.argmin(np.abs(g_vals - v))] if v > g_vals[0] else -20
    return float(snr)


def vad_snr(w16: np.ndarray, ts: list[dict]) -> float | None:
    """SNR = công suất đoạn có tiếng nói / công suất đoạn im lặng (theo VAD), dB.

    Đáng tin hơn WADA-SNR với dữ liệu đọc trong phòng thu (đã kiểm trên InfoRe:
    WADA loại nhầm nhiều câu sạch). Trả None nếu không đủ 0,1 s im lặng để đo nền nhiễu.
    """
    mask = np.zeros(len(w16), dtype=bool)
    for t in ts:
        mask[t["start"] : t["end"]] = True
    # bỏ 50 ms quanh biên để không tính đuôi âm vào nhiễu
    edge = np.convolve(mask.astype(np.float32), np.ones(1601, dtype=np.float32), "same") > 0.5
    noise = w16[~edge]
    if len(noise) < 1600:
        return None
    return float(10 * np.log10((w16[mask] ** 2).mean() / ((noise**2).mean() + 1e-12)))


def _init():
    import pyloudnorm
    import silero_vad

    _W["vad"] = silero_vad.load_silero_vad(onnx=True)
    _W["get_ts"] = silero_vad.get_speech_timestamps
    _W["meter"] = pyloudnorm.Meter(SR)
    _W["mel"] = VocosMel()
    _W["vocab"] = Vocab(with_punct=True)


def _process_one(audio_bytes: bytes, text: str, args) -> tuple[str, dict]:
    import torch

    raw, sr = __import__("soundfile").read(__import__("io").BytesIO(audio_bytes), dtype="float32", always_2d=True)
    raw = raw.mean(1)
    info = {"orig_sr": int(sr)}
    if len(raw) < sr * 0.3 or np.abs(raw).max() < 1e-4:
        return "silent", info
    info["clip_ratio"] = float((np.abs(raw) >= 0.99).mean())
    if info["clip_ratio"] > args.max_clip_ratio:
        return "clipping", info

    # VAD chạy ở 16 kHz
    w16 = resample(raw, sr, 16000)
    ts = _W["get_ts"](torch.from_numpy(w16), _W["vad"], sampling_rate=16000, min_silence_duration_ms=300)
    if not ts:
        return "no_speech", info
    pad = int(0.15 * sr)
    start = max(0, ts[0]["start"] * sr // 16000 - pad)
    end = min(len(raw), ts[-1]["end"] * sr // 16000 + pad)
    wav = resample(raw[start:end], sr, SR)

    dur = len(wav) / SR
    info["dur"] = round(dur, 3)
    if not (args.min_dur <= dur <= args.max_dur):
        return "duration", info

    # Khoảng lặng dài bất thường giữa câu (đọc vấp, ngừng lâu) -> cắt câu sai
    gaps = [(b["start"] - a["end"]) / 16000 for a, b in zip(ts, ts[1:])]
    info["max_gap"] = round(max(gaps), 3) if gaps else 0.0
    if info["max_gap"] > args.max_gap:
        return "long_pause", info

    loud = _W["meter"].integrated_loudness(wav)
    if not np.isfinite(loud):
        return "silent", info
    info["lufs_in"] = round(float(loud), 2)
    wav = wav * (10 ** ((TARGET_LUFS - loud) / 20))
    peak = np.abs(wav).max()
    lim = 10 ** (PEAK_DB / 20)
    if peak > lim:
        wav = wav * (lim / peak)
    snr = vad_snr(w16, ts)
    info["snr_method"] = "vad" if snr is not None else "wada"
    info["snr"] = round(snr if snr is not None else wada_snr(wav), 1)
    if info["snr"] < args.min_snr:
        return "low_snr", info

    norm = normalize(text or "", keep_punct=True)
    ids = _W["vocab"].encode(norm)
    if not norm or _W["vocab"].unk_id in ids:
        return "text", info
    syl = len([t for t in norm.split() if t not in {",", ".", "?", "!"}])
    info["sps"] = round(syl / dur, 2)  # âm tiết / giây
    info["text"] = norm
    info["_wav"] = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    info["_mel"] = _W["mel"](wav.astype(np.float32)).astype(np.float16)
    return "ok", info


def _process_rg(task):
    path, rg, args = task
    t = pq.ParquetFile(path).read_row_group(rg, columns=["audio", "transcription"])
    out = []
    for i, (a, txt) in enumerate(zip(t.column("audio").to_pylist(), t.column("transcription").to_pylist())):
        try:
            status, info = _process_one(a["bytes"], txt, args)
        except Exception as e:  # file hỏng
            status, info = "decode", {"err": str(e)[:100]}
        info["id"] = f"{os.path.basename(path)}:{rg}:{i}"
        out.append((status, info))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", default="spk0")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-gap", type=float, default=1.5, help="giây im lặng tối đa giữa câu")
    ap.add_argument("--max-clip-ratio", type=float, default=1e-3)
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--limit-files", type=int, default=0)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(args.parquet))[: args.limit_files or None]
    if not files:
        raise SystemExit(f"Không thấy file {args.parquet}")
    tasks = [(f, rg, args) for f in files for rg in range(pq.ParquetFile(f).num_row_groups)]
    os.makedirs(args.out, exist_ok=True)

    counts: dict[str, int] = {}
    stats = {k: [] for k in ("dur", "snr", "sps", "lufs_in", "max_gap", "orig_sr")}
    wav_off = mel_off = 0
    with open(os.path.join(args.out, "wav.i16"), "wb") as fw, open(os.path.join(args.out, "mel.f16"), "wb") as fm, \
            open(os.path.join(args.out, "index.jsonl"), "w", encoding="utf-8") as fi, \
            open(os.path.join(args.out, "rejected.jsonl"), "w", encoding="utf-8") as fr, \
            ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for n, results in enumerate(pool.map(_process_rg, tasks), 1):
            for status, info in results:
                counts[status] = counts.get(status, 0) + 1
                if status != "ok":
                    fr.write(json.dumps({"status": status, **info}, ensure_ascii=False) + "\n")
                    continue
                wav, mel = info.pop("_wav"), info.pop("_mel")
                fw.write(wav.tobytes())
                fm.write(mel.tobytes())
                rec = {"id": info["id"], "speaker": args.speaker, "text": info["text"], "wav_offset": wav_off,
                       "wav_len": len(wav), "mel_offset": mel_off, "mel_frames": len(mel),
                       **{k: info[k] for k in stats if k in info}}
                fi.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for k in stats:
                    stats[k].append(info[k])
                wav_off += len(wav)
                mel_off += len(mel)
            print(f"[{n}/{len(tasks)}] {counts} hours={wav_off / SR / 3600:.2f}", flush=True)

    def summary(v):
        v = np.asarray(v, dtype=float)
        return {p: round(float(np.percentile(v, q)), 2) for p, q in [("p1", 1), ("p50", 50), ("p99", 99)]} if len(v) else {}

    report = {"counts": counts, "hours": wav_off / SR / 3600, "sample_rate": SR, "mel": "vocos-mel-24khz (100 bin, hop 256)",
              "wav_samples": wav_off, "mel_frames": mel_off, "stats": {k: summary(v) for k, v in stats.items()},
              "params": {k: v for k, v in vars(args).items() if k not in ("parquet", "out")}, "sources": files}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("counts", "hours", "stats")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
