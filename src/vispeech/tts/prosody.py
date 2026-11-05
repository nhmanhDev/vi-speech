"""Đặc trưng ngữ điệu (prosody) mức câu, dùng để điều khiển giọng TTS.

3 đặc trưng mỗi câu:
    pitch   cao độ trung vị (semitone so với 100 Hz)
    range   biên độ cao độ p90 - p10 (semitone; p90-p10 bền với lỗi nhảy quãng tám của bộ đo F0).
            Tiếng Việt có 6 thanh nên biên độ vốn lớn; đặc trưng này vẫn phản ánh câu đọc "hào hứng"
            (rộng) hay "đều đều" (hẹp) khi so trong cùng giọng.
    rate    log(âm tiết / giây)

Không dùng năng lượng: pipeline đã chuẩn hoá mọi câu về -20 LUFS nên năng lượng mức câu gần như hằng
số (đo trên InfoRe: độ lệch chuẩn 0,73 dB), đưa vào chỉ thêm nhiễu.

Sau đó chuẩn hoá z-score theo từng người nói: model học "cao hơn/thấp hơn bình thường của chính giọng đó".
Khi suy luận: `offsets_to_z(stats, pitch_st=+2, rate=1.2)` đổi mong muốn của người dùng sang z-score.

    python -m vispeech.tts.prosody --data data/tts/infore      # ghi prosody.jsonl + prosody_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NAMES = ["pitch", "range", "rate"]
SR = 24000


def features(wav: np.ndarray, sr: int, n_syllables: int) -> list[float] | None:
    import parselmouth

    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.sum() < 20:
        return None
    st = 12 * np.log2(f0[voiced] / 100.0)
    dur = len(wav) / sr
    return [float(np.median(st)), float(np.percentile(st, 90) - np.percentile(st, 10)),
            float(math.log(max(n_syllables, 1) / dur))]


def _work(args):
    path, rows, sr = args
    wav_all = np.memmap(path, dtype=np.int16, mode="r")
    out = []
    for r in rows:
        w = wav_all[r["wav_offset"] : r["wav_offset"] + r["wav_len"]].astype(np.float32) / 32768
        syl = len([t for t in r["text"].split() if t not in {",", ".", "?", "!"}])
        out.append((r["id"], r["speaker"], features(w, sr, syl)))
    return out


def compute_stats(feats_by_spk: dict[str, list[list[float]]]) -> dict:
    stats = {}
    for spk, fs in feats_by_spk.items():
        a = np.array(fs)
        stats[spk] = {"mean": a.mean(0).tolist(), "std": np.maximum(a.std(0), 1e-3).tolist(), "n": len(a)}
    return stats


def to_z(feat: list[float], stats_spk: dict, clip: float = 3.0) -> list[float]:
    z = (np.array(feat) - np.array(stats_spk["mean"])) / np.array(stats_spk["std"])
    return np.clip(z, -clip, clip).tolist()


def offsets_to_z(stats_spk: dict, pitch_st: float = 0.0, range_scale: float = 1.0, rate: float = 1.0) -> list[float]:
    """Mong muốn dễ hiểu -> z-score: cao hơn `pitch_st` semitone, biên độ ×range_scale, nhanh ×rate
    so với trung bình của giọng đó. Ngoài khoảng ±3 độ lệch chuẩn của dữ liệu thì model chưa từng thấy."""
    mean, std = np.array(stats_spk["mean"]), np.array(stats_spk["std"])
    target = mean + np.array([pitch_st, mean[1] * (range_scale - 1), math.log(rate)])
    return ((target - mean) / std).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args(argv)
    for d in args.data:
        with open(os.path.join(d, "meta.json")) as f:
            sr = json.load(f).get("sample_rate", SR)
        with open(os.path.join(d, "index.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        chunks = [rows[i : i + 200] for i in range(0, len(rows), 200)]
        by_spk: dict[str, list] = {}
        res = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for out in pool.map(_work, [(os.path.join(d, "wav.i16"), c, sr) for c in chunks]):
                for uid, spk, feat in out:
                    if feat is not None:
                        res[uid] = (spk, feat)
                        by_spk.setdefault(spk, []).append(feat)
        stats = compute_stats(by_spk)
        with open(os.path.join(d, "prosody.jsonl"), "w") as f:
            for uid, (spk, feat) in res.items():
                f.write(json.dumps({"id": uid, "raw": feat, "z": to_z(feat, stats[spk])}) + "\n")
        with open(os.path.join(d, "prosody_stats.json"), "w") as f:
            json.dump({"names": NAMES, "speakers": stats}, f, indent=1)
        for spk, st in stats.items():
            print(d, spk, {n: f"{m:.2f}±{s:.2f}" for n, m, s in zip(NAMES, st["mean"], st["std"])}, f"({st['n']} câu, thiếu {len(rows) - len(res)})")


if __name__ == "__main__":
    main()
