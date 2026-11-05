"""Đánh giá STT trên tập test chuẩn, ở cả chế độ offline và streaming.

    python -m vispeech.stt.evaluate --ckpt runs/stt_small/best.pt \
        --fleurs data/test/fleurs_vi_test.parquet --vivos data/test/vivos/test \
        --chunks 0 16 8 --out runs/stt_small/eval.json

- Nhãn và kết quả đều qua cùng `normalize()` (số -> chữ, bỏ dấu câu, một kiểu bỏ dấu).
  Báo cáo WER/CER sau chuẩn hoá (N-WER). Không so trực tiếp được với số của bên khác nếu họ
  chuẩn hoá khác; khi so phải nói rõ.
- `--chunks 0` = offline (full context); 16 = streaming 640 ms; 8 = 320 ms.
- Ghi từng câu (ref/hyp/cer) ra file *.utts.jsonl để soi lỗi.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from ..audio import LogMel, load_audio
from ..metrics import ErrorRate, edit_distance
from ..text.normalize import normalize
from ..text.vocab import Vocab
from .export import load_model


def load_fleurs(path: str):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["id", "audio", "transcription"]).to_pylist()
    for r in t:
        yield f"fleurs:{r['id']}", r["audio"]["bytes"], r["transcription"]


def load_vivos(test_dir: str):
    with open(os.path.join(test_dir, "prompts.txt"), encoding="utf-8") as f:
        for line in f:
            uid, text = line.strip().split(" ", 1)
            spk = uid.split("_")[0]
            yield f"vivos:{uid}", os.path.join(test_dir, "waves", spk, uid + ".wav"), text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fleurs")
    ap.add_argument("--vivos")
    ap.add_argument("--chunks", type=int, nargs="+", default=[0, 16, 8])
    ap.add_argument("--left-chunks", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt).to(device)
    fe, vocab = LogMel(), Vocab()
    sets = {}
    if args.fleurs:
        sets["fleurs"] = load_fleurs(args.fleurs)
    if args.vivos:
        sets["vivos"] = load_vivos(args.vivos)

    report = {}
    for name, it in sets.items():
        utts = []
        for n, (uid, src, text) in enumerate(it):
            if args.limit and n >= args.limit:
                break
            utts.append((uid, fe(load_audio(src, 16000)), normalize(text)))
        report[name] = {"n_utts": len(utts), "hours": sum(len(f) for _, f, _ in utts) / 100 / 3600}
        for chunk in args.chunks:
            er, rows = ErrorRate(), []
            for uid, feats, ref in utts:
                with torch.no_grad():
                    x = torch.from_numpy(feats)[None].to(device)
                    logits, ol = model(x, torch.tensor([len(feats)], device=device), chunk=chunk, left_chunks=args.left_chunks if chunk else -1)
                hyp = vocab.ctc_collapse(logits[0, : ol[0]].argmax(-1).tolist()).strip()
                er.update(ref, hyp)
                c = edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")) / max(1, len(ref.replace(" ", "")))
                rows.append({"id": uid, "ref": ref, "hyp": hyp, "cer": round(c, 4)})
            mode = "offline" if chunk == 0 else f"stream_{chunk * 40}ms"
            report[name][mode] = {"wer": round(er.wer, 4), "cer": round(er.cer, 4)}
            print(f"{name:7s} {mode:14s} WER {100 * er.wer:6.2f}%  CER {100 * er.cer:6.2f}%  ({len(utts)} câu)", flush=True)
            if args.out:
                with open(args.out.replace(".json", f".{name}.{mode}.utts.jsonl"), "w", encoding="utf-8") as f:
                    for r in sorted(rows, key=lambda r: -r["cer"]):
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, **report}, f, indent=1, ensure_ascii=False)
    return report


if __name__ == "__main__":
    main()
