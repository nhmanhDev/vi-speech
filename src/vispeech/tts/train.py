"""Train Matcha-TTS trên dữ liệu từ vispeech.tts.prepare (+ asr_filter).

    python -m vispeech.tts.train --data data/tts/infore --out runs/tts_matcha [--max-cer 0.05] [--max-hours 11.3]

Hỗ trợ torchrun nhiều GPU, resume từ last.pt, dừng an toàn theo thời gian (Kaggle).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import DynamicBatchSampler
from ..stt.train import find_checkpoint, lr_at, save
from ..text.vocab import Vocab
from .matcha import MatchaConfig, MatchaTTS, intersperse


class TTSDataset(Dataset):
    def __init__(self, dirs, split="train", max_cer=0.05, dev_percent=1.0, max_frames=1400):
        self.vocab = Vocab(with_punct=True)
        self.blank = len(self.vocab)  # token chèn giữa ký tự
        self.mels, self.items, self.speakers = [], [], {}
        self.stats = {"total": 0, "cer": 0, "length": 0, "no_asr": 0}
        for di, d in enumerate([dirs] if isinstance(dirs, str) else dirs):
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self.mels.append(np.memmap(os.path.join(d, "mel.f16"), dtype=np.float16, mode="r", shape=(meta["mel_frames"], 100)))
            asr = os.path.join(d, "index.asr.jsonl")
            path = asr if os.path.exists(asr) else os.path.join(d, "index.jsonl")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.stats["total"] += 1
                    if "cer" not in r:
                        self.stats["no_asr"] += 1
                    elif r["cer"] > max_cer:
                        self.stats["cer"] += 1
                        continue
                    if r["mel_frames"] > max_frames:
                        self.stats["length"] += 1
                        continue
                    if (zlib.crc32(r["id"].encode()) % 10000 < dev_percent * 100) != (split == "dev"):
                        continue
                    spk = self.speakers.setdefault(r["speaker"], len(self.speakers))
                    self.items.append((di, r["mel_offset"], r["mel_frames"], r["text"], spk))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        di, off, n, text, spk = self.items[i]
        mel = torch.from_numpy(np.asarray(self.mels[di][off : off + n], dtype=np.float32)).T  # (100, T)
        x = torch.tensor(intersperse(self.vocab.encode(text), self.blank))
        return x, mel, spk, text


def collate(batch):
    xs, mels, spks, texts = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xs])
    y_lens = torch.tensor([m.shape[1] for m in mels])
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    y = torch.zeros(len(mels), 100, int(y_lens.max()))
    for i, m in enumerate(mels):
        y[i, :, : m.shape[1]] = m
    return x, x_lens, y, y_lens, torch.tensor(spks), list(texts)


DEFAULTS = {"model": {}, "train": {"lr": 2e-4, "warmup_steps": 2000, "total_steps": 300000, "weight_decay": 0.0,
            "max_batch_frames": 24000, "grad_clip": 1.0, "num_workers": 2, "log_every": 50, "eval_every": 2000,
            "save_every_min": 20, "seed": 1234, "max_cer": 0.05}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", nargs="*", default=[])
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-cer", type=float)
    args = ap.parse_args(argv)

    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                cfg.setdefault(k, {}).update(v)
    tc = cfg["train"]
    if args.max_cer is not None:
        tc["max_cer"] = args.max_cer

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    ddp = world > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0
    log = print if is_main else (lambda *a, **k: None)
    random.seed(tc["seed"] + rank)
    torch.manual_seed(tc["seed"])
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    train_set = TTSDataset(args.data, "train", tc["max_cer"])
    dev_set = TTSDataset(args.data, "dev", tc["max_cer"])
    log(f"train={len(train_set)} dev={len(dev_set)} filter={train_set.stats} speakers={train_set.speakers}")
    if train_set.stats["no_asr"]:
        log("CẢNH BÁO: dữ liệu chưa qua asr_filter, không lọc được câu nhãn sai")

    mcfg = MatchaConfig(**{**cfg["model"], "vocab_size": train_set.blank + 1, "n_speakers": max(1, len(train_set.speakers))})
    model = MatchaTTS(mcfg).to(device)
    # Chuẩn hoá mel toàn cục (Matcha dùng mean/std của dataset)
    sample = np.concatenate([np.asarray(train_set.mels[di][off : off + n], dtype=np.float32)
                             for di, off, n, _, _ in random.Random(0).sample(train_set.items, min(500, len(train_set)))])
    model.set_mel_stats(float(sample.mean()), float(sample.std()))
    log(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M mel mean={sample.mean():.3f} std={sample.std():.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, epoch = 0, 0
    ck_path = find_checkpoint(args.resume + [args.out])
    if ck_path:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch = ck["step"], ck["epoch"]
        log(f"resumed {ck_path} step {step}")
    train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if ddp else model

    def checkpoint(name="last.pt"):
        if is_main:
            save(os.path.join(args.out, name), model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                 step=step, epoch=epoch, model_config=vars(mcfg), config=cfg, vocab_with_punct=True)

    lengths = [it[2] for it in train_set.items]
    sampler = DynamicBatchSampler(lengths, tc["max_batch_frames"], seed=tc["seed"], rank=rank, world_size=world)
    log_path = os.path.join(args.out, "log.jsonl")
    last_save, done = time.time(), False
    while not done:
        sampler.set_epoch(epoch)
        t_log, acc = time.time(), {"dur": 0.0, "prior": 0.0, "fm": 0.0}
        n_acc = 0
        loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate, num_workers=tc["num_workers"], pin_memory=use_amp)
        for x, xl, y, yl, spk, _ in loader:
            x, xl, y, yl, spk = (t.to(device, non_blocking=True) for t in (x, xl, y, yl, spk))
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                losses = train_model(x, xl, y, yl, spk if mcfg.n_speakers > 1 else None)
            loss = losses["dur"].float() + losses["prior"].float() + losses["fm"].float()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            for k in acc:
                acc[k] += losses[k].item()
            n_acc += 1

            if step % tc["log_every"] == 0 and is_main:
                rec = {"step": step, "epoch": epoch, **{k: v / n_acc for k, v in acc.items()}, "gnorm": float(gnorm),
                       "lr": lr_at(step, tc), "sec_per_step": (time.time() - t_log) / n_acc, "elapsed_h": (time.time() - t_start) / 3600}
                log(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                t_log, acc, n_acc = time.time(), {k: 0.0 for k in acc}, 0

            if step % tc["eval_every"] == 0 and is_main and len(dev_set):
                model.eval()
                with torch.no_grad():
                    xs, xls, ys, yls, spks, _ = collate([dev_set[i] for i in range(min(16, len(dev_set)))])
                    l = model(xs.to(device), xls.to(device), ys.to(device), yls.to(device),
                              spks.to(device) if mcfg.n_speakers > 1 else None)
                rec = {"step": step, "dev": {k: float(v) for k, v in l.items() if k != "attn"}}
                log(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                model.train()

            out_of_time = bool(args.max_hours and (time.time() - t_start) / 3600 > args.max_hours)
            out_of_steps = bool((args.max_steps and step >= args.max_steps) or step >= tc["total_steps"])
            if ddp:
                flag = torch.tensor([int(out_of_time), int(out_of_steps)], device=device)
                dist.broadcast(flag, 0)
                out_of_time, out_of_steps = bool(flag[0]), bool(flag[1])
            if time.time() - last_save > tc["save_every_min"] * 60 or out_of_time or out_of_steps:
                checkpoint()
                last_save = time.time()
            if out_of_time or out_of_steps:
                done = True
                break
        else:
            epoch += 1
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
