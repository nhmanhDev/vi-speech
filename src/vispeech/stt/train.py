"""Train streaming Conformer-CTC.

    python -m vispeech.stt.train --config configs/stt_small.yaml \
        --data data/prep/vlsp --out runs/stt_small [--resume runs/ /kaggle/input/...] [--max-hours 11.3]

Thiết kế để chạy được trên Kaggle (giới hạn 12 giờ / phiên):
- Lưu `last.pt` định kỳ và trước khi hết giờ; phiên sau truyền --resume để chạy tiếp.
- `best.pt` theo WER dev ở chế độ streaming (chế độ sẽ deploy).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import DynamicBatchSampler, MelDataset, collate
from ..ema import EMA
from ..metrics import ErrorRate
from ..text.vocab import Vocab
from .model import ConformerConfig, ConformerCTC, count_params

DEFAULTS = {
    "model": {},
    "train": {
        "lr": 1.5e-3,
        "weight_decay": 1e-3,
        "warmup_steps": 5000,
        "total_steps": 150000,
        "max_batch_frames": 40000,  # tổng số mel frame / batch (10 ms/frame) -> ~6,7 phút audio
        "grad_accum": 1,
        "grad_clip": 5.0,
        "max_utt_frames": 2000,
        "dev_percent": 1.0,
        "inter_ctc_weight": 0.3,
        "ema_decay": 0.9999,  # 0 = tắt EMA trọng số  # chỉ dùng khi model.inter_ctc_layers khác rỗng
        "num_workers": 2,
        "log_every": 50,
        "eval_every": 2000,
        "save_every_min": 20,
        "eval_utts": 300,
        "seed": 42,
    },
    "streaming": {
        "full_context_prob": 0.4,  # tỷ lệ batch train full context
        "max_chunk": 25,  # chunk ngẫu nhiên 1..25 frame (40 ms .. 1 s)
        "max_left_chunks": 8,
        "min_left_chunks": 1,  # left=0 + chunk nhỏ: attention gần như vô dụng, làm khó tối ưu
        "start_step": 0,  # trước bước này chỉ train full context (curriculum)
        "eval_chunk": 16,  # 640 ms
        "eval_left_chunks": 4,
    },
    # ESPnet librispeech_100 / WeNet: 2 mặt nạ tần, 2 mặt nạ thời gian. Bản "adaptive" 10 mặt nạ
    # (Park et al. 2020) dành cho dữ liệu hàng nghìn giờ; với ~100 giờ nó làm model khó thoát plateau.
    "specaug": {"freq_masks": 2, "freq_width": 27, "time_masks": 2, "time_ratio": 0.05, "time_width_max": 40},
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
    return cfg


def spec_augment(x: torch.Tensor, lengths: torch.Tensor, cfg: dict) -> torch.Tensor:
    """SpecAugment: che ngẫu nhiên dải tần và đoạn thời gian (trên GPU, theo batch)."""
    b, t, f = x.shape
    x = x.clone()
    fill = x.mean()
    for i in range(b):
        L = int(lengths[i])
        for _ in range(cfg["freq_masks"]):
            w = random.randint(0, cfg["freq_width"])
            s = random.randint(0, max(0, f - w))
            x[i, :, s : s + w] = fill
        max_w = max(1, min(int(L * cfg["time_ratio"]), cfg.get("time_width_max", 10**9)))
        for _ in range(cfg["time_masks"]):
            w = random.randint(0, max_w)
            s = random.randint(0, max(0, L - w))
            x[i, s : s + w, :] = fill
    return x


def lr_at(step: int, cfg: dict) -> float:
    w, total, peak = cfg["warmup_steps"], cfg["total_steps"], cfg["lr"]
    if step < w:
        return peak * step / w
    p = min(1.0, (step - w) / max(1, total - w))
    return peak * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def find_checkpoint(paths: list[str]) -> str | None:
    found = []
    for p in paths:
        if os.path.isfile(p):
            found.append(p)
        else:
            found += glob.glob(os.path.join(p, "**", "last.pt"), recursive=True)
    return max(found, key=os.path.getmtime) if found else None


def save(path: str, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, dev, cfg, device, vocab) -> dict:
    model.eval()
    s = cfg["streaming"]
    n = min(len(dev), cfg["train"]["eval_utts"])
    results = {}
    for name, chunk, left in [("full", 0, -1), ("stream", s["eval_chunk"], s["eval_left_chunks"])]:
        er = ErrorRate()
        loss_sum, count = 0.0, 0
        for i in range(0, n, 16):
            x, xl, y, yl, texts = collate([dev[j] for j in range(i, min(n, i + 16))])
            x, xl, y, yl = x.to(device), xl.to(device), y.to(device), yl.to(device)
            logits, ol = model(x, xl, chunk=chunk, left_chunks=left)
            lp = logits.float().log_softmax(-1)
            loss_sum += F.ctc_loss(lp.transpose(0, 1), y, ol, yl, blank=0, reduction="sum", zero_infinity=True).item()
            count += int(yl.sum())
            for b, ref in enumerate(texts):
                er.update(ref, vocab.ctc_collapse(lp[b, : ol[b]].argmax(-1).tolist()))
        results[name] = {"loss_per_char": loss_sum / max(count, 1), "wer": er.wer, "cer": er.cer}
    model.train()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data", nargs="+", required=True, help="thư mục output của vispeech.data.prepare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", nargs="*", default=[], help="file/thư mục để tìm last.pt")
    ap.add_argument("--max-hours", type=float, default=0, help="dừng an toàn sau N giờ (Kaggle: 11.3)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    t_start = time.time()
    cfg = load_config(args.config)
    tc = cfg["train"]
    # Multi-GPU: chạy bằng `torchrun --nproc_per_node=N`, torchrun đặt RANK/WORLD_SIZE/LOCAL_RANK.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp = world > 1
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            dist.init_process_group("nccl")
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:  # gloo: để test DDP trên CPU
            dist.init_process_group("gloo")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    is_main = rank == 0
    if not is_main:
        import builtins

        builtins.print = lambda *a, **k: None  # chỉ rank 0 in log
    random.seed(tc["seed"] + rank)  # SpecAugment / chunk khác nhau giữa các GPU
    torch.manual_seed(tc["seed"])  # nhưng khởi tạo model giống nhau
    os.makedirs(args.out, exist_ok=True)
    vocab = Vocab()

    train_set = MelDataset(args.data, "train", tc["dev_percent"], tc["max_utt_frames"])
    dev_set = MelDataset(args.data, "dev", tc["dev_percent"], tc["max_utt_frames"])
    print(f"train={len(train_set)} dev={len(dev_set)} utts")

    mcfg = ConformerConfig(**{**cfg["model"], "vocab_size": len(vocab)})
    model = ConformerCTC(mcfg).to(device)
    # CMVN gộp từ mọi thư mục dữ liệu, trọng số theo số frame (mean/var của hỗn hợp)
    import numpy as np

    metas = []
    for d in args.data:
        with open(os.path.join(d, "meta.json")) as f:
            metas.append(json.load(f))
    w = np.array([m["total_frames"] for m in metas], dtype=np.float64)
    mu = np.array([m["mean"] for m in metas])
    var = np.array([m["std"] for m in metas]) ** 2
    mean = (w[:, None] * mu).sum(0) / w.sum()
    std = np.sqrt((w[:, None] * (var + mu**2)).sum(0) / w.sum() - mean**2)
    model.set_cmvn(mean, std)
    print(f"params={count_params(model) / 1e6:.1f}M config={mcfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.98), weight_decay=tc["weight_decay"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, epoch, best_wer = 0, 0, float("inf")
    ema = EMA(model, tc["ema_decay"]) if tc.get("ema_decay") and is_main else None

    ckpt_path = find_checkpoint(args.resume + [args.out])
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch, best_wer = ck["step"], ck["epoch"], ck.get("best_wer", best_wer)
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
        print(f"resumed from {ckpt_path} at step {step}, epoch {epoch}")

    train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if ddp else model

    def checkpoint(name="last.pt"):
        if not is_main:
            return
        extra = {"ema": ema.state_dict(), "model_ema": ema.weights(model)} if ema is not None else {}
        save(os.path.join(args.out, name), model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
             step=step, epoch=epoch, best_wer=best_wer, model_config=vars(mcfg), config=cfg, **extra)

    sampler = DynamicBatchSampler([train_set.frames(i) for i in range(len(train_set))], tc["max_batch_frames"], seed=tc["seed"],
                                  rank=rank, world_size=world)
    loader_kw = dict(batch_sampler=sampler, collate_fn=collate, num_workers=tc["num_workers"], pin_memory=use_amp)
    if tc["num_workers"] > 0:
        loader_kw["persistent_workers"] = False
    s = cfg["streaming"]
    last_save = time.time()
    log_path = os.path.join(args.out, "log.jsonl")
    model.train()
    done = False
    while not done:
        sampler.set_epoch(epoch)
        t_log, loss_acc, n_acc = time.time(), 0.0, 0
        data_wait, t_fetch = 0.0, time.time()
        blank_acc, clip_acc = 0.0, 0  # chẩn đoán: tỉ lệ blank trong argmax, số bước bị clip gradient
        for bi, (x, xl, y, yl, _) in enumerate(DataLoader(train_set, **loader_kw)):
            data_wait += time.time() - t_fetch  # thời gian GPU ngồi chờ batch: lớn nghĩa là nghẽn I/O / CPU
            x, xl, y, yl = x.to(device, non_blocking=True), xl.to(device), y.to(device), yl.to(device)
            x = spec_augment(x, xl, cfg["specaug"])
            if step < s["start_step"] or random.random() < s["full_context_prob"]:
                chunk, left = 0, -1
            else:
                chunk = random.randint(1, s["max_chunk"])
                left = random.choice([-1, random.randint(s["min_left_chunks"], s["max_left_chunks"])])
            accumulating = (bi + 1) % tc["grad_accum"] != 0
            sync_ctx = train_model.no_sync() if ddp and accumulating else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits, ol, inter = train_model(x, xl, chunk=chunk, left_chunks=left, return_inter=True)

                def ctc(lg):
                    lp_ = lg.float().log_softmax(-1)
                    return F.ctc_loss(lp_.transpose(0, 1), y, ol, yl, blank=0, reduction="sum", zero_infinity=True) / x.shape[0]

                loss = ctc(logits)
                with torch.no_grad():  # model kẹt ở plateau CTC thì argmax gần như toàn blank (~1.0)
                    valid = torch.arange(logits.shape[1], device=device)[None] < ol[:, None]
                    blank_acc += ((logits.argmax(-1) == 0) & valid).sum().item() / max(1, valid.sum().item())
                if inter:  # InterCTC (Lee & Watanabe 2021): giám sát thêm lớp giữa, model sâu hội tụ nhanh hơn
                    w_i = tc["inter_ctc_weight"]
                    loss = (1 - w_i) * loss + w_i * sum(ctc(lg) for lg in inter) / len(inter)
                scaler.scale(loss / tc["grad_accum"]).backward()
            loss_acc += loss.item()
            n_acc += 1
            if accumulating:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc)
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            clip_acc += int(float(gnorm) > tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % tc["log_every"] == 0 and is_main:
                rec = {"step": step, "epoch": epoch, "loss": loss_acc / n_acc, "lr": lr_at(step, tc), "gnorm": float(gnorm),
                       "sec_per_step": (time.time() - t_log) / tc["log_every"], "data_wait_per_step": data_wait / tc["log_every"],
                       "frames_per_batch": int(xl.sum()),
                       "blank_ratio": round(blank_acc / max(1, n_acc), 3), "clipped_frac": round(clip_acc / tc["log_every"], 2), "elapsed_h": (time.time() - t_start) / 3600}
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                t_log, loss_acc, n_acc, data_wait, blank_acc, clip_acc = time.time(), 0.0, 0, 0.0, 0.0, 0

            if step % tc["eval_every"] == 0 and len(dev_set) and is_main:
                res = evaluate(model, dev_set, cfg, device, vocab)
                if ema is not None:  # đánh giá cả bản EMA; chọn best theo bản tốt hơn ở chế độ streaming
                    with ema.applied(model):
                        res["ema"] = evaluate(model, dev_set, cfg, device, vocab)
                print(json.dumps({"step": step, "eval": res}), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "eval": res}) + "\n")
                cur = min(res["stream"]["wer"], res.get("ema", {}).get("stream", {}).get("wer", float("inf")))
                if cur < best_wer:
                    best_wer = cur
                    checkpoint("best.pt")

            out_of_time = args.max_hours and (time.time() - t_start) / 3600 > args.max_hours
            out_of_steps = (args.max_steps and step >= args.max_steps) or step >= tc["total_steps"]
            if ddp:  # mọi GPU phải dừng cùng một bước, theo quyết định của rank 0
                flag = torch.tensor([int(bool(out_of_time)), int(bool(out_of_steps))], device=device)
                dist.broadcast(flag, 0)
                out_of_time, out_of_steps = bool(flag[0]), bool(flag[1])
            if time.time() - last_save > tc["save_every_min"] * 60 or out_of_time or out_of_steps:
                checkpoint()
                last_save = time.time()
            t_fetch = time.time()
            if out_of_time or out_of_steps:
                print(f"stopping at step {step} ({'time' if out_of_time else 'steps'})")
                done = True
                break
        else:
            epoch += 1
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if len(dev_set) and is_main:
        res = evaluate(model, dev_set, cfg, device, vocab)
        print(json.dumps({"step": step, "final_eval": res}), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "final_eval": res}) + "\n")


if __name__ == "__main__":
    main()
