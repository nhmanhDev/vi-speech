"""Điều khiển Kaggle từ máy dev / cloud: đẩy code, chạy kernel tiền xử lý (CPU) và train (GPU).

Cần biến môi trường KAGGLE_API_TOKEN (hoặc ~/.kaggle/access_token). Tài khoản Kaggle phải
xác minh số điện thoại thì mới bật được GPU và internet trong kernel.

    python scripts/kaggle_job.py push-code
    python scripts/kaggle_job.py prep  --hf doof-ferb/vlsp2020_vinai_100h --name vlsp
    python scripts/kaggle_job.py train --prep vlsp --config configs/stt_small.yaml --run stt-small [--resume]
    python scripts/kaggle_job.py status  <kernel-slug>
    python scripts/kaggle_job.py output  <kernel-slug> [--dest runs/kaggle]

Luồng nhiều phiên: train phiên 1 -> hết ~11,3 giờ tự lưu last.pt vào output -> chạy lại
`train ... --resume`: kernel nhận output phiên trước làm input và train tiếp.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_SLUG = "vi-speech-code"


def kaggle(*args: str, check=True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def username() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    out = kaggle("kernels", "list", "--mine", "--page-size", "1", "--csv")
    lines = out.splitlines()
    if len(lines) > 1:
        return lines[1].split(",")[0].split("/")[0]
    raise SystemExit("Không xác định được username; đặt KAGGLE_USERNAME")


# Đoạn code chung đầu mỗi kernel: tìm code đã upload trong /kaggle/input và import.
BOOTSTRAP = r'''
import glob, os, subprocess, sys
src = [p for p in glob.glob("/kaggle/input/**/src/vispeech", recursive=True)]
assert src, "Không thấy dataset code vi-speech-code trong /kaggle/input"
ROOT = os.path.dirname(os.path.dirname(src[0]))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["PYTHONPATH"] = os.path.join(ROOT, "src")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soxr", "pyyaml", "soundfile", "huggingface_hub", "onnxruntime"], check=False)
print("code:", ROOT, flush=True)
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
'''

PREP = BOOTSTRAP + r'''
from huggingface_hub import snapshot_download
d = snapshot_download("{hf}", repo_type="dataset", allow_patterns=["data/*.parquet"], local_dir="/tmp/hf/{name}")
sh(f"python -m vispeech.data.prepare --parquet '{{d}}/data/*.parquet' --out /kaggle/working/prep/{name} --workers 4 {extra}")
sh("rm -rf /tmp/hf")
'''

TTS_PREP = BOOTSTRAP + r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyloudnorm", "silero-vad", "transformers"], check=False)
from huggingface_hub import snapshot_download
d = snapshot_download("{hf}", repo_type="dataset", allow_patterns=["data/*.parquet"], local_dir="/tmp/hf/{name}")
out = "/kaggle/working/tts/{name}"
sh(f"python -m vispeech.tts.prepare --parquet '{{d}}/data/*.parquet' --out {{out}} --speaker {name} --workers 4 {extra}")
sh("rm -rf /tmp/hf")
sh(f"python -m vispeech.tts.asr_filter --data {{out}} --model {asr} --batch 24")
'''

TRAIN = BOOTSTRAP + r'''
sh("nvidia-smi")
# /kaggle/input là ổ mạng: đọc ngẫu nhiên memmap rất chậm (đo được ~6 s/bước).
# Copy tuần tự sang ổ local /tmp trước (vài phút) rồi mới train.
srcs = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/prep/*/meta.json", recursive=True))]
dsts = []
for s_ in srcs:
    d_ = "/tmp/prep/" + os.path.basename(s_)
    sh(f"mkdir -p /tmp/prep && cp -r {{s_}} {{d_}}")
    dsts.append(d_)
data = " ".join(dsts)
print("data:", data)
resume = " ".join(glob.glob("/kaggle/input/**/runs/{run}", recursive=True))
sh(f"cp -r {{resume}} /kaggle/working/runs_prev || true") if resume else None
import torch
ngpu = max(1, torch.cuda.device_count())
sh(f"torchrun --standalone --nproc_per_node={{ngpu}} -m vispeech.stt.train --config {{ROOT}}/{config} --data {{data}} --out /kaggle/working/runs/{run} "
   f"--resume /kaggle/working/runs_prev --max-hours {max_hours}")
sh("rm -rf /kaggle/working/runs_prev")
'''


TTS_TRAIN = BOOTSTRAP + r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vocos", "praat-parselmouth"], check=False)  # mẫu giọng + đo ngữ điệu
srcs = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/tts/*/meta.json", recursive=True))]
dsts = []
for s_ in srcs:  # copy sang ổ local (ổ /kaggle/input đọc ngẫu nhiên rất chậm)
    d_ = "/tmp/tts/" + os.path.basename(s_)
    sh(f"mkdir -p /tmp/tts && cp -r {{s_}} {{d_}}")
    dsts.append(d_)
if "prosody" in "{config}":  # đo ngữ điệu từng câu (CPU, vài phút) trước khi train
    sh(f"python -m vispeech.tts.prosody --data {{' '.join(dsts)}} --workers 4")
resume = " ".join(glob.glob("/kaggle/input/**/runs/{run}", recursive=True))
sh(f"cp -r {{resume}} /kaggle/working/runs_prev || true") if resume else None
import torch
ngpu = max(1, torch.cuda.device_count())
sh(f"torchrun --standalone --nproc_per_node={{ngpu}} -m vispeech.tts.train --config {{ROOT}}/{config} --data {{' '.join(dsts)}} "
   f"--out /kaggle/working/runs/{run} --resume /kaggle/working/runs_prev --max-hours {max_hours}")
sh("rm -rf /kaggle/working/runs_prev")
'''


STT_EXP = BOOTSTRAP + r'''
# Thí nghiệm song song: mỗi GPU chạy 1 cấu hình, cùng dữ liệu, cùng số giờ -> so sánh công bằng.
srcs = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/prep/*/meta.json", recursive=True))]
dsts = []
for s_ in srcs:
    d_ = "/tmp/prep/" + os.path.basename(s_)
    sh(f"mkdir -p /tmp/prep && cp -r {{s_}} {{d_}}")
    dsts.append(d_)
procs = []
for gpu, cfg in enumerate({configs}):
    name = os.path.splitext(os.path.basename(cfg))[0]
    cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} python -m vispeech.stt.train --config {{ROOT}}/{{cfg}} --data {{' '.join(dsts)}} "
           f"--out /kaggle/working/runs/{{name}} --max-hours {max_hours} > /kaggle/working/{{name}}.log 2>&1")
    print("+", cmd, flush=True)
    procs.append(subprocess.Popen(cmd, shell=True))
import time
while any(p.poll() is None for p in procs):
    time.sleep(300)
    for cfg in {configs}:
        name = os.path.splitext(os.path.basename(cfg))[0]
        subprocess.run(f"grep -h eval /kaggle/working/{{name}}.log | tail -1 | cut -c1-260", shell=True)
print("exit codes", [p.returncode for p in procs])
'''


def push_kernel(slug: str, title: str, code: str, gpu: bool, datasets: list[str], kernels: list[str]):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "main.py"), "w") as f:
            f.write(code)
        meta = {
            "id": slug, "title": title, "code_file": "main.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": datasets, "kernel_sources": kernels, "competition_sources": [],
        }
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(kaggle("kernels", "push", "-p", d))
        print(f"https://www.kaggle.com/code/{slug}")


def cmd_push_code(args):
    user = username()
    with tempfile.TemporaryDirectory() as d:
        for item in ("src", "configs", "pyproject.toml", "README.md"):
            p = os.path.join(ROOT, item)
            (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join(d, item))
        for pyc in subprocess.run(["find", d, "-name", "__pycache__"], capture_output=True, text=True).stdout.split():
            shutil.rmtree(pyc, ignore_errors=True)
        with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
            json.dump({"id": f"{user}/{CODE_SLUG}", "title": CODE_SLUG, "licenses": [{"name": "other"}]}, f)
        mine = kaggle("datasets", "list", "--mine", "-s", CODE_SLUG, "--csv", check=False)
        exists = any(line.split(",")[0] == f"{user}/{CODE_SLUG}" for line in mine.splitlines())
        if exists:
            print(kaggle("datasets", "version", "-p", d, "-m", args.message, "-r", "tar"))
        else:
            print(kaggle("datasets", "create", "-p", d, "-r", "tar"))


def cmd_prep(args):
    user = username()
    extra = f"--limit-files {args.limit_files}" if args.limit_files else ""
    code = PREP.format(hf=args.hf, name=args.name, extra=extra)
    push_kernel(f"{user}/vi-speech-prep-{args.name}", f"vi-speech-prep-{args.name}", code, gpu=False,
                datasets=[f"{user}/{CODE_SLUG}"], kernels=[])


def cmd_tts_prep(args):
    user = username()
    extra = f"--limit-files {args.limit_files}" if args.limit_files else ""
    code = TTS_PREP.format(hf=args.hf, name=args.name, extra=extra, asr=args.asr)
    push_kernel(f"{user}/vi-speech-tts-prep-{args.name}", f"vi-speech-tts-prep-{args.name}", code, gpu=True,
                datasets=[f"{user}/{CODE_SLUG}"], kernels=[])


def cmd_train(args):
    user = username()
    slug = f"{user}/vi-speech-train-{args.run}"
    kernels = [f"{user}/vi-speech-prep-{p}" for p in args.prep]
    if args.resume:
        kernels.append(slug)  # output phiên trước của chính kernel này
    code = TRAIN.format(run=args.run, config=args.config, max_hours=args.max_hours)
    push_kernel(slug, f"vi-speech-train-{args.run}", code, gpu=True, datasets=[f"{user}/{CODE_SLUG}"], kernels=kernels)


def cmd_tts_train(args):
    user = username()
    slug = f"{user}/vi-speech-train-{args.run}"
    kernels = [f"{user}/vi-speech-tts-prep-{p}" for p in args.prep] + ([slug] if args.resume else [])
    code = TTS_TRAIN.format(run=args.run, config=args.config, max_hours=args.max_hours)
    push_kernel(slug, f"vi-speech-train-{args.run}", code, gpu=True, datasets=[f"{user}/{CODE_SLUG}"], kernels=kernels)


def cmd_stt_exp(args):
    user = username()
    code = STT_EXP.format(configs=repr(args.configs), max_hours=args.max_hours)
    push_kernel(f"{user}/vi-speech-exp-{args.name}", f"vi-speech-exp-{args.name}", code, gpu=True,
                datasets=[f"{user}/{CODE_SLUG}"], kernels=[f"{user}/vi-speech-prep-{p}" for p in args.prep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push-code")
    p.add_argument("-m", "--message", default="update")
    p = sub.add_parser("prep")
    p.add_argument("--hf", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--limit-files", type=int, default=0)
    p = sub.add_parser("tts-prep")
    p.add_argument("--hf", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--asr", default="vinai/PhoWhisper-medium")
    p.add_argument("--limit-files", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--prep", nargs="+", required=True)
    p.add_argument("--config", default="configs/stt_small.yaml")
    p.add_argument("--run", default="stt-small")
    p.add_argument("--max-hours", type=float, default=11.3)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("tts-train")
    p.add_argument("--prep", nargs="+", required=True)
    p.add_argument("--config", default="configs/tts_matcha.yaml")
    p.add_argument("--run", default="tts-matcha")
    p.add_argument("--max-hours", type=float, default=11.3)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("stt-exp")
    p.add_argument("--name", required=True)
    p.add_argument("--configs", nargs="+", required=True, help="tối đa 1 cấu hình / GPU")
    p.add_argument("--prep", nargs="+", default=["vlsp", "infore"])
    p.add_argument("--max-hours", type=float, default=2.0)
    p = sub.add_parser("status")
    p.add_argument("slug")
    p = sub.add_parser("output")
    p.add_argument("slug")
    p.add_argument("--dest", default="runs/kaggle")
    args = ap.parse_args()

    if args.cmd == "push-code":
        cmd_push_code(args)
    elif args.cmd == "prep":
        cmd_prep(args)
    elif args.cmd == "tts-prep":
        cmd_tts_prep(args)
    elif args.cmd == "stt-exp":
        cmd_stt_exp(args)
    elif args.cmd == "tts-train":
        cmd_tts_train(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "status":
        print(kaggle("kernels", "status", args.slug))
    elif args.cmd == "output":
        os.makedirs(args.dest, exist_ok=True)
        print(kaggle("kernels", "output", args.slug, "-p", args.dest))


if __name__ == "__main__":
    sys.exit(main())
