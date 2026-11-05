"""EMA (exponential moving average) trọng số: giữ bản trung bình trượt của tham số khi train.

Bản EMA thường cho WER/chất lượng tốt hơn trọng số tại bước cuối (ít nhiễu hơn), gần như không tốn
compute. Đánh giá và export dùng bản EMA; train vẫn cập nhật trọng số gốc.
"""

from __future__ import annotations

import contextlib

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        # đầu quá trình train dùng decay nhỏ hơn để EMA không "kẹt" ở trọng số khởi tạo
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, sd: dict):
        self.shadow = {k: v.float() for k, v in sd["shadow"].items()}
        self.num_updates = sd["num_updates"]
        self.decay = sd.get("decay", self.decay)

    def weights(self, model: torch.nn.Module) -> dict:
        """state_dict của model với tham số thay bằng bản EMA (để lưu / export)."""
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v) for k, v in sd.items()}

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        """Tạm thay trọng số model bằng EMA (vd khi đánh giá), xong trả lại."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.weights(model))
        try:
            yield model
        finally:
            model.load_state_dict(backup)
