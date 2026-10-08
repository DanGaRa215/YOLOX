#!/usr/bin/env python3
"""eval_pr.py を MPS で動かすためのラッパー（eval_pr.py 本体と YOLOX は変更しない）。

YOLOX の decode_outputs は `tensor.type(dtype)` に Tensor.type() が返した型文字列を渡す。MPS では
'torch.mps.FloatTensor' が再入力できず ValueError になるため、Tensor.type を差し替えてから eval_pr.py を実行する。
PYTORCH_ENABLE_MPS_FALLBACK=1 も設定する（postprocess の NMS などで未対応演算があれば CPU にフォールバック）。

使い方: PYTHONPATH=<YOLOX の clone>:<exp のあるディレクトリ> python scripts/eval_pr_mps.py <eval_pr.py のパス> <eval_pr.py の引数> --device mps
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import runpy
import sys

import torch

_MAP = {
    "torch.mps.FloatTensor": torch.float32,
    "torch.mps.HalfTensor": torch.float16,
    "torch.mps.LongTensor": torch.int64,
    "torch.mps.BoolTensor": torch.bool,
}
_orig = torch.Tensor.type


def _type(self, dtype=None, *a, **k):
    if isinstance(dtype, str) and dtype in _MAP:
        return self.to(device="mps", dtype=_MAP[dtype])
    return _orig(self, dtype, *a, **k)


torch.Tensor.type = _type

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
