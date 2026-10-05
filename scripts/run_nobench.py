#!/usr/bin/env python3
"""YOLOX の tools/*.py スクリプトを cudnn.benchmark を強制的にオフにして実行する（CUDNN_BENCHMARK=1 のときを除く）。

tools/train.py は `cudnn.benchmark = True` を設定する。マルチスケール学習では入力サイズが
変わるたびに cuDNN のアルゴリズム探索が走る。このラッパーは YOLOX を編集せずにその代入を無効化する。
使い方: python run_nobench.py tools/train.py <train.py の引数...>
"""
import os
import runpy
import sys

import torch

if os.environ.get("CUDNN_BENCHMARK") != "1":
    cudnn_mod_cls = type(torch.backends.cudnn)
    cudnn_mod_cls.benchmark = property(
        lambda self: False,
        lambda self, value: torch._C._set_cudnn_benchmark(False),
    )
    torch._C._set_cudnn_benchmark(False)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
