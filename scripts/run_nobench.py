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

# YOLOX の exp.merge は、既定値が None の属性（seed）を文字列のまま代入する（型変換は既定値が None でないときだけ）。
# `seed 2` を末尾の opts で渡しても exp.seed が "2" になるため、merge の後に int へ直す。
# （torch.manual_seed や InfiniteSampler は内部で int() するので動くが、random.seed("2") は random.seed(2) と
# 別の系列になる。型を揃えておく。）yolox が import できない環境では何もしない。
try:
    from yolox.exp.base_exp import BaseExp

    _orig_merge = BaseExp.merge

    def _merge_int_seed(self, cfg_list):
        _orig_merge(self, cfg_list)
        if isinstance(getattr(self, "seed", None), str):
            self.seed = int(self.seed)

    BaseExp.merge = _merge_int_seed
except ImportError:
    pass

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
