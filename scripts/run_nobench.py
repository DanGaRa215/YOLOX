#!/usr/bin/env python3
"""Run a YOLOX tools/*.py script with cudnn.benchmark forced off (unless CUDNN_BENCHMARK=1).

tools/train.py sets `cudnn.benchmark = True`; with multiscale training every new input size
triggers a cuDNN algorithm search. This wrapper makes that assignment a no-op without editing YOLOX.
Usage: python run_nobench.py tools/train.py <train.py args...>
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
