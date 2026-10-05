#!/usr/bin/env python3
# Measure (C): extend the close-mosaic period, no_aug_epochs 8 -> 12 (max_epoch stays 25, so
# Mosaic/MixUp are used for epochs 1-13 instead of 1-17). Everything else is the baseline.
#
# Hypothesis: Mosaic/MixUp composites cut and shrink the large cable towers (median sqrt(area)
# ~200px), so the model sees few intact towers. Finishing with a longer augmentation-free phase
# (original images, L1 loss on) should improve cable tower detection.
#
# Train:  python tools/train.py -f exps/yolox_s_windfarm_noaug12.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
import os

from yolox_s_windfarm import Exp as BaseExp  # same dir (exps/) is on sys.path, like the baseline


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # baseline derives exp_name from ITS __file__; override so outputs do not collide
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.no_aug_epochs = 12
