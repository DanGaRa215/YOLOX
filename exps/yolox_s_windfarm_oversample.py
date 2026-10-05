#!/usr/bin/env python3
# Improvement (A): oversample train images that contain "cable tower".
# Only change vs. yolox_s_windfarm.py: the training dataset is wrapped in a *virtual* index view
# in which every image containing >=1 cable tower appears K times.
#
# Env vars (in addition to the baseline ones):
#   OVERSAMPLE_K         repeat factor K for cable-tower images (int >= 1), default 3
#   OVERSAMPLE_CLASS_ID  COCO category_id to oversample, default 1 (cable tower)
#
# Train:  python tools/train.py -f exps/yolox_s_windfarm_oversample.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram
#
# Design notes (verified against YOLOX main):
#  * MosaicDetection draws the main index from the sampler (range(len(MosaicDetection)) == len(view)),
#    the 3 extra mosaic images via random.randint(0, len(self._dataset)-1), and the mixup partner via
#    random.randint(0, self.__len__()-1). All of these index the view, so all three are weighted.
#  * The view only maps index -> base index; images are NOT duplicated, so `--cache ram`
#    (CacheDataset.imgs, built on the original 2643 images before get_data_loader) uses the original memory.
#  * One epoch = len(view) samples, i.e. the epoch gets longer by (K-1)*n_cable_tower_images iterations.
#  * close_mosaic() only flips batch_sampler.mosaic, so the no-aug phase still works; oversampling
#    stays active there too (main index comes from the same sampler). Eval datasets are untouched.
import os

import numpy as np

# tools/train.py puts the exp file's directory on sys.path, so the baseline is importable by name.
from yolox_s_windfarm import Exp as BaseExp


class OversampledDataset:
    """Index view over a COCODataset: index i -> base index idx_map[i]. No image is copied."""

    def __init__(self, base, idx_map):
        self._base = base
        self._idx_map = list(idx_map)

    def __len__(self):
        return len(self._idx_map)

    def pull_item(self, index):
        return self._base.pull_item(self._idx_map[index])

    def load_anno(self, index):
        return self._base.load_anno(self._idx_map[index])

    # MosaicDetection reads `dataset.input_dim` and assigns `dataset._input_dim`; forward both to the base.
    @property
    def input_dim(self):
        return self._base.input_dim

    @property
    def _input_dim(self):
        return self._base.input_dim

    @_input_dim.setter
    def _input_dim(self, value):
        self._base._input_dim = value

    def __getattr__(self, name):  # only called when normal lookup fails
        if name in ("_base", "_idx_map"):
            raise AttributeError(name)
        return getattr(self._base, name)


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # BaseExp derives exp_name from its own __file__; keep outputs in a separate dir.
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.oversample_k = int(os.environ.get("OVERSAMPLE_K", "3"))
        if self.oversample_k < 1:
            raise ValueError("OVERSAMPLE_K must be >= 1")
        self.oversample_category_id = int(os.environ.get("OVERSAMPLE_CLASS_ID", "1"))

    def _build_index_map(self, base):
        cls_idx = base.class_ids.index(self.oversample_category_id)
        idx_map = []
        n_pos = 0
        for i in range(len(base.annotations)):
            labels = base.annotations[i][0]
            has = len(labels) > 0 and bool(np.any(labels[:, 4] == cls_idx))
            n_pos += has
            idx_map.extend([i] * (self.oversample_k if has else 1))
        self.oversample_stats = (len(base.annotations), n_pos, len(idx_map))
        return idx_map

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        from loguru import logger

        # Same preconditions as the base class: with a cache, train.py already built self.dataset.
        if self.dataset is None:
            assert cache_img is None, "cache_img must be None if you didn't create self.dataset before launch"
            self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        if not isinstance(self.dataset, OversampledDataset):
            base = self.dataset
            self.dataset = OversampledDataset(base, self._build_index_map(base))
            n, n_pos, n_virt = self.oversample_stats
            logger.info(
                f"[oversample] K={self.oversample_k}: {n_pos}/{n} images contain category "
                f"{self.oversample_category_id}; virtual epoch length {n} -> {n_virt}"
            )
        return super().get_data_loader(batch_size, is_distributed, no_aug=no_aug, cache_img=cache_img)
