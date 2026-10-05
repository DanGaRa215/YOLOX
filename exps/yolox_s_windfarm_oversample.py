#!/usr/bin/env python3
# 改善 (A): "cable tower" を含む学習画像をオーバーサンプリングする。
# yolox_s_windfarm.py との違いはこの一点のみ: 学習データセットを *仮想* インデックスビューで包み、
# cable tower を 1 つ以上含む画像が K 回ずつ現れるようにする。
#
# 環境変数（ベースラインのものに加えて）:
#   OVERSAMPLE_K         cable tower 画像の繰り返し倍率 K（整数 >= 1）。既定 3
#   OVERSAMPLE_CLASS_ID  オーバーサンプリング対象の COCO category_id。既定 1 (cable tower)
#
# 学習:  python tools/train.py -f exps/yolox_s_windfarm_oversample.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram
#
# 設計メモ（YOLOX main で確認済み）:
#  * MosaicDetection は、メインのインデックスをサンプラー（range(len(MosaicDetection)) == len(view)）から、
#    Mosaic の追加 3 枚を random.randint(0, len(self._dataset)-1) から、MixUp の相手を
#    random.randint(0, self.__len__()-1) から引く。いずれもビューを参照するため、3 つとも重み付けされる。
#  * ビューは index -> ベースの index を対応づけるだけで、画像は複製しない。したがって `--cache ram`
#    （CacheDataset.imgs。get_data_loader より前に元の 2643 枚で構築される）は元のメモリ量のままである。
#  * 1 エポック = len(view) サンプルなので、エポックは (K-1)*(cable tower 画像数) イテレーション長くなる。
#  * close_mosaic() は batch_sampler.mosaic を切り替えるだけなので no-aug フェーズも動作し、
#    オーバーサンプリングもそこで有効なままである（メインのインデックスは同じサンプラーから来る）。
#    評価用データセットは変更しない。
import os

import numpy as np

# tools/train.py は exp ファイルのディレクトリを sys.path に追加するため、ベースラインを名前で import できる。
from yolox_s_windfarm import Exp as BaseExp


class OversampledDataset:
    """COCODataset に対するインデックスビュー: index i -> ベースのインデックス idx_map[i]。画像はコピーしない。"""

    def __init__(self, base, idx_map):
        self._base = base
        self._idx_map = list(idx_map)

    def __len__(self):
        return len(self._idx_map)

    def pull_item(self, index):
        return self._base.pull_item(self._idx_map[index])

    def load_anno(self, index):
        return self._base.load_anno(self._idx_map[index])

    # MosaicDetection は `dataset.input_dim` を読み、`dataset._input_dim` に代入するため、どちらもベースへ転送する。
    @property
    def input_dim(self):
        return self._base.input_dim

    @property
    def _input_dim(self):
        return self._base.input_dim

    @_input_dim.setter
    def _input_dim(self, value):
        self._base._input_dim = value

    def __getattr__(self, name):  # 通常の属性検索に失敗したときのみ呼ばれる
        if name in ("_base", "_idx_map"):
            raise AttributeError(name)
        return getattr(self._base, name)


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # BaseExp は自身の __file__ から exp_name を決めるため、出力を別ディレクトリに分ける。
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

        # 基底クラスと同じ前提: キャッシュ使用時は train.py が self.dataset を構築済みである。
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
