#!/usr/bin/env python3
# 改善 (E): 縦長の地上写真（cable tower を含むもの）を Repeat Factor Sampling 風に多めに学習させる。
#
# 【根拠】
#  * エラー分析: cable tower の見逃しは、縦長（height > width）の地上写真（pexels、1365x2048 など）に集中している。
#    test ではベースラインがこの種類の cable tower を 0/11 しか検出できていない。
#    train の縦長画像は 66 枚で、うち cable tower を含むものは 21 枚（48 インスタンス）しかなく、学習で見る機会が少ない。
#  * LVIS の Repeat Factor Sampling（Gupta ら, CVPR 2019）は、クラス c の出現画像率 f_c から
#        r_c = max(1, sqrt(t / f_c))
#    を求め、画像の繰り返し係数をその画像が含むクラスの r_c の最大値とする。頻度の低いクラスほど多く繰り返す。
#    ここではこれを「クラス」ではなく「画像の種類（縦長 かつ cable tower を含む）」に当てはめる:
#        f = 対象画像数 / train の全画像数,   r = max(1, sqrt(t / f))
#    対象画像は r を整数にして繰り返し、それ以外は 1 回とする。
#  * LVIS は r の小数部を確率的に丸める（エポックごとに ceil / floor を確率 frac で選ぶ）。
#    ここでは施策 A と同じ仮想インデックスビューを使うため、決定的に round(r) 回とする。
#    （仮想エポックの長さを固定でき、ログと比較が簡単になる。期待値は小数部の分だけ LVIS とずれる。）
#
# 施策 A（yolox_s_windfarm_oversample.py）との違いは繰り返し回数の決め方だけである:
#  A: cable tower を含む全画像を一律 K 倍 / E: 縦長かつ cable tower を含む画像だけを round(r) 倍。
#  OversampledDataset（仮想 index ビュー）は施策 A のファイルから import して再利用する。
#  そのため `--cache ram` と共存し、Mosaic の追加画像や MixUp の相手にも効く（詳細は施策 A のファイル冒頭を参照）。
#
# 環境変数（ベースラインのものに加えて）:
#   PORTRAIT_RFS_T         RFS の閾値 t。既定 0.1（大きいほど多く繰り返す）
#   OVERSAMPLE_CLASS_ID    対象クラスの COCO category_id。既定 1 (cable tower)
#
# 学習:  python tools/train.py -f exps/yolox_s_windfarm_portrait.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram
import math
import os

import numpy as np

# tools/train.py は exp ファイルのディレクトリを sys.path に追加するため、名前で import できる。
from yolox_s_windfarm_oversample import Exp as OversampleExp
from yolox_s_windfarm_oversample import OversampledDataset


class Exp(OversampleExp):
    def __init__(self):
        super().__init__()
        # exp_name はファイル名から作る（出力を別ディレクトリに分ける）。
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.rfs_t = float(os.environ.get("PORTRAIT_RFS_T", "0.1"))
        if self.rfs_t <= 0:
            raise ValueError("PORTRAIT_RFS_T must be > 0")

    def _build_index_map(self, base):
        cls_idx = base.class_ids.index(self.oversample_category_id)
        n = len(base.annotations)
        is_target = []
        for i in range(n):
            labels = base.annotations[i][0]
            has = len(labels) > 0 and bool(np.any(labels[:, 4] == cls_idx))
            info = base.coco.imgs[base.ids[i]]  # images[].width / height
            portrait = info["height"] > info["width"]
            is_target.append(has and portrait)
        n_pos = int(sum(is_target))
        f = n_pos / n
        r = max(1.0, math.sqrt(self.rfs_t / f)) if n_pos > 0 else 1.0
        rep = int(round(r))
        idx_map = []
        for i in range(n):
            idx_map.extend([i] * (rep if is_target[i] else 1))
        self.portrait_stats = (n, n_pos, f, r, rep, len(idx_map))
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
            n, n_pos, f, r, rep, n_virt = self.portrait_stats
            logger.info(
                f"[portrait-rfs] t={self.rfs_t}: 対象（縦長かつ category {self.oversample_category_id}）"
                f" {n_pos}/{n} 枚, f={f:.4f}, r={r:.3f} -> {rep} 回繰り返し; "
                f"仮想エポック長 {n} -> {n_virt} 枚 (約 {-(-n_virt // batch_size)} iter @ batch {batch_size})"
            )
        # 施策 A の get_data_loader は isinstance 判定でビュー化を飛ばし、そのまま基底へ進む。
        return super().get_data_loader(batch_size, is_distributed, no_aug=no_aug, cache_img=cache_img)
