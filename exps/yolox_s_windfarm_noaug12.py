#!/usr/bin/env python3
# 施策 (C): close-mosaic 期間を延長する。no_aug_epochs をベースラインの 15 -> 25 にする（max_epoch は 50 なので、
# Mosaic/MixUp を使うのはエポック 1-35 ではなく 1-25 になる）。それ以外はベースラインと同じ。
# ファイル名の noaug12 は、25 エポックで実験していたとき（no_aug_epochs 8 -> 12）の名残。
#
# 仮説: Mosaic/MixUp の合成で大きな cable tower（sqrt(area) の中央値が約 200px）が切断・縮小され、
# モデルが無傷の tower を見る機会が少ない。拡張なし（元画像、L1 loss オン）の最終フェーズを
# 長くすれば cable tower の検出が改善するはずである。
#
# 学習:  python tools/train.py -f exps/yolox_s_windfarm_noaug12.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
import os

from yolox_s_windfarm import Exp as BaseExp  # ベースラインと同様、同じディレクトリ (exps/) が sys.path にある


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ベースラインは自身の __file__ から exp_name を決めるため、出力が衝突しないよう上書きする
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.no_aug_epochs = 25
