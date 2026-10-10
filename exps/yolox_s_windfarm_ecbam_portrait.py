#!/usr/bin/env python3
# 施策 D+E の組み合わせ: モデルは施策 D（E-CBAM 入り）、学習データは施策 E（縦長写真の Repeat Factor Sampling）。
# それ以外（エポック数・拡張・損失など）はベースラインのまま。
#
# 実装は多重継承だけで、コードは複製しない。
#  * get_model                      <- yolox_s_windfarm_ecbam.py（E-CBAM を挿入した backbone）
#  * _build_index_map/get_data_loader <- yolox_s_windfarm_portrait.py（縦長かつ cable tower の画像を繰り返す）
#  MRO は Exp -> portrait.Exp -> oversample.Exp -> ecbam.Exp -> ベースライン。
#  portrait / oversample は get_model を定義せず、ecbam は get_dataset 系を定義しないため、役割が衝突しない。
#
# 環境変数: PORTRAIT_RFS_T / OVERSAMPLE_CLASS_ID（施策 E）、YOLOX_DATA_DIR / EVAL_SPLIT（ベースライン）。
# 学習:  python tools/train.py -f exps/yolox_s_windfarm_ecbam_portrait.py -c yolox_s.pth -d 1 -b 16 --fp16 --cache ram
import os

# tools/train.py は exp ファイルのディレクトリを sys.path に追加するため、名前で import できる。
from yolox_s_windfarm_ecbam import Exp as EcbamExp
from yolox_s_windfarm_portrait import Exp as PortraitExp


class Exp(PortraitExp, EcbamExp):
    def __init__(self):
        super().__init__()
        # 各親は自身の __file__ から exp_name を決めるため、出力が衝突しないよう上書きする。
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
