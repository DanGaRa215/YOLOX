#!/usr/bin/env python3
# 施策 (D): 注目機構 E-CBAM を backbone（CSPDarknet）と neck（PAFPN）の間に挿入する。
# ベースライン (exps/yolox_s_windfarm.py) との違いは注目モジュールの追加だけで、
# スケジュール・データ・拡張・損失はそのまま継承する。
#
# 【根拠】
#   B. Zeng et al., "Lightweight insulator target detection algorithm based on improved YOLOX,"
#   Scientific Reports, 2025. doi:10.1038/s41598-025-04023-2 (PMC12130486)
#   元の CBAM: Woo et al., ECCV 2018 (arXiv:1807.06521)
#
# 【E-CBAM の構造（論文本文で確認できた範囲）】
#   - YOLOX の backbone と neck（特徴ピラミッド）の間に挿入する。
#   - CBAM のチャネル注意を ECA (Efficient Channel Attention) に置き換え、空間注意 (SAM) と直列に組む（ECA -> SAM）。
#   - ECA: GAP でチャネルごとにスカラー化 -> 1 次元畳み込み（FC 層なし、チャネル圧縮なし）-> Sigmoid
#          -> 入力に要素積。1D 畳み込みのカーネルサイズはチャネル数 C から適応的に決める（式 (5)）。
#   - SAM: チャネル方向の Max プーリングと Average プーリングを concat -> 7x7 畳み込み -> Sigmoid -> 要素積。
#
# 【本文で確認できなかった点（推測で作らず、下記のとおり代用した）】
#   - 何枚の特徴マップに入れるか: 記載なし。-> dark3/dark4/dark5 の 3 枚すべてに、スケールごとの別モジュールで入れた。
#   - ECA のカーネルサイズの具体値: 式 (5) は本文にあるが具体値の例はない。論文の記号 (b=2, t=1) の読み取りに
#     曖昧さがあるため、ECA 原論文 (Wang et al., CVPR 2020) の標準式 k = |log2(C)/2 + 1/2|_odd を使った。
#     （C=128/256/512 のとき k=5/5/5。論文式の b, t の解釈が違えば値が変わりうる。）
#   - 残差接続・BN・活性化の詳細: 記載なし。ECA / SAM とも標準形（Sigmoid ゲート、BN なし）にした。
#   - SAM の 7x7 畳み込みは論文記載どおり（バイアスなし、CBAM 標準）。
#
# 【論文との違い（fine-tune を安定させるため）】
#   論文は注意の出力をそのまま次段に渡す（と読める。残差ゲートの記載なし）。本実装は COCO 事前学習済みの
#   yolox_s.pth から fine-tune するため、ランダム初期化の注目モジュールが事前学習済みの特徴を壊さないよう、
#     out = x + gamma * (attn(x) * x - x)
#   とし、gamma を 0 で初期化する。初期状態は厳密に恒等写像（ベースラインと出力が一致）で、
#   学習で gamma が育つにつれて注意が効く。gamma=1 で論文どおりの E-CBAM になる。
#   なお ECA / SAM の畳み込み重みは gamma=0 の間は勾配が 0 で、まず gamma が動いてから学習が始まる。
#   gamma は「weight」という名前のパラメータを持つ小モジュール (_Gate) に入れている。YOLOX の
#   get_optimizer は .weight / .bias を持つモジュールのパラメータしかオプティマイザに登録しないため、
#   素の nn.Parameter にすると学習されない。
#
# 【チェックポイント互換】
#   既存のキー名は変わらない（YOLOPAFPN のサブクラスで、属性を足すだけ）。追加キーは
#   backbone.attn_dark{3,4,5}.* のみ。yolox_s.pth は load_ckpt で既存層がすべて読み込まれ、
#   新しい層だけが初期化のまま残る（"is not in the ckpt" の警告が追加キー分だけ出るのは想定どおり）。
#
# 環境変数: ベースラインの YOLOX_DATA_DIR / EVAL_SPLIT と同じ。
# 学習:  python tools/train.py -f exps/yolox_s_windfarm_ecbam.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
import math
import os

import torch
import torch.nn as nn
from loguru import logger

from yolox.models import YOLOPAFPN

from yolox_s_windfarm import Exp as BaseExp  # ベースラインと同様、同じディレクトリ (exps/) が sys.path にある


def _eca_kernel_size(channels, gamma=2, b=1):
    """ECA 原論文の適応カーネルサイズ k = |log2(C)/gamma + b/gamma|_odd。"""
    k = int(abs(math.log2(channels) / gamma + b / gamma))
    return k if k % 2 else k + 1


class ECA(nn.Module):
    """チャネル注意: GAP -> 1D 畳み込み -> Sigmoid（FC なし）。重み (B, C, 1, 1) を返す。"""

    def __init__(self, channels):
        super().__init__()
        k = _eca_kernel_size(channels)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean(dim=(2, 3))  # (B, C)  GAP
        y = self.conv(y.unsqueeze(1))  # (B, 1, C) チャネル方向の 1D 畳み込み
        return torch.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)


class SAM(nn.Module):
    """空間注意: チャネル方向の Max / Avg を concat -> 7x7 畳み込み -> Sigmoid。重み (B, 1, H, W) を返す。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        y = torch.cat([x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(y))


class _Gate(nn.Module):
    """スカラー gamma（0 初期化）。属性名は weight にする（YOLOX の get_optimizer に拾わせるため）。"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class ECBAM(nn.Module):
    """E-CBAM（ECA -> SAM の直列）+ 恒等写像から始める残差ゲート: out = x + gamma * (attn(x) - x)。"""

    def __init__(self, channels):
        super().__init__()
        self.eca = ECA(channels)
        self.sam = SAM(7)
        self.gate = _Gate()

    def forward(self, x):
        y = x * self.eca(x)  # チャネル注意
        y = y * self.sam(y)  # 空間注意
        g = self.gate.weight.to(x.dtype)  # fp16 / bf16 の autocast でも dtype が float32 に昇格しないように合わせる
        return x + g * (y - x)


class ECBAMPAFPN(YOLOPAFPN):
    """YOLOPAFPN の forward を上書きし、backbone の dark3/4/5 に neck へ入る前に E-CBAM をかける。"""

    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__(depth, width, in_features, in_channels, depthwise, act)
        # 属性名は attn_dark3 など（"bn" を含めない: get_optimizer が名前に "bn" を含むモジュールを特別扱いするため）
        for name, c in zip(in_features, in_channels):
            setattr(self, f"attn_{name}", ECBAM(int(c * width)))

    def forward(self, input):
        out_features = self.backbone(input)
        # 注目機構を適用（ここだけが上流の YOLOPAFPN.forward との違い。以降は同一）
        features = [getattr(self, f"attn_{f}")(out_features[f]) for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        return (pan_out2, pan_out1, pan_out0)


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ベースラインは自身の __file__ から exp_name を決めるため、出力が衝突しないよう上書きする
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        # 上流の Exp.get_model のコピー。backbone（YOLOPAFPN）を ECBAMPAFPN に差し替えている
        from yolox.models import YOLOX, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = ECBAMPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

            # 追加パラメータ数をログに出す（ベースラインの YOLOPAFPN + head と比較）
            base = YOLOX(
                YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act),
                YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act),
            )
            n_base = sum(p.numel() for p in base.parameters())
            n_new = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "E-CBAM: 追加パラメータ {:,}（ベースライン {:,} -> {:,}, +{:.4f}%）".format(
                    n_new - n_base, n_base, n_new, 100.0 * (n_new - n_base) / n_base
                )
            )
            del base

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
