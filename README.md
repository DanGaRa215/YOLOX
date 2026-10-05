# YOLOX-s による風力発電所の検出（鉄塔 / タービン）

Roboflow の "Wind Farms" v5 データセット（CC BY 4.0、作者 Kyle Graupe）で
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) をファインチューニングし、
クラス別の Precision / Recall を比較する実験。

## データ
- 2 クラス: `cable tower`（学習インスタンス 441）/ `turbine`（15,534）。約 35:1 の不均衡。
- train 2643 / valid 247 / test 130 枚。データセットはこのリポジトリに含まれない。
- `convert_yolo_to_coco.py` は Roboflow の YOLO txt エクスポートを YOLOX が読む COCO 形式に変換する。
  `verify_coco.py` は件数の確認と bbox の描画を行う。

## 実験（全て 25 epoch、YOLOX-s、COCO 事前学習済みの `yolox_s.pth`）
| Exp ファイル | ベースラインからの変更点 |
|---|---|
| `exps/yolox_s_windfarm.py` | ベースライン（`no_aug_epochs=8`） |
| `exps/yolox_s_windfarm_noaug12.py` | (C) close-mosaic 期間を延長: `no_aug_epochs=12` |
| `exps/yolox_s_windfarm_clsweight.py` | (B) クラス重み付き cls loss（`CLS_WEIGHTS`、既定 `4,1`） |
| `exps/yolox_s_windfarm_oversample.py` | (A) cable tower を含む画像をオーバーサンプリング（`OVERSAMPLE_K`、既定 3） |

## 評価
`eval_pr.py` は IoU 0.5、conf 0.3、NMS 0.45 でクラス別および micro の Precision / Recall を出力する
（`test_eval_pr.py` はそのマッチング処理の単体テスト）。COCO mAP は YOLOX の `tools/eval.py` で求める。

## Colab での実行
`colab_setup.md` を参照。`scripts/` には Colab で実際に使ったヘルパーがある。
`run_nobench.py`（`cudnn.benchmark` をオフにして YOLOX のツールを実行）、`queue2.sh`（3 つの
バリアントを順に学習・評価）、`bench.sh`（`cudnn.benchmark` の on/off の所要時間計測）。
