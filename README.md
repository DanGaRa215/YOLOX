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

`summarize_results.py` は `results/` の `<exp名>_{val,test}.json`（eval_pr の出力）と
`<exp名>_{val,test}_cocomap.txt`（tools/eval.py のログ）から、実験 × split の比較表（ベースラインとの差分つき）を作る。
`python summarize_results.py --results-dir results --out results/summary` で `.md` と `.csv` を書き、Markdown は標準出力にも出す
（未実施の実験・split は「未実施」と表示。テストは `python test_summarize_results.py`）。

### seed をまたいだ集計（`--aggregate-seeds`）
`python summarize_results.py --results-dir results --aggregate-seeds --out results/summary_seeds` で、
seed 違いの結果をまとめた表（平均 ± 標準偏差、本数 n つき）を出す。付けなければ従来の表のまま。
- グループ化: 実験名の末尾の `_s<数字>` を取り除いた名前でまとめる。末尾にそれが無い名前（例 `yolox_s_windfarm`）は seed 1 として
  同じグループに入る（`yolox_s_windfarm` と `yolox_s_windfarm_s2` は同じグループ）。名前の途中の `_s_` には反応しない。
- 指標: クラス別（cable tower / turbine / 全体）の P・R・TP・FP・FN と AP50・AP50:95。標準偏差は標本標準偏差（n-1）。
  n=1 のときは `±` を付けず平均だけ。ある指標だけ本数が少ないときは `(n=2)` のように付く。
- `Δ` 列はベースライン（`yolox_s_windfarm` のグループ）との平均の差（P・R・AP はポイント、TP・FP・FN は件数）。
- `--out` を付けると、この表が `.md` と `.csv` に書かれる。

## Colab での実行
`colab_setup.md` を参照。`scripts/` には Colab で実際に使ったヘルパーがある。
`run_nobench.py`（`cudnn.benchmark` をオフにして YOLOX のツールを実行）、`queue2.sh`（3 つの
バリアントを順に学習・評価）、`bench.sh`（`cudnn.benchmark` の on/off の所要時間計測）。
