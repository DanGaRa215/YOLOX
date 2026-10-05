# Colab 手順（YOLOX-s, Wind Farms）

2026-10 に無料版 Colab（T4、Python 3.13、torch 2.11）で実際に行った手順。以下は各セルに貼る想定。

## 0. Drive に置くもの
- ローカルで `convert_yolo_to_coco.py --mode copy` により変換したデータ（`datasets_upload/windfarm/`）を zip 1 個にまとめ、
  Drive の `MyDrive/windfarm.zip` に置く（約 640MB）。
  symlink 版（`--mode symlink`）は Drive 上でリンクが切れるため使わない。

## 1. YOLOX の取得と依存インストール
```bash
cd /content
git clone -q https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
# ONNX は今回使わない。onnx / onnx-simplifier は新しい Python でビルドに失敗しやすいので除外する
sed -i '/onnx/d' requirements.txt
pip install -q -r requirements.txt
pip install -q pycocotools thop ninja loguru tabulate
wget -q https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```
`pip install -e .` は新しい setuptools で失敗する（`setup.py develop` が動かない）ため、インストールせず
`PYTHONPATH=/content/YOLOX` で読み込む。

このリポジトリの `exps/*.py`、`eval_pr.py`、`scripts/run_nobench.py` を `/content/YOLOX/` 以下の同じ相対パスに置く
（`run_nobench.py` は `/content/YOLOX/run_nobench.py`）。

## 2. Drive のマウントとデータ展開（ローカルディスクに置いて読み込みを速くする）
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
mkdir -p /content/datasets && cd /content/datasets
unzip -q -o /content/drive/MyDrive/windfarm.zip
ls /content/datasets/windfarm   # annotations test2017 train2017 val2017
```

## 3. 学習（25 epoch、最後の 8 epoch は Mosaic/MixUp オフ）
共通の環境変数:
```bash
export PYTHONPATH=/content/YOLOX
export YOLOX_DATA_DIR=/content/datasets/windfarm
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch 2.6 以降で YOLOX の ckpt 読み込みを通すため
```
```bash
cd /content/YOLOX
python run_nobench.py tools/train.py -f exps/yolox_s_windfarm.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram
# 出力: YOLOX_outputs/yolox_s_windfarm/{best_ckpt.pth,last_epoch_ckpt.pth,train_log.txt}
```
- `--cache ram`: 縮小済み画像を RAM に置く（約 1.7GB）。Colab の CPU は 2 コアしかなく、1920x1080 の JPEG を毎回デコードすると
  1 エポック目の見積もりが約 4 時間になったため付けた。学習結果は変わらない。
- `run_nobench.py`: `cudnn.benchmark` をオフにして起動するラッパー。計測では、定常時の速度は変わらず、
  学習開始直後のアルゴリズム探索の分だけ短くなった。ベースラインはこれを使わず `python tools/train.py ...` で学習した。
- 定常時は T4 で 1 iter 約 0.6 秒、1 本あたり約 50 分（学習 25 epoch + 評価）。
- ノートブックのセルで直接実行すると、セルを中断したときに学習も止まる。実際には `subprocess.Popen(..., start_new_session=True)`
  でバックグラウンド起動し、`scripts/queue2.sh` で施策 3 本を順に学習・評価した。

## 4. 評価
クラス別・全体の Precision / Recall（IoU 0.5、conf 0.3、NMS 0.45 で固定。全実験で同じ値を使う）:
```bash
CK=YOLOX_outputs/yolox_s_windfarm/best_ckpt.pth
for S in val test; do
  python eval_pr.py -f exps/yolox_s_windfarm.py -c $CK --data-dir $YOLOX_DATA_DIR \
    --split $S --conf 0.3 --nms 0.45 --out results/yolox_s_windfarm_$S --fp16
done
```
COCO mAP（YOLOX 内蔵、参考値）:
```bash
EVAL_SPLIT=val  python tools/eval.py -f exps/yolox_s_windfarm.py -c $CK -d 1 -b 16 --conf 0.001 --fp16 --fuse
EVAL_SPLIT=test python tools/eval.py -f exps/yolox_s_windfarm.py -c $CK -d 1 -b 16 --conf 0.001 --fp16 --fuse
```
`results/<exp名>_{val,test}.{json,csv}` にクラス別の TP/FP/FN、P、R が保存される。結果とログは
`MyDrive/windfarm_outputs/` にコピーしておく（ランタイムが切れても残る）。

## メモ
- 無料版 Colab では GPU セッションを同時に 1 つしか使えなかった（2 つ目は「アクティブなセッションが多すぎます」）。
- 入力サイズは YOLOX 既定の 640（学習時は multiscale で 480〜800）。元画像は主に 1920x1080 で、小さい turbine は縮小の影響を受ける。
- バッチサイズ 16 で OOM になる場合は `-b 8`。
