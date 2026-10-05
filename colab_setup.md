# Colab 手順 (YOLOX-s baseline, Wind Farms)

ランタイム: GPU (T4 以上)。以下は各セルに貼る想定。

## 0. Drive に置くもの（ローカルで準備済み）
- `yolox_task/datasets_upload/windfarm/`（約 643MB、画像は **実体コピー**、`--mode copy` 変換済み）を Drive の `MyDrive/windfarm_task/windfarm/` にアップロード。
  Colab/Drive では symlink が切れるため symlink 版 (`datasets/windfarm`) は使わないこと。
  圧縮して上げる場合は zip 1個にして Colab 側で展開すると速い。
- `exps/yolox_s_windfarm.py`, `eval_pr.py` を Drive の `MyDrive/windfarm_task/` にアップロード。

## 1. Drive マウントとデータ配置（ローカルディスクにコピーして高速化）
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
mkdir -p /content/datasets
cp -r /content/drive/MyDrive/windfarm_task/windfarm /content/datasets/windfarm
ls /content/datasets/windfarm /content/datasets/windfarm/annotations
```

## 2. YOLOX の clone・依存インストール
```bash
cd /content
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
# ONNX は今回使わない。onnx / onnx-simplifier は Python 3.12 でビルド失敗しやすいので除外
sed -i '/onnx/d' requirements.txt
pip install -r requirements.txt
pip install -v -e . --no-build-isolation
pip install pycocotools
# numpy 2.x 系で import エラーが出る場合: pip install "numpy<2" してランタイム再起動
```
```bash
cd /content/YOLOX
wget -q https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
cp /content/drive/MyDrive/windfarm_task/exps/yolox_s_windfarm.py exps/yolox_s_windfarm.py
cp /content/drive/MyDrive/windfarm_task/eval_pr.py .
```

## 3. 学習（ベースライン: 50 epoch, 最後の 15 epoch は Mosaic/MixUp オフ）
```bash
cd /content/YOLOX
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export YOLOX_DATA_DIR=/content/datasets/windfarm
python tools/train.py -f exps/yolox_s_windfarm.py -c yolox_s.pth -d 1 -b 16 --fp16 -o
# 出力: YOLOX_outputs/yolox_s_windfarm/{best_ckpt.pth,last_epoch_ckpt.pth,train_log.txt}
```
学習中の COCO mAP は valid で 5 epoch ごとに評価される。ランタイム切断対策に
`YOLOX_OUTPUTS` を Drive にコピーするか、`--resume -c <last_epoch_ckpt.pth>` で再開する。
```bash
cp -r YOLOX_outputs /content/drive/MyDrive/windfarm_task/YOLOX_outputs_baseline
```

## 4. 評価
COCO mAP（YOLOX 内蔵）:
```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CK=YOLOX_outputs/yolox_s_windfarm/best_ckpt.pth
EVAL_SPLIT=val  python tools/eval.py -f exps/yolox_s_windfarm.py -c $CK -d 1 -b 16 --conf 0.001 --fp16 --fuse
EVAL_SPLIT=test python tools/eval.py -f exps/yolox_s_windfarm.py -c $CK -d 1 -b 16 --conf 0.001 --fp16 --fuse
```
クラス別/全体 Precision・Recall（IoU=0.5, conf=0.3, NMS=0.45 固定。全実験で同じ値を使うこと）:
```bash
for S in val test; do
python eval_pr.py -f exps/yolox_s_windfarm.py -c $CK --data-dir /content/datasets/windfarm \
  --split $S --conf 0.3 --nms 0.45 --out results/baseline_$S
done
cp -r results /content/drive/MyDrive/windfarm_task/results_baseline
```
`results/baseline_{val,test}.json|csv` にクラス別 TP/FP/FN, P, R が保存される。

## メモ
- 入力サイズは YOLOX デフォルト 640x640。元画像は主に 1920x1080 で、小さい turbine は縮小の影響を受ける（施策候補）。
- バッチサイズ 16 で OOM の場合は `-b 8`。
- 環境変数: `YOLOX_DATA_DIR`（データルート）, `EVAL_SPLIT`（val/test）。
