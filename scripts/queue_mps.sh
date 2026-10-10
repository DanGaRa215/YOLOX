#!/bin/bash
# Mac（M3、MPS、fp32）で施策 E → D → A を 50 エポックずつ順に学習・評価する。
# 途中で止まっても、同じコマンドで再実行すれば latest_ckpt から --resume し、完了した実験は DONE で飛ばす。
# 使い方: YOLOX_WORK=<作業ディレクトリ> caffeinate -i bash scripts/queue_mps.sh   （caffeinate でスリープを防ぐ）
set -u
# YOLOX_WORK: YOLOX の clone（YOLOX/）と COCO 事前学習済み重み（mps/yolox_s.pth）を置いた作業ディレクトリ
S=${YOLOX_WORK:?環境変数 YOLOX_WORK を設定してください}
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=$R/outputs_mps
PY=$R/.venv/bin/python
export PYTHONPATH=$S/YOLOX:$R/exps
export YOLOX_DATA_DIR=$R/datasets/windfarm
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT/results"
LOG=$OUT/queue_mps.log
log() { echo "[$(date '+%m-%d %T')] $*" | tee -a "$LOG"; }

cd "$R"
for name in yolox_s_windfarm_portrait yolox_s_windfarm_ecbam yolox_s_windfarm_oversample; do
  D=$OUT/$name
  if [ -f "$D/DONE" ]; then log "skip $name (DONE)"; continue; fi
  if [ -f "$D/latest_ckpt.pth" ]; then
    CK="--resume -c $D/latest_ckpt.pth"; log "resume $name"
  else
    CK="-c $S/mps/yolox_s.pth"; log "train $name"
  fi
  $PY scripts/train_mps.py --yolox-dir $S/YOLOX -f $R/exps/$name.py -b 16 $CK --cache ram \
    -expn $name --output-dir "$OUT" >> "$OUT/train_$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$D/best_ckpt.pth" ]; then log "train $name FAILED rc=$rc"; exit 1; fi

  log "eval $name"
  for s in val test; do
    $PY scripts/eval_pr_mps.py $R/eval_pr.py -f $R/exps/$name.py -c "$D/best_ckpt.pth" \
      --data-dir "$YOLOX_DATA_DIR" --split $s --conf 0.3 --nms 0.45 --device mps \
      --out "$OUT/results/${name}_$s" --save-preds "$OUT/results/${name}_${s}_preds.json" >> "$LOG" 2>&1 \
      || log "eval $name $s FAILED"
  done
  touch "$D/DONE"
  log "done $name"
done
log "ALLDONE"
