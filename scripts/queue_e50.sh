#!/bin/bash
# 50 エポックでベースライン → C → B → A を順に学習・評価する。
# ランタイムが切れても続きから再開できるよう、チェックポイントを 10 分ごとに Drive に保存し、
# 再実行時は Drive の latest_ckpt.pth から --resume する。完了した実験は DONE ファイルで飛ばす。
cd /content/YOLOX
export PYTHONPATH=/content/YOLOX YOLOX_DATA_DIR=/content/datasets/windfarm TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
OUT=/content/drive/MyDrive/windfarm_outputs_e50
mkdir -p results "$OUT/results"
log() { echo "[$(date +%T)] $*" >> /content/queue_e50.log; cp /content/queue_e50.log "$OUT/" 2>/dev/null; }

for name in yolox_s_windfarm yolox_s_windfarm_noaug12 yolox_s_windfarm_clsweight yolox_s_windfarm_oversample yolox_s_windfarm_portrait yolox_s_windfarm_ecbam; do
  D="$OUT/$name"
  mkdir -p "$D"
  if [ -f "$D/DONE" ]; then log "skip $name (DONE)"; continue; fi

  if [ -f "$D/latest_ckpt.pth" ]; then
    mkdir -p YOLOX_outputs/$name
    cp "$D"/*.pth YOLOX_outputs/$name/ 2>/dev/null
    CK="--resume -c YOLOX_outputs/$name/latest_ckpt.pth"
    log "resume $name"
  else
    rm -rf YOLOX_outputs/$name
    CK="-c yolox_s.pth"
    log "train $name"
  fi

  # 学習中のチェックポイントとログを 10 分ごとに Drive へ
  ( while true; do sleep 600
      cp YOLOX_outputs/$name/latest_ckpt.pth YOLOX_outputs/$name/best_ckpt.pth YOLOX_outputs/$name/train_log.txt "$D/" 2>/dev/null
    done ) &
  SYNC=$!

  python run_nobench.py tools/train.py -f exps/$name.py $CK -d 1 -b 16 --fp16 -o --cache ram \
    >> /content/train_e50_$name.log 2>&1
  RC=$?
  kill $SYNC 2>/dev/null
  cp YOLOX_outputs/$name/*.pth YOLOX_outputs/$name/train_log.txt "$D/" 2>/dev/null
  cp /content/train_e50_$name.log "$D/"
  if [ $RC -ne 0 ]; then log "train $name FAILED rc=$RC"; exit 1; fi

  log "eval $name"
  for s in val test; do
    python eval_pr.py -f exps/$name.py -c YOLOX_outputs/$name/best_ckpt.pth --data-dir $YOLOX_DATA_DIR --split $s \
      --conf 0.3 --nms 0.45 --out results/${name}_$s --fp16 --save-preds results/${name}_${s}_preds.json \
      >> /content/queue_e50.log 2>&1
    EVAL_SPLIT=$s python tools/eval.py -f exps/$name.py -c YOLOX_outputs/$name/best_ckpt.pth -b 16 -d 1 \
      --conf 0.001 --fp16 --fuse > results/${name}_${s}_cocomap.txt 2>&1
  done
  cp results/${name}_* "$OUT/results/"
  touch "$D/DONE"
  log "done $name"
done
log "ALLDONE"
