#!/bin/bash
# ベースラインの学習完了を待って評価し、続けて cudnn.benchmark の on/off で所要時間を計測する（同じ exp、ワーカー 2）。
cd /content/YOLOX
export PYTHONPATH=/content/YOLOX YOLOX_DATA_DIR=/content/datasets/windfarm TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
OUT=/content/drive/MyDrive/windfarm_outputs
mkdir -p results "$OUT/results"
log() { echo "[$(date +%T)] $*" >> /content/bench.log; }

while pgrep -f "exps/yolox_s_windfarm.py " > /dev/null; do sleep 20; done
cp /content/train_baseline.log "$OUT/"
log "baseline finished; eval"
ck=YOLOX_outputs/yolox_s_windfarm/best_ckpt.pth
for s in val test; do
  python eval_pr.py -f exps/yolox_s_windfarm.py -c $ck --data-dir $YOLOX_DATA_DIR --split $s \
    --conf 0.3 --nms 0.45 --out results/yolox_s_windfarm_$s --fp16 >> /content/bench.log 2>&1
  EVAL_SPLIT=$s python tools/eval.py -f exps/yolox_s_windfarm.py -c $ck -b 16 -d 1 --conf 0.001 --fp16 --fuse \
    > results/yolox_s_windfarm_${s}_cocomap.txt 2>&1
done
cp results/* "$OUT/results/"; cp -r YOLOX_outputs/yolox_s_windfarm "$OUT/"
log "baseline eval done"

ARGS="-f exps/yolox_s_windfarm_noaug12.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram print_interval 10"
for mode in on off; do
  rm -rf YOLOX_outputs/yolox_s_windfarm_noaug12
  log "bench $mode start"
  if [ $mode = on ]; then
    timeout 330 python tools/train.py $ARGS > /content/bench_$mode.log 2>&1
  else
    timeout 330 python run_nobench.py tools/train.py $ARGS > /content/bench_$mode.log 2>&1
  fi
  sleep 5
done
rm -rf YOLOX_outputs/yolox_s_windfarm_noaug12
for mode in on off; do
  echo "== $mode ==" >> /content/bench.log
  grep -oE "iter: [0-9]+/166.*iter_time: [0-9.]+s, data_time: [0-9.]+s.*size: [0-9]+" /content/bench_$mode.log \
    | sed -E 's/gpu mem.*iter_time/iter_time/; s/total_loss.*size/size/' >> /content/bench.log
done
log "BENCHDONE"
cp /content/bench*.log "$OUT/"
