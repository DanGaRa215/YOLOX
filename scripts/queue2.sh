#!/bin/bash
# Trains + evaluates C, B, A in sequence (cudnn.benchmark off via run_nobench.py, default 4 workers).
cd /content/YOLOX
export PYTHONPATH=/content/YOLOX YOLOX_DATA_DIR=/content/datasets/windfarm TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
OUT=/content/drive/MyDrive/windfarm_outputs
mkdir -p results "$OUT/results"
for name in yolox_s_windfarm_noaug12 yolox_s_windfarm_clsweight yolox_s_windfarm_oversample; do
  echo "[$(date +%T)] train $name" >> /content/queue.log
  rm -rf YOLOX_outputs/$name
  python run_nobench.py tools/train.py -f exps/$name.py -c yolox_s.pth -d 1 -b 16 --fp16 -o --cache ram \
    > /content/train_$name.log 2>&1
  cp /content/train_$name.log "$OUT/"
  echo "[$(date +%T)] eval $name" >> /content/queue.log
  ck=YOLOX_outputs/$name/best_ckpt.pth
  for s in val test; do
    python eval_pr.py -f exps/$name.py -c $ck --data-dir $YOLOX_DATA_DIR --split $s \
      --conf 0.3 --nms 0.45 --out results/${name}_$s --fp16 >> /content/queue.log 2>&1
    EVAL_SPLIT=$s python tools/eval.py -f exps/$name.py -c $ck -b 16 -d 1 --conf 0.001 --fp16 --fuse \
      > results/${name}_${s}_cocomap.txt 2>&1
  done
  cp results/* "$OUT/results/"
  cp -r YOLOX_outputs/$name "$OUT/" 2>/dev/null
done
echo "[$(date +%T)] ALLDONE" >> /content/queue.log
cp /content/queue.log "$OUT/"
