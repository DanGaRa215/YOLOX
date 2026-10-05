# YOLOX-s wind farm detection (cable tower / turbine)

Fine-tuning experiments of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) on the Roboflow
"Wind Farms" v5 dataset (CC BY 4.0, by Kyle Graupe), comparing per-class Precision / Recall.

## Data
- 2 classes: `cable tower` (441 train instances) / `turbine` (15,534) — about 35:1 imbalance.
- train 2643 / valid 247 / test 130 images. The dataset is not included in this repository.
- `convert_yolo_to_coco.py` converts the Roboflow YOLO txt export to the COCO layout YOLOX reads;
  `verify_coco.py` checks counts and draws boxes.

## Experiments (all 25 epochs, YOLOX-s, COCO-pretrained `yolox_s.pth`)
| Exp file | Change vs. baseline |
|---|---|
| `exps/yolox_s_windfarm.py` | baseline (`no_aug_epochs=8`) |
| `exps/yolox_s_windfarm_noaug12.py` | (C) longer close-mosaic phase: `no_aug_epochs=12` |
| `exps/yolox_s_windfarm_clsweight.py` | (B) class-weighted cls loss (`CLS_WEIGHTS`, default `4,1`) |
| `exps/yolox_s_windfarm_oversample.py` | (A) oversample cable-tower images (`OVERSAMPLE_K`, default 3) |

## Evaluation
`eval_pr.py` reports per-class and micro Precision / Recall at IoU 0.5, conf 0.3, NMS 0.45
(`test_eval_pr.py` unit-tests the matching). COCO mAP comes from YOLOX `tools/eval.py`.

## Running on Colab
See `colab_setup.md`. `scripts/` holds the helpers actually used on Colab:
`run_nobench.py` (runs a YOLOX tool with `cudnn.benchmark` off), `queue2.sh` (trains + evaluates the
three variants in sequence), `bench.sh` (cudnn.benchmark on/off timing).
