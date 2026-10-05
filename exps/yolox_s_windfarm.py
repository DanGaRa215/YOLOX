#!/usr/bin/env python3
# YOLOX-s baseline for Wind Farms (cable tower / turbine). Defaults kept except dataset/epochs.
#
# Env vars:
#   YOLOX_DATA_DIR  dataset root (contains train2017/ val2017/ test2017/ annotations/); default datasets/windfarm
#   EVAL_SPLIT      "val" (default) or "test"  -> chooses eval images dir + annotation file
#
# Train:  python tools/train.py -f exps/yolox_s_windfarm.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
# Eval on test: EVAL_SPLIT=test python tools/eval.py -f exps/yolox_s_windfarm.py -c <ckpt> -b 16 -d 1 --conf 0.001
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # yolox-s
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 2
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

        # data
        self.data_dir = os.environ.get("YOLOX_DATA_DIR", "datasets/windfarm")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.eval_split = os.environ.get("EVAL_SPLIT", "val")
        if self.eval_split not in ("val", "test"):
            raise ValueError("EVAL_SPLIT must be val or test")

        # schedule (baseline)
        self.max_epoch = 25
        self.no_aug_epochs = 8
        self.eval_interval = 5
        self.print_interval = 50

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        legacy = kwargs.get("legacy", False)
        if self.eval_split == "test":
            ann, name = self.test_ann, "test2017"
        else:
            ann, name = self.val_ann, "val2017"
        return COCODataset(
            data_dir=self.data_dir,
            json_file=ann,
            name=name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
