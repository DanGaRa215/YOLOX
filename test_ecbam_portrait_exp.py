"""exps/yolox_s_windfarm_ecbam_portrait.py（施策 D+E）のテスト。

  * get_exp で読み込める / exp_name が自分のファイル名になる
  * get_model() が E-CBAM 付き（ベースラインより追加パラメータが 312 多い）
  * CPU で 1x3x640x640 のダミー入力の順伝播が通る
  * データセット構築と repeat（datasets/windfarm があるときだけ。無ければ skip）
yolox が import できない環境では全体を skip する。
使い方: PYTHONPATH=<YOLOX の clone> python test_ecbam_portrait_exp.py
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPS = ROOT / "exps"
EXP_FILE = EXPS / "yolox_s_windfarm_ecbam_portrait.py"
DATA_DIR = Path(os.environ.get("YOLOX_DATA_DIR", ROOT / "datasets" / "windfarm"))

try:
    import torch
    import yolox  # noqa: F401
    HAVE_YOLOX = True
except Exception:
    HAVE_YOLOX = False

if HAVE_YOLOX:
    sys.path.insert(0, str(EXPS))
    os.environ["YOLOX_DATA_DIR"] = str(DATA_DIR)


@unittest.skipUnless(HAVE_YOLOX, "yolox / torch が import できないため skip")
class EcbamPortraitExpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from yolox.exp import get_exp
        cls.exp = get_exp(str(EXP_FILE), None)

    def test_load(self):
        self.assertEqual(self.exp.exp_name, "yolox_s_windfarm_ecbam_portrait")
        # ベースラインのスケジュールをそのまま使う
        self.assertEqual((self.exp.max_epoch, self.exp.no_aug_epochs), (50, 15))
        self.assertEqual(self.exp.num_classes, 2)

    def test_model_has_ecbam(self):
        from yolox.exp import get_exp
        from yolox_s_windfarm_ecbam import ECBAMPAFPN
        base = get_exp(str(EXPS / "yolox_s_windfarm.py"), None).get_model()
        model = self.exp.get_model()
        self.assertIsInstance(model.backbone, ECBAMPAFPN)
        n_base = sum(p.numel() for p in base.parameters())
        n_new = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_new - n_base, 312)
        for name in ("attn_dark3", "attn_dark4", "attn_dark5"):
            self.assertTrue(hasattr(model.backbone, name))

    def test_forward_cpu(self):
        model = self.exp.get_model().eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 640, 640))
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 7)  # 4 box + obj + 2 クラス


@unittest.skipUnless(HAVE_YOLOX, "yolox / torch が import できないため skip")
@unittest.skipUnless((DATA_DIR / "annotations" / "instances_train2017.json").exists(),
                     "datasets/windfarm が無いため skip")
class PortraitRepeatTest(unittest.TestCase):
    def test_repeat(self):
        from yolox.exp import get_exp
        from yolox_s_windfarm_portrait import Exp as PortraitExp
        exp = get_exp(str(EXP_FILE), None)
        self.assertIsInstance(exp, PortraitExp)
        base = exp.get_dataset(cache=False)
        idx_map = exp._build_index_map(base)
        n, n_pos, f, r, rep, n_virt = exp.portrait_stats
        self.assertEqual(n, len(base.annotations))
        self.assertEqual(n_virt, len(idx_map))
        self.assertEqual(n_virt, n + n_pos * (rep - 1))
        self.assertGreaterEqual(rep, 1)
        # 縦長かつ cable tower の画像は rep 回、それ以外は 1 回出る
        counts = {}
        for i in idx_map:
            counts[i] = counts.get(i, 0) + 1
        self.assertEqual(set(counts.values()), {1, rep})
        self.assertEqual(sum(1 for c in counts.values() if c == rep), n_pos)
        print(f"[portrait] 対象 {n_pos}/{n} 枚, r={r:.3f} -> {rep} 回, 仮想エポック {n_virt}")


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=1).result
    if r.wasSuccessful():
        print("ok")
    sys.exit(0 if r.wasSuccessful() else 1)
