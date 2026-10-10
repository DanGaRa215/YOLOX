"""scripts/run_experiments.py のテスト。実際の学習は走らせない（dry-run と純粋関数だけ）。

  * 実験リスト JSON（scripts/experiments_local.json）の中身（9 本、名前の規則）
  * seed / opts がコマンドの末尾に渡ること、seed が null なら付かないこと
  * 状態（fresh / resume / eval_only / done）による resume・skip の切り替え
  * 入力サイズを変えた実験で、評価にも test_size が渡ること
  * yolox が import できるときだけ: 組み立てた末尾 opts が train.py のパーサと exp.merge で正しい型になること
"""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as R  # noqa: E402

try:
    import torch  # noqa: F401
    import yolox
    HAVE_YOLOX = True
except Exception:
    HAVE_YOLOX = False


def make_args(out):
    return R.make_parser().parse_args(["--yolox-dir", str(out / "YOLOX"), "--data-dir", str(out / "data"),
                                       "--out-dir", str(out / "out"), "--pretrained", str(out / "yolox_s.pth")])


def dry_run_output(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(argv + ["--dry-run"])
    assert rc == 0
    return buf.getvalue()


class ListTest(unittest.TestCase):
    def test_local_list(self):
        items = R.load_experiments(R.DEFAULT_LIST)
        self.assertEqual(len(items), 9)
        by = {e["name"]: e for e in items}
        for base in ("yolox_s_windfarm", "yolox_s_windfarm_ecbam", "yolox_s_windfarm_portrait"):
            for s in (2, 3):
                self.assertEqual(by[f"{base}_s{s}"]["exp"], base)
                self.assertEqual(by[f"{base}_s{s}"]["seed"], s)
        self.assertEqual(by["yolox_s_windfarm_ecbam_portrait_s1"]["exp"], "yolox_s_windfarm_ecbam_portrait")
        self.assertEqual(by["yolox_s_windfarm_ecbam_in800_s1"]["opts"],
                         {"input_size": "(800,800)", "test_size": "(800,800)"})
        self.assertEqual(by["yolox_s_windfarm_ecbam_e100_s1"]["opts"], {"max_epoch": 100})
        for e in items:  # exp ファイルが実在する
            self.assertTrue((ROOT / "exps" / (e["exp"] + ".py")).exists(), e)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.json"
            p.write_text(json.dumps([{"name": "a", "exp": "e"}, {"name": "a", "exp": "e"}]))
            with self.assertRaises(ValueError):
                R.load_experiments(p)
            p.write_text(json.dumps([{"name": "a", "exp": "e", "seed": "2"}]))
            with self.assertRaises(ValueError):
                R.load_experiments(p)
            p.write_text(json.dumps([{"name": "a", "exp": "e"}]))
            e = R.load_experiments(p)[0]
            self.assertEqual((e["seed"], e["opts"]), (None, {}))


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.args = make_args(self.out)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_fresh_with_seed_and_opts(self):
        e = {"name": "n_in800_s1", "exp": "yolox_s_windfarm_ecbam", "seed": 1,
             "opts": {"input_size": "(800,800)", "max_epoch": 100}}
        cmd = R.build_train_cmd(e, self.args, resume=False)
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith("run_nobench.py") and cmd[2].endswith("train.py"))
        self.assertEqual(cmd[cmd.index("-expn") + 1], "n_in800_s1")
        for flag in ("--fp16", "-d", "-b"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--cache") + 1], "ram")
        self.assertNotIn("-o", cmd)          # 既定ではオフ
        self.assertNotIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], self.args.pretrained)
        i = cmd.index("output_dir")           # ここから末尾が opts
        self.assertEqual(cmd[i:], ["output_dir", str(self.out / "out"), "seed", "1",
                                   "input_size", "(800,800)", "max_epoch", "100"])

    def test_no_seed_and_occupy(self):
        e = {"name": "n", "exp": "yolox_s_windfarm", "seed": None, "opts": {}}
        args = R.make_parser().parse_args(["--occupy", "--out-dir", str(self.out)])
        cmd = R.build_train_cmd(e, args, resume=False)
        self.assertIn("-o", cmd)
        self.assertNotIn("seed", cmd)

    def test_resume(self):
        e = {"name": "n", "exp": "yolox_s_windfarm", "seed": 2, "opts": {}}
        cmd = R.build_train_cmd(e, self.args, resume=True)
        self.assertIn("--resume", cmd)
        ck = cmd[cmd.index("-c") + 1]
        self.assertEqual(Path(ck), self.out / "out" / "n" / "latest_ckpt.pth")
        self.assertNotIn(self.args.pretrained, cmd)

    def test_eval_passes_test_size_only(self):
        e = {"name": "n", "exp": "yolox_s_windfarm_ecbam", "seed": 1,
             "opts": {"input_size": "(800,800)", "test_size": "(800,800)", "max_epoch": 100}}
        steps = R.build_eval_steps(e, self.args)
        self.assertEqual(len(steps), 4)
        for s in steps:
            self.assertEqual(s["cmd"][-2:], ["test_size", "(800,800)"])
            self.assertNotIn("max_epoch", s["cmd"])
        pr = [s for s in steps if s["cmd"][1].endswith("eval_pr.py")]
        self.assertEqual(len(pr), 2)
        self.assertIn("--save-preds", pr[0]["cmd"])
        self.assertEqual([s["env_extra"]["EVAL_SPLIT"] for s in steps if s["stdout_to"]], ["val", "test"])
        # opts が無い実験では何も付かない
        plain = R.build_eval_steps({"name": "n", "exp": "x", "seed": None, "opts": {}}, self.args)
        self.assertNotIn("test_size", plain[0]["cmd"])


class StateTest(unittest.TestCase):
    def test_dry_run_states(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            base = ["--out-dir", str(out), "--yolox-dir", str(out / "Y")]
            name = "yolox_s_windfarm_s2"
            text = dry_run_output(base + ["--only", name])
            self.assertIn("状態=fresh", text)
            self.assertIn("train:", text)
            self.assertNotIn("--resume", text)
            self.assertIn(" seed 2", text)
            self.assertEqual(text.count("eval:"), 4)

            d = out / name
            d.mkdir()
            (d / "latest_ckpt.pth").write_text("x")
            text = dry_run_output(base + ["--only", name])
            self.assertIn("状態=resume", text)
            self.assertIn("--resume", text)

            (d / "TRAIN_DONE").write_text("")
            text = dry_run_output(base + ["--only", name])
            self.assertIn("状態=eval_only", text)
            self.assertNotIn("train:", text)
            self.assertEqual(text.count("eval:"), 4)

            (d / "DONE").write_text("")
            text = dry_run_output(base + ["--only", name])
            self.assertIn("状態=done", text)
            self.assertNotIn("train:", text)
            self.assertNotIn("eval:", text)

    def test_only_and_all(self):
        with tempfile.TemporaryDirectory() as t:
            base = ["--out-dir", t]
            self.assertEqual(dry_run_output(base).count("[dry-run]"), 9)
            self.assertEqual(dry_run_output(base + ["--only", "yolox_s_windfarm_ecbam_e100_s1"]).count("[dry-run]"), 1)
            self.assertIn("max_epoch 100", dry_run_output(base + ["--only", "yolox_s_windfarm_ecbam_e100_s1"]))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(R.main(base + ["--only", "nonexistent", "--dry-run"]), 2)


@unittest.skipUnless(HAVE_YOLOX, "yolox / torch が import できないため skip")
class YoloxMergeTest(unittest.TestCase):
    """組み立てた末尾 opts が、実際の YOLOX の train.py のパーサと exp.merge で正しい型になるか。"""

    def test_opts_types(self):
        train_py = Path(yolox.__file__).resolve().parent.parent / "tools" / "train.py"
        if not train_py.exists():
            self.skipTest("tools/train.py が見つからない")
        try:
            import torch.utils.tensorboard  # noqa: F401
        except ImportError:  # tensorboard が無い環境では、パーサの確認のためだけに空の代用を入れる
            import types
            fake = types.ModuleType("torch.utils.tensorboard")
            fake.SummaryWriter = object
            sys.modules["torch.utils.tensorboard"] = fake
        spec = importlib.util.spec_from_file_location("yolox_train_tool", train_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as t:
            args = make_args(Path(t))
            e = {"name": "n", "exp": "yolox_s_windfarm_ecbam", "seed": 3,
                 "opts": {"input_size": "(800,800)", "test_size": "(800,800)", "max_epoch": 100}}
            cmd = R.build_train_cmd(e, args, resume=True)
            parsed = mod.make_parser().parse_args(cmd[3:])  # run_nobench.py と train.py の後ろ
            self.assertEqual((parsed.experiment_name, parsed.batch_size, parsed.devices, parsed.resume, parsed.cache),
                             ("n", 16, 1, True, "ram"))
            # run_nobench.py 経由で merge して型を確かめる（seed は int に直される）
            script = Path(t) / "chk.py"
            script.write_text(
                "import sys, json\n"
                f"sys.path.insert(0, {str(ROOT / 'exps')!r})\n"
                "from yolox.exp import get_exp\n"
                f"e = get_exp({str(ROOT / 'exps' / 'yolox_s_windfarm_ecbam.py')!r}, None)\n"
                f"e.merge({parsed.opts!r})\n"
                "print(json.dumps([e.seed, list(e.input_size), list(e.test_size), e.max_epoch, e.output_dir]))\n")
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_nobench.py"), str(script)],
                                 capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(out.returncode, 0, out.stderr)
            got = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(got, [3, [800, 800], [800, 800], 100, str(Path(t) / "out")])
            self.assertIs(type(got[0]), int)


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=1).result
    if r.wasSuccessful():
        print("ok")
    sys.exit(0 if r.wasSuccessful() else 1)
