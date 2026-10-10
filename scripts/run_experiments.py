#!/usr/bin/env python3
"""実験リスト（JSON）の学習 → 評価を順に実行する。Python だけで書いてあり、Windows でも動く（bash 不要）。

1 実験ごとに、queue_e50.sh と同じ引数で
  学習:  run_nobench.py 経由で YOLOX の tools/train.py（-d 1 -b 16 --fp16 --cache ram、事前学習 yolox_s.pth）
  評価:  val / test それぞれに eval_pr.py（--save-preds 付き）と tools/eval.py（COCO mAP）
を実行し、最後に DONE を作る。出力は <out-dir>/<name>/（学習結果）と <out-dir>/results/（評価結果）。

再開:
  * <out-dir>/<name>/DONE がある実験は飛ばす。
  * 学習が終わっていて（TRAIN_DONE）評価だけ残っているなら、評価だけやり直す。
  * <out-dir>/<name>/latest_ckpt.pth があれば --resume で続きから学習する。

seed と opts は YOLOX の train.py の末尾の opts（exp.merge）で渡す。型の変換について:
  * input_size / test_size: 既定値がタプルなので "(800,800)" から (800, 800) に変換される
  * max_epoch: 既定値が int なので int に変換される
  * seed: 既定値が None のため文字列のまま入る。run_nobench.py が merge の後に int へ直す
入力サイズを変えた実験は、評価でも同じ test_size を eval_pr.py / tools/eval.py に渡す。

使い方:
  python scripts/run_experiments.py --dry-run                      # コマンドだけ表示
  python scripts/run_experiments.py --only yolox_s_windfarm_s2     # 1 本だけ
  python scripts/run_experiments.py                                # 全部（再実行すると続きから）
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LIST = Path(__file__).resolve().parent / "experiments_local.json"
# 評価にも引き継ぐ opts のキー（入力サイズを変えた実験で、評価の test_size を学習と揃える）
EVAL_OPT_KEYS = ("test_size",)


def load_experiments(path):
    """JSON を読んで検証する。各要素: name, exp, seed（null 可）, opts（辞書、省略可）。"""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    names = set()
    for e in items:
        for k in ("name", "exp"):
            if not isinstance(e.get(k), str) or not e[k]:
                raise ValueError(f"実験の {k} がありません: {e}")
        if e["name"] in names:
            raise ValueError(f"name が重複しています: {e['name']}")
        names.add(e["name"])
        seed = e.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed は整数か null です: {e}")
        e.setdefault("seed", None)
        e["opts"] = e.get("opts") or {}
    return items


def opts_to_list(opts):
    """{"max_epoch": 100} -> ["max_epoch", "100"]（train.py / eval.py の末尾 opts の形式）。"""
    out = []
    for k, v in opts.items():
        out += [str(k), str(v)]
    return out


def exp_dir(args, name):
    return Path(args.out_dir) / name


def state(args, name):
    """出力先の状態: done / eval_only（学習済み） / resume / fresh。"""
    d = exp_dir(args, name)
    if (d / "DONE").exists():
        return "done"
    if (d / "TRAIN_DONE").exists():
        return "eval_only"
    if (d / "latest_ckpt.pth").exists():
        return "resume"
    return "fresh"


def build_env(args):
    env = dict(os.environ)
    py_path = [str(Path(args.yolox_dir))]
    if env.get("PYTHONPATH"):
        py_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path)
    env["YOLOX_DATA_DIR"] = str(Path(args.data_dir))
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def build_train_cmd(e, args, resume):
    d = exp_dir(args, e["name"])
    cmd = [sys.executable, str(REPO / "scripts" / "run_nobench.py"), str(Path(args.yolox_dir) / "tools" / "train.py"),
           "-f", str(REPO / "exps" / (e["exp"] + ".py")), "-expn", e["name"],
           "-d", "1", "-b", "16", "--fp16", "--cache", "ram"]
    if args.occupy:
        cmd.append("-o")
    if resume:
        cmd += ["--resume", "-c", str(d / "latest_ckpt.pth")]
    else:
        cmd += ["-c", str(args.pretrained)]
    # 以降は末尾の opts（exp.merge）。output_dir を変えると出力先が <out-dir>/<name>/ になる
    cmd += ["output_dir", str(Path(args.out_dir))]
    if e["seed"] is not None:
        cmd += ["seed", str(e["seed"])]
    cmd += opts_to_list(e["opts"])
    return cmd


def build_eval_steps(e, args):
    """評価のコマンド列。各要素は dict(cmd, env_extra, stdout_to)。stdout_to があればそのファイルへ出力する。"""
    name = e["name"]
    exp_file = str(REPO / "exps" / (e["exp"] + ".py"))
    ckpt = str(exp_dir(args, name) / "best_ckpt.pth")
    res = Path(args.out_dir) / "results"
    eval_opts = opts_to_list({k: v for k, v in e["opts"].items() if k in EVAL_OPT_KEYS})
    steps = []
    for s in ("val", "test"):
        steps.append({
            "cmd": [sys.executable, str(REPO / "eval_pr.py"), "-f", exp_file, "-c", ckpt,
                    "--data-dir", str(Path(args.data_dir)), "--split", s, "--conf", "0.3", "--nms", "0.45",
                    "--out", str(res / f"{name}_{s}"), "--fp16",
                    "--save-preds", str(res / f"{name}_{s}_preds.json")] + eval_opts,
            "env_extra": {}, "stdout_to": None})
        steps.append({
            "cmd": [sys.executable, str(Path(args.yolox_dir) / "tools" / "eval.py"), "-f", exp_file, "-c", ckpt,
                    "-b", "16", "-d", "1", "--conf", "0.001", "--fp16", "--fuse"] + eval_opts,
            "env_extra": {"EVAL_SPLIT": s}, "stdout_to": res / f"{name}_{s}_cocomap.txt"})
    return steps


class Logger:
    """標準出力とログファイルの両方に書く。"""

    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)

    def log(self, msg):
        self.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def fmt_cmd(cmd):
    return " ".join(shlex.quote(c) for c in cmd)


def run_cmd(cmd, env, cwd, logger, stdout_to=None):
    """コマンドを実行し、出力を標準出力・ログ・（あれば）stdout_to に流す。戻り値は終了コード。"""
    sink = None
    if stdout_to is not None:
        Path(stdout_to).parent.mkdir(parents=True, exist_ok=True)
        sink = open(stdout_to, "w", encoding="utf-8")
    try:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            logger.write(line)
            if sink:
                sink.write(line)
        return p.wait()
    finally:
        if sink:
            sink.close()


def run_one(e, args, logger):
    """1 実験を実行する。成功（または DONE で skip）なら True。"""
    name, st = e["name"], state(args, e["name"])
    d = exp_dir(args, name)
    if st == "done":
        logger.log(f"skip {name} (DONE)")
        return True
    env, cwd = build_env(args), str(Path(args.yolox_dir))
    if st in ("fresh", "resume"):
        cmd = build_train_cmd(e, args, resume=(st == "resume"))
        logger.log(f"{'resume' if st == 'resume' else 'train'} {name}")
        logger.log("  " + fmt_cmd(cmd))
        d.mkdir(parents=True, exist_ok=True)
        rc = run_cmd(cmd, env, cwd, Logger(d / "train_stdout.log"))
        if rc != 0:
            logger.log(f"train {name} FAILED rc={rc}")
            return False
        (d / "TRAIN_DONE").touch()
    logger.log(f"eval {name}")
    for step in build_eval_steps(e, args):
        logger.log("  " + fmt_cmd(step["cmd"]))
        rc = run_cmd(step["cmd"], dict(env, **step["env_extra"]), cwd, logger, step["stdout_to"])
        if rc != 0 and step["stdout_to"] is None:  # eval_pr の失敗は致命的。tools/eval.py（参考値）は続行
            logger.log(f"eval {name} FAILED rc={rc}")
            return False
    (d / "DONE").touch()
    logger.log(f"done {name}")
    return True


def dry_run(e, args, logger):
    st = state(args, e["name"])
    logger.log(f"[dry-run] {e['name']}: 状態={st}")
    if st == "done":
        logger.write("  (DONE があるため飛ばす)\n")
        return
    if st in ("fresh", "resume"):
        logger.write("  train: " + fmt_cmd(build_train_cmd(e, args, resume=(st == "resume"))) + "\n")
    for step in build_eval_steps(e, args):
        env = "".join(f"{k}={v} " for k, v in step["env_extra"].items())
        redirect = f" > {step['stdout_to']}" if step["stdout_to"] else ""
        logger.write(f"  eval: {env}{fmt_cmd(step['cmd'])}{redirect}\n")


def make_parser():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default=str(DEFAULT_LIST), help="実験リスト JSON（既定 scripts/experiments_local.json）")
    ap.add_argument("--yolox-dir", default=env("YOLOX_DIR", "YOLOX"), help="YOLOX の clone（環境変数 YOLOX_DIR、既定 ./YOLOX）")
    ap.add_argument("--data-dir", default=env("YOLOX_DATA_DIR", "datasets/windfarm"),
                    help="データセットのルート（環境変数 YOLOX_DATA_DIR、既定 datasets/windfarm）")
    ap.add_argument("--out-dir", default=env("OUT_DIR", "outputs_local"),
                    help="出力先（環境変数 OUT_DIR、既定 outputs_local）。<name>/ と results/ ができる")
    ap.add_argument("--pretrained", default=env("PRETRAINED"), help="事前学習 yolox_s.pth（環境変数 PRETRAINED、既定 <yolox-dir>/yolox_s.pth）")
    ap.add_argument("--occupy", action="store_true", help="学習に -o（GPU メモリの先取り）を付ける。既定はオフ")
    ap.add_argument("--only", metavar="NAME", help="この name の 1 本だけ実行する")
    ap.add_argument("--dry-run", action="store_true", help="実行するコマンドを表示するだけで、何も実行しない")
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    # 学習・評価は cwd を YOLOX の clone にして実行するため、相対パスは先に絶対パスへ直しておく
    for k in ("yolox_dir", "data_dir", "out_dir", "experiments"):
        setattr(args, k, str(Path(getattr(args, k)).resolve()))
    args.pretrained = str(Path(args.pretrained).resolve() if args.pretrained else Path(args.yolox_dir) / "yolox_s.pth")
    items = load_experiments(args.experiments)
    if args.only:
        items = [e for e in items if e["name"] == args.only]
        if not items:
            print(f"--only {args.only}: 実験リストにありません", file=sys.stderr)
            return 2
    if args.dry_run:
        logger = Logger(None)
        for e in items:
            dry_run(e, args, logger)
        return 0
    missing = [p for p in (Path(args.yolox_dir) / "tools" / "train.py", Path(args.pretrained),
                           Path(args.data_dir) / "annotations") if not p.exists()]
    if missing:
        print("見つかりません: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 2
    logger = Logger(Path(args.out_dir) / "run_experiments.log")
    failed = []
    for e in items:
        if not run_one(e, args, logger):
            failed.append(e["name"])  # 失敗しても次の実験へ進む
    logger.log("ALLDONE" if not failed else "失敗: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
