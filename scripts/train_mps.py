#!/usr/bin/env python3
"""YOLOX を Apple Silicon (MPS) で学習するランナー。YOLOX 本体のファイルは一切変更しない。

tools/train.py と同じ引数（-f, -b, -c, --resume, --cache, -expn, -e, opts）を受け付け、
CUDA 前提の箇所を実行時にパッチ（モンキーパッチ / Trainer のサブクラス）して MPS で回す。

パッチする箇所:
  * Trainer（サブクラス MPSTrainer）: torch.cuda.set_device / .cuda() / GradScaler / autocast /
    DataPrefetcher（cuda stream）/ occupy / DDP を使わない版に置き換え。fp32 のみ（--fp16 は無視して警告）。
  * Exp.random_resize: torch.LongTensor(2).cuda() を CPU テンソルに（単一プロセスなので同期は不要）。
  * COCOEvaluator.evaluate: torch.cuda.FloatTensor / HalfTensor を使う箇所を、ソースを書き換えて MPS 向けにする。
  * yolox.core.trainer.gpu_mem_usage: torch.cuda.max_memory_allocated の代わりに MPS の確保量を返す。
  * Tensor.type(): 'torch.mps.FloatTensor' 文字列を再入力できない問題を、dtype に直すラッパーで回避。
  * COCOeval_opt（ninja で C++ をビルドする高速版）を外し、pycocotools 標準の COCOeval を使わせる。
  * torch.load の既定を weights_only=False に戻す（--resume で自前のチェックポイントを読むため）。
  * tensorboard が未インストールなら SummaryWriter を何もしないダミーにする。
  * launch（nccl / cuda 必須）は呼ばず、main を直接呼ぶ。cudnn の設定も触らない。
  * DataLoader の子プロセスは fork を使う（spawn だと --cache ram のキャッシュを子ごとに pickle してメモリが溢れるため）。
  * PYTORCH_ENABLE_MPS_FALLBACK=1（MPS 未対応の演算は CPU にフォールバック）。torch を import する前に設定する。

使い方（YOLOX の tools/train.py と同じ）:
  python scripts/train_mps.py -f <exp.py> -b 16 -c yolox_s.pth --cache ram -expn name \
      --yolox-dir <YOLOX の clone> --output-dir <出力先>
追加引数:
  --yolox-dir   YOLOX のソースのルート（既定: 環境変数 YOLOX_DIR、なければ import 済みの yolox を使う）
  --output-dir  YOLOX_outputs の出力先（exp.output_dir を上書き）
  --bench-iters N  先頭 N iter だけ回して計測結果を出して終了（チェックポイントは保存しない）
"""
import os

# torch を import する前に設定する必要がある
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import inspect
import multiprocessing
import sys
import textwrap
import time
import types
import warnings

# --yolox-dir を先に拾って sys.path に入れる（yolox の import より前）
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--yolox-dir", default=os.environ.get("YOLOX_DIR"))
_pre_args, _ = _pre.parse_known_args()
if _pre_args.yolox_dir:
    sys.path.insert(0, os.path.abspath(_pre_args.yolox_dir))

import torch

# tensorboard が無い環境ではダミーの SummaryWriter を差し込む（trainer が import 時に要求する）
try:
    import torch.utils.tensorboard  # noqa: F401
except Exception:
    _tb = types.ModuleType("torch.utils.tensorboard")

    class _DummyWriter:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    _tb.SummaryWriter = _DummyWriter
    sys.modules["torch.utils.tensorboard"] = _tb
    torch.utils.tensorboard = _tb

from loguru import logger

import yolox.core.trainer as trainer_mod
from yolox.core import Trainer
from yolox.data import DataPrefetcher  # noqa: F401  (参照のみ。MPS では使わない)
from yolox.evaluators import coco_evaluator as coco_eval_mod
from yolox.exp import Exp, check_exp_value, get_exp

DEVICE = "mps"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")


def mps_mem_mb():
    """MPS の現在の確保量 (MB)。"""
    return torch.mps.current_allocated_memory() / (1024 * 1024)


def mps_driver_mem_mb():
    """MPS ドライバが確保している総量 (MB)。"""
    return torch.mps.driver_allocated_memory() / (1024 * 1024)


class MPSPrefetcher:
    """DataPrefetcher の代替。cuda stream は使わず、バッチを MPS に転送するだけ。"""

    def __init__(self, loader):
        self.loader = iter(loader)

    def next(self):
        inp, target, _, _ = next(self.loader)
        return inp.to(DEVICE), target.to(DEVICE)


def _random_resize_cpu(self, data_loader, epoch, rank, is_distributed):
    """Exp.random_resize の CPU 版（.cuda() を除いただけ。単一プロセス前提）。"""
    import random

    size_factor = self.input_size[1] * 1.0 / self.input_size[0]
    if not hasattr(self, "random_size"):
        min_size = int(self.input_size[0] / 32) - self.multiscale_range
        max_size = int(self.input_size[0] / 32) + self.multiscale_range
        self.random_size = (min_size, max_size)
    size = random.randint(*self.random_size)
    return (int(32 * size), 32 * int(size * size_factor))


_MPS_TYPE_MAP = {
    "torch.mps.FloatTensor": torch.float32,
    "torch.mps.HalfTensor": torch.float16,
    "torch.mps.DoubleTensor": torch.float64,
    "torch.mps.LongTensor": torch.int64,
    "torch.mps.IntTensor": torch.int32,
    "torch.mps.BoolTensor": torch.bool,
}
_orig_tensor_type = torch.Tensor.type


def _tensor_type_mps(self, dtype=None, *args, **kwargs):
    """Tensor.type() が返す 'torch.mps.FloatTensor' を、Tensor.type() に再度渡すと受け付けられない問題の回避。
    YOLOX の head / 損失は `x.type(y.type())` の形で dtype を受け渡しているため、文字列を (MPS, dtype) への .to() に直す。
    CUDA の型文字列と同じく、デバイスも MPS に移す。"""
    if isinstance(dtype, str) and dtype in _MPS_TYPE_MAP:
        # CUDA の 'torch.cuda.FloatTensor' と同様、デバイスも MPS へ移す（CPU 上で作った grid などを head で使うため）
        return self.to(device=DEVICE, dtype=_MPS_TYPE_MAP[dtype])
    return _orig_tensor_type(self, dtype, *args, **kwargs)


def _patch_evaluator():
    """COCOEvaluator.evaluate 内の cuda 専用の記述を、ソース置換で MPS 向けにする。"""
    src = textwrap.dedent(inspect.getsource(coco_eval_mod.COCOEvaluator.evaluate))
    reps = [
        ("tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor",
         "tensor_type = torch.float16 if half else torch.float32"),
        ("imgs = imgs.type(tensor_type)", "imgs = imgs.to(_DEV, tensor_type)"),
        ("statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])",
         "statistics = torch.tensor([inference_time, nms_time, n_samples], dtype=torch.float32)"),
    ]
    for old, new in reps:
        assert old in src, f"evaluate のソースが想定と違う: {old}"
        src = src.replace(old, new)
    ns = dict(vars(coco_eval_mod))
    ns["_DEV"] = DEVICE
    exec(src, ns)
    coco_eval_mod.COCOEvaluator.evaluate = ns["evaluate"]


class MPSTrainer(Trainer):
    """CUDA 依存を外した Trainer。ログ・チェックポイント・resume・評価は親クラスのものをそのまま使う。"""

    bench_iters = 0

    def __init__(self, exp, args):
        super().__init__(exp, args)
        # 親は GradScaler(enabled=fp16) と cuda デバイス名を設定しているので上書きする
        self.amp_training = False
        self.scaler = None
        self.device = DEVICE
        self.data_type = torch.float32
        self._bench = {"iter": [], "data": [], "size": [], "loss": []}

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        outputs = self.model(inps, targets)  # fp32（autocast なし）
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # MPS は非同期実行なので、時間計測と meter 用の値取り出しの前に同期する
        outputs = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()}
        torch.mps.synchronize()
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        if self.bench_iters:
            b = self._bench
            b["iter"].append(iter_end_time - iter_start_time)
            b["data"].append(data_end_time - iter_start_time)
            b["size"].append(tuple(self.input_size))
            b["loss"].append(float(outputs["total_loss"]))
            if len(b["iter"]) <= 10 or len(b["iter"]) % 10 == 0:
                print(f"[bench] it={len(b['iter'])} size={self.input_size} iter={b['iter'][-1]:.2f}s "
                      f"data={b['data'][-1]:.2f}s loss={b['loss'][-1]:.2f} "
                      f"mps_driver={mps_driver_mem_mb():.0f}MB", flush=True)

    def before_train(self):
        # Trainer.before_train から cuda.set_device / occupy / DDP / DataPrefetcher を除いたもの
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        model = self.exp.get_model()
        from yolox.utils import get_model_info

        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        model = self.resume_train(model)

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=False,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher...")
        self.prefetcher = MPSPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        from yolox.utils import ModelEMA

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=False)

        if self.args.logger == "tensorboard":
            self.tblogger = trainer_mod.SummaryWriter(os.path.join(self.file_name, "tensorboard"))
        logger.info("Training start...")

    def train(self):
        # 親の train() は logger.error("...", e) の書式不一致で本来の例外が読めなくなるため、トレースバックを先に出す
        import traceback

        self.before_train()
        try:
            self.train_in_epoch()
        except _BenchDone:
            raise
        except Exception:
            traceback.print_exc()
            raise
        finally:
            self.after_train()

    def train_in_iter(self):
        # --bench-iters 指定時は先頭 N iter で打ち切る
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
            if self.bench_iters and self.iter + 1 >= self.bench_iters:
                raise _BenchDone()

    def after_train(self):
        if self.bench_iters:
            return
        super().after_train()


class _BenchDone(Exception):
    pass


def _report_bench(tr):
    import statistics as st

    b = tr._bench
    skip = min(5, len(b["iter"]) // 5)  # 立ち上がり（MPS カーネルのコンパイル等）を除外
    by_size = {}
    for t, d, s in zip(b["iter"][skip:], b["data"][skip:], b["size"][skip:]):
        by_size.setdefault(s[0], []).append((t, d))
    print("\n===== MPS bench =====")
    print(f"iters={len(b['iter'])} (最初の {skip} iter は除外して集計)")
    for s in sorted(by_size):
        ts = [x[0] for x in by_size[s]]
        ds = [x[1] for x in by_size[s]]
        print(f"size {s}x{s}: n={len(ts)} iter_time mean={st.mean(ts):.3f}s median={st.median(ts):.3f}s "
              f"data_time mean={st.mean(ds):.3f}s")
    all_t = b["iter"][skip:]
    all_d = b["data"][skip:]
    print(f"全体: iter_time mean={st.mean(all_t):.3f}s data_time mean={st.mean(all_d):.3f}s")
    print(f"loss: first={b['loss'][0]:.3f} last={b['loss'][-1]:.3f} nan={any(l != l for l in b['loss'])}")
    print(f"MPS allocated={mps_mem_mb():.0f}MB driver={mps_driver_mem_mb():.0f}MB")
    import psutil

    print(f"プロセス RSS={psutil.Process().memory_info().rss / 2**30:.2f}GB "
          f"system used={psutil.virtual_memory().used / 2**30:.1f}GB")
    n_iter = tr.max_iter
    est = st.mean(all_t) * n_iter * tr.max_epoch
    print(f"1 epoch={n_iter} iter, {tr.max_epoch} epoch の学習のみの推定: {est / 3600:.2f} h（評価時間は含まない）")


def main():
    from yolox.utils import configure_module

    configure_module()
    sys.path.insert(0, os.path.join(os.path.abspath(_pre_args.yolox_dir or "."), "tools"))
    from train import make_parser  # YOLOX の tools/train.py の引数定義を流用する

    parser = make_parser()
    parser.add_argument("--yolox-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bench-iters", type=int, default=0)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS が使えません")
    if args.fp16:
        logger.warning("--fp16 は MPS では未対応のため無視して fp32 で学習します")
        args.fp16 = False

    # fork を使う（理由は冒頭の docstring）。MPS を初期化する前に設定する
    multiprocessing.set_start_method("fork", force=True)

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)
    if args.output_dir:
        exp.output_dir = args.output_dir
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # インスタンス属性にすると exp の repr（ログ出力）が bound method 経由で再帰するため、クラス側を差し替える
    type(exp).random_resize = _random_resize_cpu
    _patch_evaluator()
    # COCOeval_opt は C++ の JIT ビルド（ninja）が必要。標準の pycocotools COCOeval に落とす
    # （evaluate_prediction は ImportError で標準版にフォールバックする）。評価結果の値は同じ
    import yolox.layers as _layers

    if hasattr(_layers, "COCOeval_opt"):
        del _layers.COCOeval_opt
    torch.Tensor.type = _tensor_type_mps
    # torch>=2.6 の torch.load は weights_only=True が既定で、numpy スカラー（curr_ap）を含む自前のチェックポイントを
    # --resume で読めない。自分で保存したチェックポイントだけを読むので False を既定に戻す
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{"weights_only": False, **k})
    trainer_mod.gpu_mem_usage = mps_mem_mb

    if exp.seed is not None:
        import random

        random.seed(exp.seed)
        torch.manual_seed(exp.seed)

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    tr = MPSTrainer(exp, args)
    tr.bench_iters = args.bench_iters
    if args.bench_iters:
        try:
            tr.train()
        except _BenchDone:
            pass
        _report_bench(tr)
    else:
        tr.train()


if __name__ == "__main__":
    main()
