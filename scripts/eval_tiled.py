#!/usr/bin/env python3
"""学習済み YOLOX に対して、推論の仕方だけを変えて Precision/Recall を測る（再学習なし）。

方式（--tile で切り替え）:
  --tile 0      … 全体画像を 1 回推論する。入力サイズは --size、リサイズ方式は --resize で選ぶ。
                   resize=square: 正方形 size x size にレターボックス（eval_pr.py と同じ。640 なら完全に一致する）
                   resize=long  : 長辺を size に合わせ、短辺は 32 の倍数まで 0 埋め（YOLOX は全畳み込みなので
                                  長方形入力でよい。縦長写真で無駄な余白を作らず、細い物体の実効解像度を上げる）
  --tile T      … SAHI 方式のタイル推論。元画像を一辺 T px・重なり --overlap の正方形タイルに分け、各タイルを
                   --tile-size（既定 640）で推論する。--add-full（既定オン）で全体画像の推論（--size/--resize）も加える。
                   全部を元座標に戻し、--merge で統合する。

統合方法（--merge）:
  nms … クラスごとの NMS（IoU --nms、既定 0.45）。eval_pr と同じ後処理で、重複を捨てるだけ。
  nmm … SAHI の GREEDYNMM 相当。スコア降順に採用し、同クラスで IOS（共通部分 / 小さい方の面積）が
        --nmm-ios 以上の箱を採用箱に併合（外接矩形）する。タイル境界で分断された大きな鉄塔の断片を
        1 つの箱に戻すことを狙う。ただし外接矩形は余計に広がる危険もあるため、nms と比較して選ぶ。
  タイル境界で切れた箱への対処: --edge-drop を付けると、タイル内の箱のうち「タイル枠に接していて、
  かつその枠が画像端ではない」ものを捨てる（切断された断片が FP や重複になるのを防ぐ。全体画像の推論は対象外）。

P/R の計算・マッチングは eval_pr.py の compute_pr / load_gt / save_preds を再利用する（ロジックを二重に持たない）。
eval_pr.py は --save-preds 対応版が必要。置き場所は環境変数 EVAL_PR_DIR（既定: このスクリプトの親ディレクトリ）。
出力は eval_pr と同じ <out>.json / <out>.csv と、--save-preds の予測 JSON（analyze_errors.py にそのまま使える）。
conf 0.3 / IoU 0.5 / NMS 0.45 は既定値のまま固定して使う。推論は conf 0.01 で行い、P/R の計算側で 0.3 を適用する
（NMS は高スコアの箱だけが低スコアの箱を消すので、後から 0.3 で絞っても 0.3 で NMS したのと結果は同じ）。

使い方（YOLOX と exp のあるディレクトリを PYTHONPATH に通す）:
  PYTHONPATH=<YOLOX>:<exps> python scripts/eval_tiled.py -f <exp.py> -c <ckpt> --data-dir datasets/windfarm \\
      --split val --tile 640 --out <prefix> --save-preds <prefix>_preds.json --device mps
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("EVAL_PR_DIR", str(Path(__file__).resolve().parent.parent)))
from eval_pr import compute_pr, load_gt, save_preds  # noqa: E402  マッチング・P/R・予測保存は eval_pr のものを使う

import numpy as np  # noqa: E402
import torch  # noqa: E402

# MPS では Tensor.type('torch.mps.FloatTensor') が再入力できず YOLOX の decode_outputs が落ちるので差し替える
# （scripts/eval_pr_mps.py と同じ回避策）
_MAP = {"torch.mps.FloatTensor": torch.float32, "torch.mps.HalfTensor": torch.float16,
        "torch.mps.LongTensor": torch.int64, "torch.mps.BoolTensor": torch.bool}
_orig_type = torch.Tensor.type


def _type(self, dtype=None, *a, **k):
    if isinstance(dtype, str) and dtype in _MAP:
        return self.to(device="mps", dtype=_MAP[dtype])
    return _orig_type(self, dtype, *a, **k)


torch.Tensor.type = _type


def preprocess(img, size, resize):
    """BGR 画像を YOLOX の入力にする（ValTransform(legacy=False) と同じ: 正規化なし、余白は 114）。
    戻り値: (CHW float32, 元画像 -> 入力の倍率)。"""
    import cv2
    h, w = img.shape[:2]
    if resize == "square":
        r = min(size / h, size / w)
        ph, pw = size, size
    else:  # long: 長辺を size に合わせ、32 の倍数まで 0 埋め（114 埋め）
        r = size / max(h, w)
        ph, pw = -(-int(round(h * r)) // 32) * 32, -(-int(round(w * r)) // 32) * 32
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    canvas = np.full((ph, pw, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32), r


def infer(model, img, size, resize, num_classes, conf, nms, device):
    """1 枚（または 1 タイル）を推論し、画像座標の検出 ndarray [[x1,y1,x2,y2,score,cls], ...] を返す。"""
    from yolox.utils import postprocess
    x, r = preprocess(img, size, resize)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    out = postprocess(model(x), num_classes, conf, nms, class_agnostic=False)[0]
    if out is None:
        return np.zeros((0, 6), dtype=np.float32)
    out = out.cpu().float().numpy()
    return np.concatenate([out[:, :4] / r, (out[:, 4] * out[:, 5])[:, None], out[:, 6:7]], axis=1)


def tile_origins(length, tile, overlap):
    """1 軸方向のタイル開始位置。刻み = tile * (1 - overlap)、最後のタイルは画像端に揃える。"""
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, length - tile, stride))
    xs.append(length - tile)
    return xs


def merge_nms(dets, iou_thr):
    """クラスごとの NMS（torchvision の batched_nms）。"""
    import torchvision
    if len(dets) == 0:
        return dets
    t = torch.from_numpy(dets)
    keep = torchvision.ops.batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou_thr)
    return dets[keep.numpy()]


def merge_nmm(dets, ios_thr):
    """SAHI の GREEDYNMM 相当。スコア降順に採用箱を決め、同クラスで IOS >= ios_thr の箱を外接矩形で併合する。
    併合後の箱のスコアは採用箱のスコアのまま。"""
    out = []
    for c in np.unique(dets[:, 5]) if len(dets) else []:
        d = dets[dets[:, 5] == c]
        d = d[np.argsort(-d[:, 4], kind="stable")]
        alive = np.ones(len(d), dtype=bool)
        for i in range(len(d)):
            if not alive[i]:
                continue
            alive[i] = False
            box = d[i, :4].copy()
            for j in np.nonzero(alive)[0]:
                b = d[j, :4]
                iw = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
                ih = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
                inter = iw * ih
                small = min((box[2] - box[0]) * (box[3] - box[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if small > 0 and inter / small >= ios_thr:
                    alive[j] = False
                    box = np.array([min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])])
            out.append([*box, d[i, 4], c])
    return np.array(out, dtype=np.float32).reshape(-1, 6)


def predict_image(model, img, a, num_classes, device):
    """1 枚の画像に対する全処理（全体推論 + タイル推論 + 統合）。元座標の検出 ndarray を返す。"""
    h, w = img.shape[:2]
    parts = []
    if a.tile == 0 or a.add_full:
        parts.append(infer(model, img, a.size, a.resize, num_classes, a.infer_conf, a.nms, device))
    if a.tile > 0:
        for y0 in tile_origins(h, a.tile, a.overlap):
            for x0 in tile_origins(w, a.tile, a.overlap):
                x1, y1 = min(x0 + a.tile, w), min(y0 + a.tile, h)
                d = infer(model, img[y0:y1, x0:x1], a.tile_size, "square", num_classes, a.infer_conf, a.nms, device)
                if a.edge_drop and len(d):
                    # タイル枠に接している箱のうち、その枠が画像端でないもの（= 切断された可能性が高い箱）を捨てる
                    m = 2.0
                    cut = ((d[:, 0] <= m) & (x0 > 0)) | ((d[:, 1] <= m) & (y0 > 0)) | \
                          ((d[:, 2] >= x1 - x0 - m) & (x1 < w)) | ((d[:, 3] >= y1 - y0 - m) & (y1 < h))
                    d = d[~cut]
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
                parts.append(d)
    dets = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    if len(parts) > 1 or a.tile > 0:
        dets = merge_nmm(dets, a.nmm_ios) if a.merge == "nmm" else merge_nms(dets, a.nms)
    return dets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--conf", type=float, default=0.3, help="P/R 計算の conf 閾値（固定 0.3）")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--infer-conf", type=float, default=0.01, help="推論時の conf 下限（保存用。P/R は --conf で適用）")
    ap.add_argument("--size", type=int, default=640, help="全体画像の推論サイズ（resize=long なら長辺）")
    ap.add_argument("--resize", choices=["square", "long"], default="square")
    ap.add_argument("--tile", type=int, default=0, help="タイルの一辺 px。0 ならタイル推論なし")
    ap.add_argument("--tile-size", type=int, default=640, help="各タイルの推論サイズ")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--add-full", type=int, default=1, help="タイル推論に全体画像の推論も加える（1/0）")
    ap.add_argument("--merge", choices=["nms", "nmm"], default="nms")
    ap.add_argument("--nmm-ios", type=float, default=0.5)
    ap.add_argument("--edge-drop", action="store_true")
    ap.add_argument("--out", required=True, help="出力プレフィックス（.json と .csv を書く）")
    ap.add_argument("--save-preds", default=None, metavar="PATH")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    import cv2
    from yolox.exp import get_exp
    os.environ["YOLOX_DATA_DIR"] = a.data_dir
    os.environ["EVAL_SPLIT"] = a.split
    exp = get_exp(a.exp_file, None)
    name = {"val": "val2017", "test": "test2017"}[a.split]
    imgs, gts, names = load_gt(Path(a.data_dir) / "annotations" / f"instances_{name}.json")

    model = exp.get_model()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.to(a.device).eval()

    def sync():
        if a.device == "mps":
            torch.mps.synchronize()

    preds, times = {}, []
    with torch.no_grad():
        for n, (iid, info) in enumerate(imgs.items()):
            img = cv2.imread(str(Path(a.data_dir) / name / info["file_name"]))
            if n == 0:  # ウォームアップ（計時しない）
                predict_image(model, img, a, exp.num_classes, a.device)
            sync()
            t0 = time.perf_counter()
            dets = predict_image(model, img, a, exp.num_classes, a.device)
            sync()
            times.append(time.perf_counter() - t0)
            preds[iid] = [(int(d[5]), float(d[4]), [float(v) for v in d[:4]]) for d in dets]

    cfg = {k: getattr(a, k) for k in ("size", "resize", "tile", "tile_size", "overlap", "add_full", "merge",
                                     "nmm_ios", "edge_drop")}
    if a.save_preds:
        save_preds(a.save_preds, preds, imgs, names,
                   {"split": a.split, "ckpt": a.ckpt, "nms": a.nms, "infer_conf": a.infer_conf, "infer": cfg})
    rows = compute_pr(gts, preds, names, a.iou, a.conf)
    res = {"split": a.split, "ckpt": a.ckpt, "conf": a.conf, "nms": a.nms, "iou": a.iou, "infer": cfg,
           "n_images": len(imgs), "sec_per_image": float(np.mean(times)), "results": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['class']:14s} P={r['precision']:.4f} R={r['recall']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}")
    print(f"sec/image={np.mean(times):.3f}")


if __name__ == "__main__":
    main()
