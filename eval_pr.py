#!/usr/bin/env python3
"""YOLOX モデルのクラス別および全体の Precision/Recall を、固定の conf/NMS・IoU=0.5 で求める。

Colab で YOLOX リポジトリのルートから実行する（torch, yolox, cv2 が必要）:
  python eval_pr.py -f exps/yolox_s_windfarm.py -c YOLOX_outputs/yolox_s_windfarm/best_ckpt.pth \
      --data-dir /content/datasets/windfarm --split test --conf 0.3 --nms 0.45 --out results/baseline_test

マッチング: 画像・クラスごとに予測をスコア降順に並べ、各予測を、未マッチの GT のうち
IoU が最大のもの（iou 閾値以上）に対応づける。TP/FP/FN はデータセット全体で合計し、
P=TP/(TP+FP)、R=TP/(TP+FN) とする。
<out>.json と <out>.csv を出力する。COCO mAP は tools/eval.py を別途実行する。

--save-preds <path> を付けると、推論した全予測（conf 閾値をかける前のもの）を別の JSON に保存する
（analyze_errors.py の入力になる）。指定しなければ保存せず、出力・挙動は従来どおり。

末尾に YOLOX の train.py / eval.py と同じ形式の上書き（キーと値の組）を置ける。例: ... test_size "(800,800)"
入力サイズを変えて学習したモデルを、同じ test_size で評価するために使う。指定しなければ従来どおり。
"""
import argparse, csv, json, os
from pathlib import Path


def iou_xyxy(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_preds(preds_img, c, conf_thr):
    """1 枚の画像の予測 [(cls, score, box), ...] から、クラス c で score が conf_thr 以上のものを
    score 降順（同点は元の順）に並べた [(score, box), ...] を返す。"""
    return sorted([(s, b) for k, s, b in preds_img if k == c and s >= conf_thr], key=lambda x: -x[0])


def match_greedy(g_boxes, p_sorted, iou_thr):
    """1 画像・1 クラス分のマッチング。p_sorted は select_preds の結果（score 降順）。
    各予測を、未使用の GT のうち IoU が最大のもの（iou_thr 以上）に対応づける。
    戻り値: (matches, used)。matches[i] は p_sorted[i] に対応した GT の添字（なければ -1 で FP）、
    used[j] は g_boxes[j] がマッチ済みか（False なら FN）。"""
    used = [False] * len(g_boxes)
    matches = []
    for s, b in p_sorted:
        best, bj = iou_thr, -1
        for j, gb in enumerate(g_boxes):
            if used[j]:
                continue
            v = iou_xyxy(b, gb)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
        matches.append(bj)
    return matches, used


def compute_pr(gts, preds, class_names, iou_thr=0.5, conf_thr=0.3):
    """gts: {img_id: [(cls, [x1,y1,x2,y2]), ...]}; preds: {img_id: [(cls, score, [x1,y1,x2,y2]), ...]}.
    cls は class_names への 0 始まりのインデックス。"""
    n = len(class_names)
    tp, fp, fn = [0] * n, [0] * n, [0] * n
    for img in set(gts) | set(preds):
        for c in range(n):
            g = [b for k, b in gts.get(img, []) if k == c]
            p = select_preds(preds.get(img, []), c, conf_thr)
            matches, used = match_greedy(g, p, iou_thr)
            n_tp = sum(1 for m in matches if m >= 0)
            tp[c] += n_tp
            fp[c] += len(matches) - n_tp
            fn[c] += used.count(False)

    def row(name, t, f, m):
        return {"class": name, "tp": t, "fp": f, "fn": m,
                "precision": t / (t + f) if t + f else 0.0,
                "recall": t / (t + m) if t + m else 0.0}
    rows = [row(class_names[c], tp[c], fp[c], fn[c]) for c in range(n)]
    rows.append(row("all (micro)", sum(tp), sum(fp), sum(fn)))
    return rows


def load_gt(ann_path):
    d = json.load(open(ann_path))
    cat_ids = sorted(c["id"] for c in d["categories"])
    names = [next(c["name"] for c in d["categories"] if c["id"] == i) for i in cat_ids]
    imgs = {i["id"]: i for i in d["images"]}
    gts = {i: [] for i in imgs}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        gts[a["image_id"]].append((cat_ids.index(a["category_id"]), [x, y, x + w, y + h]))
    return imgs, gts, names


def predict(exp, ckpt, imgs, img_dir, conf, nms, device, fp16, fuse):
    import cv2, torch
    from yolox.data.data_augment import ValTransform
    from yolox.utils import postprocess

    model = exp.get_model()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.to(device).eval()
    if fuse:
        from yolox.utils import fuse_model
        model = fuse_model(model)
    if fp16:
        model = model.half()
    tf = ValTransform(legacy=False)
    size = exp.test_size
    preds = {}
    with torch.no_grad():
        for iid, info in imgs.items():
            img = cv2.imread(str(Path(img_dir) / info["file_name"]))
            ratio = min(size[0] / img.shape[0], size[1] / img.shape[1])
            x, _ = tf(img, None, size)
            x = torch.from_numpy(x).unsqueeze(0).float().to(device)
            if fp16:
                x = x.half()
            out = postprocess(model(x), exp.num_classes, conf, nms, class_agnostic=False)[0]
            preds[iid] = []
            if out is not None:
                out = out.cpu().float()
                for o in out:
                    preds[iid].append((int(o[6]), float(o[4] * o[5]), [float(v) / ratio for v in o[:4]]))
    return preds


def save_preds(path, preds, imgs, class_names, meta):
    """全予測を JSON に保存する。bbox は元画像座標の xyxy、クラスは 0 始まりの添字と名前の両方を持つ。"""
    rows = []
    for iid in sorted(preds):
        for k, s, b in preds[iid]:
            rows.append({"image_id": iid, "file_name": imgs[iid]["file_name"], "class": k,
                         "class_name": class_names[k], "score": s, "bbox_xyxy": b})
    out = {"meta": dict(meta, class_names=class_names, n_images=len(imgs)), "predictions": rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="dataset root (train2017/val2017/test2017/annotations)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="output prefix (writes .json and .csv)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--save-preds", default=None, metavar="PATH",
                    help="推論した全予測（conf 閾値適用前）を JSON に保存する。未指定なら保存しない")
    ap.add_argument("--save-conf", type=float, default=0.01,
                    help="--save-preds 指定時の推論用 conf 下限（既定 0.01）。P/R の計算は従来どおり --conf で行う")
    ap.add_argument("opts", nargs=argparse.REMAINDER,
                    help="exp の属性の上書き（キー 値 の組。例: test_size \"(800,800)\"）。exp.merge に渡す")
    a = ap.parse_args()

    from yolox.exp import get_exp
    os.environ["YOLOX_DATA_DIR"] = a.data_dir
    os.environ["EVAL_SPLIT"] = a.split
    exp = get_exp(a.exp_file, None)
    if a.opts:
        exp.merge(a.opts)
    name = {"val": "val2017", "test": "test2017"}[a.split]
    imgs, gts, names = load_gt(Path(a.data_dir) / "annotations" / f"instances_{name}.json")
    # 保存時だけ推論の conf 下限を下げる。P/R は compute_pr 側で --conf を適用するので値は変わらない
    infer_conf = min(a.conf, a.save_conf) if a.save_preds else a.conf
    preds = predict(exp, a.ckpt, imgs, Path(a.data_dir) / name, infer_conf, a.nms, a.device, a.fp16, a.fuse)
    if a.save_preds:
        save_preds(a.save_preds, preds, imgs, names,
                   {"split": a.split, "ckpt": a.ckpt, "nms": a.nms, "infer_conf": infer_conf,
                    "test_size": list(exp.test_size)})
    rows = compute_pr(gts, preds, names, a.iou, a.conf)
    res = {"split": a.split, "ckpt": a.ckpt, "conf": a.conf, "nms": a.nms, "iou": a.iou,
           "test_size": list(exp.test_size), "n_images": len(imgs), "results": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['class']:14s} P={r['precision']:.4f} R={r['recall']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}")


if __name__ == "__main__":
    main()
