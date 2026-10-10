#!/usr/bin/env python3
"""eval_pr.py --save-preds で保存した予測から、見逃し（FN）と誤検出（FP）を分析する。

torch も GPU も使わない（標準ライブラリのみ。PNG 出力のときだけ cv2 を使う）。
  python analyze_errors.py --preds results/baseline_test_preds.json \\
      --ann datasets/windfarm/annotations/instances_test2017.json \\
      --img-dir datasets/windfarm/test2017 --out-dir results/error_analysis/baseline_test

出力（--out-dir 以下）:
  1. size_breakdown.csv … クラス別 × サイズ別の TP / FP / FN / Precision / Recall
  2. errors.csv         … FN・FP の一覧（画像名、クラス、bbox、score、最大 IoU）
  3. images/<クラス名>/ … FN または FP を含む画像に GT・TP・FN・FP を描いた PNG
                          （クラスごとに FN+FP の多い順で上位 N 枚）

マッチング規則は eval_pr.compute_pr と同一で、eval_pr の select_preds / match_greedy を再利用する
（クラスごと、score 降順、未使用 GT のうち IoU 最大で閾値以上なら TP）。
サイズは √面積（px、元画像座標）で分け、TP・FN は GT のサイズ、FP は予測のサイズで分類する。
そのためサイズ別の Precision（TP/(TP+FP)）は、分子が GT のサイズ・分母の FP が予測のサイズという
近似値になる。サイズを問わない合計（size_bin=all）は compute_pr の結果と一致する。
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

from eval_pr import iou_xyxy, load_gt, match_greedy, select_preds

# 描画色（BGR）
COLORS = {"GT": (0, 200, 0), "TP": (255, 0, 0), "FN": (0, 0, 255), "FP": (0, 255, 255)}


def sqrt_area(box):
    """xyxy の bbox の √面積（px）。"""
    return math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def size_labels(bounds):
    """サイズ境界（昇順）からビン名のリストを作る。例: [64,128,256] -> ['<64','64-128','128-256','>=256']"""
    b = [f"{v:g}" for v in bounds]
    return [f"<{b[0]}"] + [f"{b[i]}-{b[i + 1]}" for i in range(len(b) - 1)] + [f">={b[-1]}"]


def size_bin(box, bounds):
    """√面積が境界のどのビンに入るか（境界値は上側のビン）。ビン番号を返す。"""
    v = sqrt_area(box)
    for i, t in enumerate(bounds):
        if v < t:
            return i
    return len(bounds)


def load_preds(path, class_names):
    """保存した予測 JSON を読み、compute_pr と同じ形 {img_id: [(cls, score, box), ...]} を返す。
    JSON 内のクラス名が GT のクラス名と食い違う場合は例外にする（添字の取り違え防止）。"""
    d = json.load(open(path, encoding="utf-8"))
    saved = d.get("meta", {}).get("class_names")
    if saved is not None and list(saved) != list(class_names):
        raise ValueError(f"予測 JSON のクラス名 {saved} が GT のクラス名 {list(class_names)} と一致しない")
    preds = {}
    for r in d["predictions"]:
        preds.setdefault(r["image_id"], []).append((int(r["class"]), float(r["score"]), list(r["bbox_xyxy"])))
    return preds


def analyze(gts, preds, class_names, iou_thr=0.5, conf_thr=0.3):
    """全画像・全クラスのマッチング結果を返す。
    戻り値: {"tp": [...], "fn": [...], "fp": [...]}。各要素は dict。
      tp: img, cls, pred_box, score, gt_box, iou
      fn: img, cls, gt_box（+ 同クラス予測・別クラス予測との最大 IoU）
      fp: img, cls, pred_box, score（+ 同クラス GT・別クラス GT との最大 IoU）
    予測は conf_thr 以上のものだけを対象にする（compute_pr と同じ）。"""
    n = len(class_names)
    res = {"tp": [], "fn": [], "fp": []}
    for img in sorted(set(gts) | set(preds), key=str):
        gt_img = gts.get(img, [])
        # conf 閾値以上の予測（FN の IoU 計算用に全クラス分）
        kept = [(k, s, b) for k, s, b in preds.get(img, []) if s >= conf_thr]
        for c in range(n):
            g = [b for k, b in gt_img if k == c]
            p = select_preds(preds.get(img, []), c, conf_thr)
            matches, used = match_greedy(g, p, iou_thr)
            for (s, b), m in zip(p, matches):
                if m >= 0:
                    res["tp"].append({"img": img, "cls": c, "pred_box": b, "score": s,
                                      "gt_box": g[m], "iou": iou_xyxy(b, g[m])})
                else:
                    res["fp"].append({
                        "img": img, "cls": c, "pred_box": b, "score": s,
                        "max_iou_same_class": max([iou_xyxy(b, gb) for gb in g], default=0.0),
                        "max_iou_other_class": max([iou_xyxy(b, gb) for k, gb in gt_img if k != c], default=0.0),
                        "compared_with": "gt"})
            for j, gb in enumerate(g):
                if not used[j]:
                    res["fn"].append({
                        "img": img, "cls": c, "gt_box": gb,
                        "max_iou_same_class": max([iou_xyxy(gb, pb) for k, _, pb in kept if k == c], default=0.0),
                        "max_iou_other_class": max([iou_xyxy(gb, pb) for k, _, pb in kept if k != c], default=0.0),
                        "compared_with": "pred"})
    return res


def size_breakdown(res, class_names, bounds):
    """クラス別 × サイズ別の集計行を作る。各クラスの size_bin=all 行と、全クラス合計 class=all (micro) 行も含む。"""
    labels = size_labels(bounds)
    nb = len(labels)
    n = len(class_names)
    cnt = {k: [[0] * nb for _ in range(n)] for k in ("tp", "fp", "fn")}
    for r in res["tp"]:
        cnt["tp"][r["cls"]][size_bin(r["gt_box"], bounds)] += 1
    for r in res["fn"]:
        cnt["fn"][r["cls"]][size_bin(r["gt_box"], bounds)] += 1
    for r in res["fp"]:
        cnt["fp"][r["cls"]][size_bin(r["pred_box"], bounds)] += 1

    def row(name, size, t, f, m):
        return {"class": name, "size_bin": size, "tp": t, "fp": f, "fn": m,
                "precision": t / (t + f) if t + f else 0.0,
                "recall": t / (t + m) if t + m else 0.0}
    rows = []
    for c in range(n):
        for b in range(nb):
            rows.append(row(class_names[c], labels[b], cnt["tp"][c][b], cnt["fp"][c][b], cnt["fn"][c][b]))
        rows.append(row(class_names[c], "all", sum(cnt["tp"][c]), sum(cnt["fp"][c]), sum(cnt["fn"][c])))
    for b in range(nb):
        rows.append(row("all (micro)", labels[b], *(sum(cnt[k][c][b] for c in range(n)) for k in ("tp", "fp", "fn"))))
    rows.append(row("all (micro)", "all", *(sum(map(sum, cnt[k])) for k in ("tp", "fp", "fn"))))
    return rows


def error_rows(res, imgs, class_names, bounds):
    """FN・FP の一覧 CSV 用の行を作る（画像名、クラス順）。"""
    labels = size_labels(bounds)
    rows = []
    for kind in ("fn", "fp"):
        for r in res[kind]:
            box = r["gt_box"] if kind == "fn" else r["pred_box"]
            rows.append({
                "type": kind.upper(), "file_name": imgs[r["img"]]["file_name"], "image_id": r["img"],
                "class": class_names[r["cls"]],
                "x1": round(box[0], 1), "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1),
                "sqrt_area": round(sqrt_area(box), 1), "size_bin": labels[size_bin(box, bounds)],
                "score": "" if kind == "fn" else round(r["score"], 4),
                "max_iou_same_class": round(r["max_iou_same_class"], 4),
                "max_iou_other_class": round(r["max_iou_other_class"], 4),
                "compared_with": r["compared_with"]})
    rows.sort(key=lambda x: (x["file_name"], x["class"], x["type"], x["x1"], x["y1"]))
    return rows


def rank_images(res, class_names, top_n):
    """クラスごとに FN+FP の多い順で上位 top_n 枚の (img_id, FN 数, FP 数) を返す。同数は image_id 順。"""
    out = {}
    for c, name in enumerate(class_names):
        cnt = {}
        for kind, idx in (("fn", 0), ("fp", 1)):
            for r in res[kind]:
                if r["cls"] == c:
                    cnt.setdefault(r["img"], [0, 0])[idx] += 1
        order = sorted(cnt.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), str(kv[0])))
        out[name] = [(i, v[0], v[1]) for i, v in order[:top_n]]
    return out


def _ascii(text):
    """cv2.putText は日本語を描けないので ASCII 以外は ? に置き換える。"""
    return text.encode("ascii", "replace").decode()


def draw_image(img, res, image_id, class_names, max_side=1600):
    """画像 1 枚に GT・TP・FN・FP を描く。GT 緑（TP でマッチ済みの GT は細線）、TP 青、FN 赤、FP 黄。
    左上に凡例、各枠にラベル（種別・クラス・score）を入れる。max_side を超える画像は縮小する。"""
    import cv2

    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w)) if max_side else 1.0
    if k < 1.0:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    th = max(2, round(max(img.shape[:2]) / 600))
    fs = max(0.5, max(img.shape[:2]) / 1800)

    def box(b, color, label, thick):
        p1, p2 = (int(b[0] * k), int(b[1] * k)), (int(b[2] * k), int(b[3] * k))
        cv2.rectangle(img, p1, p2, color, thick)
        if label:
            (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            y0 = max(p1[1], tht + bl)
            cv2.rectangle(img, (p1[0], y0 - tht - bl), (p1[0] + tw, y0 + bl // 2), color, -1)
            cv2.putText(img, label, (p1[0], y0), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)

    for r in res["tp"]:
        if r["img"] == image_id:
            box(r["gt_box"], COLORS["GT"], "", max(1, th // 2))
    for r in res["fn"]:
        if r["img"] == image_id:
            box(r["gt_box"], COLORS["GT"], "", max(1, th // 2))
    for r in res["tp"]:
        if r["img"] == image_id:
            box(r["pred_box"], COLORS["TP"], _ascii(f"TP {class_names[r['cls']]} {r['score']:.2f}"), th)
    for r in res["fn"]:
        if r["img"] == image_id:
            box(r["gt_box"], COLORS["FN"], _ascii(f"FN {class_names[r['cls']]}"), th)
    for r in res["fp"]:
        if r["img"] == image_id:
            box(r["pred_box"], COLORS["FP"], _ascii(f"FP {class_names[r['cls']]} {r['score']:.2f}"), th)
    # 凡例
    y = int(30 * fs * 1.5)
    for name, text in (("GT", "GT (matched / missed)"), ("TP", "TP (correct detection)"),
                       ("FN", "FN (missed GT)"), ("FP", "FP (false detection)")):
        sw = int(22 * fs)
        (tw, tht), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (6, y - tht - 6), (24 + sw + tw, y + bl + 2), (0, 0, 0), -1)   # 文字の下地
        cv2.rectangle(img, (10, y - sw), (10 + sw, y), COLORS[name], -1)
        cv2.putText(img, text, (16 + sw, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(34 * fs * 1.2)
    return img


def safe_name(s):
    """ファイル名・ディレクトリ名に使えない文字を _ にする。"""
    return re.sub(r"[^\w.\-]+", "_", s)


def write_csv(path, rows, fieldnames=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="eval_pr.py --save-preds で保存した予測 JSON")
    ap.add_argument("--ann", required=True, help="COCO アノテーション JSON（instances_*.json）")
    ap.add_argument("--img-dir", required=True, help="画像ディレクトリ（PNG 出力に使う）")
    ap.add_argument("--out-dir", required=True, help="出力先ディレクトリ（リポジトリ内には置かないこと）")
    ap.add_argument("--conf", type=float, default=0.3, help="conf 閾値（既定 0.3）")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 閾値（既定 0.5）")
    ap.add_argument("--size-bounds", type=float, nargs="+", default=[64, 128, 256],
                    help="サイズ境界（√面積 px、昇順、既定 64 128 256）")
    ap.add_argument("--top-n", type=int, default=20, help="クラスごとに出力する PNG の枚数（既定 20）")
    ap.add_argument("--max-side", type=int, default=1600, help="PNG の長辺の上限 px（0 で縮小しない、既定 1600）")
    ap.add_argument("--no-images", action="store_true", help="PNG を出力しない（CSV のみ）")
    a = ap.parse_args()

    bounds = sorted(a.size_bounds)
    if not bounds:
        ap.error("--size-bounds には 1 つ以上の値が要る")
    imgs, gts, names = load_gt(a.ann)
    preds = load_preds(a.preds, names)
    unknown = set(preds) - set(imgs)
    if unknown:
        raise ValueError(f"アノテーションに無い image_id の予測がある: {sorted(unknown)[:5]} ...")
    res = analyze(gts, preds, names, a.iou, a.conf)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = size_breakdown(res, names, bounds)
    write_csv(out / "size_breakdown.csv", rows)
    errs = error_rows(res, imgs, names, bounds)
    write_csv(out / "errors.csv", errs, fieldnames=[
        "type", "file_name", "image_id", "class", "x1", "y1", "x2", "y2", "sqrt_area", "size_bin", "score",
        "max_iou_same_class", "max_iou_other_class", "compared_with"])
    for r in rows:
        if r["size_bin"] == "all":
            print(f"{r['class']:14s} P={r['precision']:.4f} R={r['recall']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']}")

    n_png = 0
    if not a.no_images:
        import cv2
        for name, ranked in rank_images(res, names, a.top_n).items():
            d = out / "images" / safe_name(name)
            d.mkdir(parents=True, exist_ok=True)
            for rank, (iid, n_fn, n_fp) in enumerate(ranked, 1):
                src = Path(a.img_dir) / imgs[iid]["file_name"]
                im = cv2.imread(str(src))
                if im is None:
                    print(f"警告: 画像を読めないためスキップ: {src}")
                    continue
                im = draw_image(im, res, iid, names, a.max_side)
                dst = d / f"{rank:02d}_fn{n_fn}_fp{n_fp}_{safe_name(Path(src).stem)}.png"
                cv2.imwrite(str(dst), im)
                n_png += 1
    print(f"出力: {out}（size_breakdown.csv, errors.csv {len(errs)} 行, PNG {n_png} 枚）")


if __name__ == "__main__":
    main()
