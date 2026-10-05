"""analyze_errors.py のテスト（torch 不要）。実行: python test_analyze_errors.py"""
import csv
import json
import random
import tempfile
from pathlib import Path

import analyze_errors as ae
from eval_pr import compute_pr

names = ["a", "b"]


def totals(rows):
    return {r["class"]: (r["tp"], r["fp"], r["fn"]) for r in rows if r["size_bin"] == "all"}


def check_matches_compute_pr(gts, preds, iou_thr, conf_thr, bounds=(64, 128, 256)):
    res = ae.analyze(gts, preds, names, iou_thr, conf_thr)
    got = totals(ae.size_breakdown(res, names, list(bounds)))
    exp = {r["class"]: (r["tp"], r["fp"], r["fn"]) for r in compute_pr(gts, preds, names, iou_thr, conf_thr)}
    assert got == exp, (got, exp)
    # サイズ別の合計も、全サイズの合計と一致する
    rows = ae.size_breakdown(res, names, list(bounds))
    for c in names + ["all (micro)"]:
        for i, k in enumerate(("tp", "fp", "fn")):
            assert sum(r[k] for r in rows if r["class"] == c and r["size_bin"] != "all") == exp[c][i]


# --- 1. test_eval_pr.py と同じ小さな例 ---
gts = {1: [(0, [0, 0, 10, 10]), (1, [20, 20, 30, 30])], 2: [(1, [0, 0, 10, 10])]}
preds = {1: [(0, 0.9, [0, 0, 10, 10]), (1, 0.8, [20, 20, 30, 30]), (1, 0.7, [20, 20, 30, 30]),
             (0, 0.2, [50, 50, 60, 60])],
         2: [(0, 0.9, [0, 0, 10, 10])]}
check_matches_compute_pr(gts, preds, 0.5, 0.3)
res = ae.analyze(gts, preds, names, 0.5, 0.3)
assert (len(res["tp"]), len(res["fp"]), len(res["fn"])) == (2, 2, 1)
# クラス違いの FP（img 2, a）は、別クラス GT（b）との IoU が 1.0 になる
fp_conf = [r for r in res["fp"] if r["img"] == 2][0]
assert fp_conf["max_iou_same_class"] == 0.0 and fp_conf["max_iou_other_class"] == 1.0
# 重複検出の FP は、同クラス GT との IoU が 1.0
fp_dup = [r for r in res["fp"] if r["img"] == 1][0]
assert fp_dup["max_iou_same_class"] == 1.0
# クラス違いの FN（img 2, b）は、別クラス予測との IoU が 1.0
fn_conf = res["fn"][0]
assert fn_conf["img"] == 2 and fn_conf["max_iou_other_class"] == 1.0 and fn_conf["max_iou_same_class"] == 0.0

# --- 2. サイズ分類: TP・FN は GT のサイズ、FP は予測のサイズ。境界値は上側のビン ---
assert ae.size_labels([64, 128, 256]) == ["<64", "64-128", "128-256", ">=256"]
assert ae.size_bin([0, 0, 63, 63], [64, 128, 256]) == 0
assert ae.size_bin([0, 0, 64, 64], [64, 128, 256]) == 1      # √面積 = 64 ちょうど
assert ae.size_bin([0, 0, 255, 255], [64, 128, 256]) == 2
assert ae.size_bin([0, 0, 256, 256], [64, 128, 256]) == 3
g = {1: [(0, [0, 0, 100, 100]), (0, [200, 200, 500, 500])]}              # 100px(64-128) と 300px(>=256)
p = {1: [(0, 0.9, [0, 0, 100, 100]),                                       # TP（GT は 64-128）
         (0, 0.8, [600, 600, 620, 620])]}                                  # FP（予測は 20px => <64）
rows = ae.size_breakdown(ae.analyze(g, p, ["a"], 0.5, 0.3), ["a"], [64, 128, 256])
d = {(r["class"], r["size_bin"]): r for r in rows}
assert (d["a", "64-128"]["tp"], d["a", "64-128"]["fp"], d["a", "64-128"]["fn"]) == (1, 0, 0)
assert (d["a", "<64"]["tp"], d["a", "<64"]["fp"], d["a", "<64"]["fn"]) == (0, 1, 0)
assert (d["a", ">=256"]["tp"], d["a", ">=256"]["fp"], d["a", ">=256"]["fn"]) == (0, 0, 1)
assert d["a", "all"]["precision"] == 0.5 and abs(d["a", "all"]["recall"] - 0.5) < 1e-9
rows = ae.size_breakdown(ae.analyze(g, p, ["a"], 0.5, 0.3), ["a"], [150])   # 境界を変更
assert [r["size_bin"] for r in rows if r["class"] == "a"] == ["<150", ">=150", "all"]

# --- 3. ランダムな入力で compute_pr と TP/FP/FN が一致する（同点スコア・重なりの多い箱を含む）---
rnd = random.Random(0)


def rbox():
    x, y = rnd.uniform(0, 300), rnd.uniform(0, 300)
    return [x, y, x + rnd.uniform(5, 300), y + rnd.uniform(5, 300)]


for trial in range(300):
    gts_r = {i: [(rnd.randrange(2), rbox()) for _ in range(rnd.randrange(6))] for i in range(4)}
    preds_r = {}
    for i in range(5):                                          # 画像 4 は GT なし（全部 FP になる）
        lst = []
        for _ in range(rnd.randrange(8)):
            if gts_r.get(i) and rnd.random() < 0.7:             # GT の近くに予測を作る
                k, b = rnd.choice(gts_r[i])
                b = [v + rnd.uniform(-20, 20) for v in b]
                lst.append((k if rnd.random() < 0.8 else 1 - k, rnd.choice([0.5, 0.5, rnd.random()]), b))
            else:
                lst.append((rnd.randrange(2), rnd.random(), rbox()))
        preds_r[i] = lst
    check_matches_compute_pr(gts_r, preds_r, rnd.choice([0.3, 0.5, 0.7]), rnd.choice([0.0, 0.3, 0.6]),
                             bounds=rnd.choice([(64, 128, 256), (50,), (10, 20, 30, 40)]))

# --- 4. 画像ランキング ---
res = ae.analyze(gts, preds, names, 0.5, 0.3)
assert ae.rank_images(res, names, 1) == {"a": [(2, 0, 1)], "b": [(1, 0, 1)]}     # b は img1(FP 1) と img2(FN 1) が同数 => image_id 順
assert ae.rank_images(res, names, 5)["b"] == [(1, 0, 1), (2, 1, 0)]

# --- 5. 保存 → 読み込み → CSV/PNG 出力（合成画像で end-to-end）---
import cv2
import numpy as np
import analyze_errors as m
from eval_pr import save_preds

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "img").mkdir()
    for i in (1, 2):
        cv2.imwrite(str(td / "img" / f"im{i}.jpg"), np.full((200, 300, 3), 128, np.uint8))
    ann = {"images": [{"id": 1, "file_name": "im1.jpg", "width": 300, "height": 200},
                      {"id": 2, "file_name": "im2.jpg", "width": 300, "height": 200}],
           "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                           {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10]},
                           {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 10, 10]}]}
    json.dump(ann, open(td / "ann.json", "w"))
    imgs, gts_f, nm = m.load_gt(td / "ann.json")
    assert nm == ["a", "b"]
    save_preds(td / "sub" / "preds.json", preds, imgs, nm, {"split": "test"})
    saved = json.load(open(td / "sub" / "preds.json"))
    assert saved["predictions"][0].keys() >= {"image_id", "file_name", "class", "class_name", "score", "bbox_xyxy"}
    assert len(saved["predictions"]) == 5                         # conf 未満の予測（0.2）も保存される
    loaded = m.load_preds(td / "sub" / "preds.json", nm)
    assert {k: sorted(v) for k, v in loaded.items()} == {k: sorted((c, s, b) for c, s, b in v) for k, v in preds.items()}
    # クラス名が食い違う JSON は拒否する
    try:
        m.load_preds(td / "sub" / "preds.json", ["x", "y"])
        raise SystemExit("クラス名の不一致が検出されなかった")
    except ValueError:
        pass
    import subprocess, sys
    out = td / "out"
    r = subprocess.run([sys.executable, "analyze_errors.py", "--preds", str(td / "sub" / "preds.json"),
                        "--ann", str(td / "ann.json"), "--img-dir", str(td / "img"), "--out-dir", str(out),
                        "--top-n", "5"], capture_output=True, text=True, cwd=Path(__file__).parent)
    assert r.returncode == 0, r.stderr
    sb = list(csv.DictReader(open(out / "size_breakdown.csv", encoding="utf-8")))
    ref = {r_["class"]: r_ for r_ in compute_pr(gts_f, loaded, nm, 0.5, 0.3)}
    for row in sb:
        if row["size_bin"] == "all" and row["class"] in ref:
            assert (int(row["tp"]), int(row["fp"]), int(row["fn"])) == (ref[row["class"]]["tp"], ref[row["class"]]["fp"], ref[row["class"]]["fn"])
    er = list(csv.DictReader(open(out / "errors.csv", encoding="utf-8")))
    assert len(er) == 3 and {e["type"] for e in er} == {"FN", "FP"}, er       # FN 1 + FP 2
    pngs = sorted((out / "images").rglob("*.png"))
    assert len(pngs) == 3, pngs                                                 # a: img2 / b: img1, img2
    assert all(cv2.imread(str(p)) is not None for p in pngs)

print("ok")
