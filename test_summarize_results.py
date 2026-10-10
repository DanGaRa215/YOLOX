import csv
import io
import json
import tempfile
from pathlib import Path

from summarize_results import (build_seed_table, build_table, discover_runs, fmt_stat, mean_std, parse_cocomap,
                               split_seed, to_csv, to_markdown)

# 実際の pycocotools 出力（と YOLOX のクラス別 AP 表）の抜粋
COCO = """\
Accumulating evaluation results...
DONE (t=0.05s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.801
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600
per class AP:
| class       | AP     | class   | AP     |
|:------------|:-------|:--------|:-------|
| cable tower | 30.100 | turbine | 72.300 |
"""
assert parse_cocomap(COCO) == (0.801, 0.512), parse_cocomap(COCO)
assert parse_cocomap("nothing") == (None, None)
# 複数回出る場合は最後を採る
assert parse_cocomap(COCO + COCO.replace("0.801", "0.900").replace("0.512", "0.600")) == (0.900, 0.600)


def write(d, exp, split, tower, turbine, allm, ap50, ap5095):
    """tower / turbine / allm は (tp, fp, fn, precision, recall)。"""
    rows = []
    for name, (tp, fp, fn, p, r) in (("cable tower", tower), ("turbine", turbine), ("all (micro)", allm)):
        rows.append({"class": name, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r})
    (d / f"{exp}_{split}.json").write_text(json.dumps({"results": rows}))
    (d / f"{exp}_{split}_cocomap.txt").write_text(
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ap5095}\n"
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {ap50}\n")


with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    # ベースライン: val / test、施策 (A): test のみ、他は欠損
    write(d, "yolox_s_windfarm", "val", (15, 5, 11, 0.750, 0.577), (90, 10, 10, 0.900, 0.900),
          (105, 15, 21, 0.875, 0.833), 0.800, 0.500)
    write(d, "yolox_s_windfarm", "test", (10, 10, 10, 0.500, 0.500), (80, 20, 20, 0.800, 0.800),
          (90, 30, 30, 0.750, 0.750), 0.700, 0.400)
    write(d, "yolox_s_windfarm_oversample", "test", (14, 6, 6, 0.700, 0.700), (80, 20, 20, 0.800, 0.800),
          (94, 26, 26, 0.783, 0.783), 0.732, 0.4)

    header, rows = build_table(d)
    assert len(rows) == 8  # 4 実験 x 2 split
    by = {(r[0], r[1]): dict(zip(header, r)) for r in rows}
    assert [r[0] for r in rows[::2]] == ["ベースライン", "(C) close-mosaic 延長", "(B) クラス重み", "(A) オーバーサンプリング"]

    b = by[("ベースライン", "val")]
    assert b["cable tower R"] == "0.577 (15/26)", b
    assert b["cable tower P"] == "0.750" and b["AP50"] == "0.800" and b["AP50:95"] == "0.500", b
    assert b["Δ全体 P"] == "+0.0"

    a = by[("(A) オーバーサンプリング", "test")]
    assert a["cable tower R"] == "0.700 (14/20)", a
    assert a["Δcable tower P"] == "+20.0" and a["Δcable tower R"] == "+20.0", a
    assert a["Δturbine P"] == "+0.0" and a["Δ全体 P"] == "+3.3", a
    assert a["ΔAP50"] == "+3.2" and a["ΔAP50:95"] == "+0.0", a

    # val が無い施策は「未実施」（差分は空欄）
    m = by[("(A) オーバーサンプリング", "val")]
    assert all(m[h] == "未実施" for h in header[2:10]), m
    assert all(m[h] == "" for h in header[10:]), m
    assert by[("(C) close-mosaic 延長", "test")]["AP50"] == "未実施"

    md = to_markdown(header, rows)
    lines = md.splitlines()
    assert len(lines) == 2 + 8 and lines[0].startswith("| 実験 | split |"), md
    assert "| ベースライン | val | 0.750 | 0.577 (15/26) |" in md
    # 完全欠損は (C) val/test、(B) val/test、(A) val の 5 行 x 8 指標
    assert md.count("未実施") == 5 * 8, md.count("未実施")

    got = list(csv.reader(io.StringIO(to_csv(header, rows))))
    assert got[0] == header and len(got) == 9
    assert got[1][:4] == ["ベースライン", "val", "0.750", "0.577 (15/26)"], got[1]

    # ベースラインの test が無ければ、その split の差分は空欄
    (d / "yolox_s_windfarm_test.json").unlink()
    (d / "yolox_s_windfarm_test_cocomap.txt").unlink()
    header, rows = build_table(d)
    by = {(r[0], r[1]): dict(zip(header, r)) for r in rows}
    a = by[("(A) オーバーサンプリング", "test")]
    assert a["cable tower P"] == "0.700" and a["Δcable tower P"] == "" and a["ΔAP50"] == "", a
# ---- seed をまたいだ集計 ----
# グループ化の規則: 末尾の _s<数字> だけを取り除く。無ければ seed 1。名前の途中の _s_ には反応しない
assert split_seed("yolox_s_windfarm") == ("yolox_s_windfarm", 1)
assert split_seed("yolox_s_windfarm_s2") == ("yolox_s_windfarm", 2)
assert split_seed("yolox_s_windfarm_ecbam_in800_s1") == ("yolox_s_windfarm_ecbam_in800", 1)
assert split_seed("yolox_s_windfarm_s12") == ("yolox_s_windfarm", 12)
assert split_seed("yolox_s_windfarm_ecbam") == ("yolox_s_windfarm_ecbam", 1)
assert split_seed("yolox_s_x") == ("yolox_s_x", 1)   # 途中の _s_ は対象外
assert split_seed("a_s2_b") == ("a_s2_b", 1)          # 途中の _s2_ は対象外
assert split_seed("a_s2x") == ("a_s2x", 1)

# 平均と標本標準偏差（n-1）。n=1 は標準偏差なし、n=0 は値なし
m, sd, n = mean_std([0.2, 0.4, 0.6])
assert abs(m - 0.4) < 1e-12 and abs(sd - 0.2) < 1e-12 and n == 3, (m, sd, n)
assert mean_std([0.5]) == (0.5, None, 1)
assert mean_std([]) == (None, None, 0)
assert fmt_stat(0.4, 0.2, 3, "fraction", 3) == "0.400 ± 0.200"
assert fmt_stat(0.5, None, 1, "fraction", 1) == "0.500"                  # n=1: ± を付けない
assert fmt_stat(0.4, 0.2, 2, "fraction", 3) == "0.400 ± 0.200 (n=2)"    # 行の本数より少ないとき
assert fmt_stat(12.0, 1.5, 3, "count", 3) == "12.0 ± 1.5"
assert fmt_stat(None, None, 0, "fraction", 3) == "未実施"

with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    # ベースライン: seedless（= seed 1）と _s2, _s3。(A) は seedless のみ（n=1）。tower の TP を 6 / 8 / 10 にする
    for exp, tp_ in (("yolox_s_windfarm", 6), ("yolox_s_windfarm_s2", 8), ("yolox_s_windfarm_s3", 10)):
        write(d, exp, "test", (tp_, 10 - tp_, 4, tp_ / 10, tp_ / (tp_ + 4)), (80, 20, 20, 0.8, 0.8),
              (tp_ + 80, 30 - tp_, 24, 0.7, 0.7), 0.7 + tp_ / 100, 0.4)
    write(d, "yolox_s_windfarm_oversample", "test", (9, 1, 1, 0.9, 0.9), (80, 20, 20, 0.8, 0.8),
          (89, 21, 21, 0.8, 0.8), 0.8, 0.5)
    (d / "yolox_s_windfarm_s2_test_preds.json").write_text("{}")  # 予測ファイルは実験として拾わない
    groups = discover_runs(d)
    assert groups == {"yolox_s_windfarm": ["yolox_s_windfarm", "yolox_s_windfarm_s2", "yolox_s_windfarm_s3"],
                      "yolox_s_windfarm_oversample": ["yolox_s_windfarm_oversample"]}, groups

    header, rows = build_seed_table(d)
    assert header[:4] == ["実験", "split", "n", "cable tower P"], header[:4]
    for h in ("cable tower TP", "turbine FP", "全体 FN", "AP50", "AP50:95", "Δcable tower P", "Δ全体 FN"):
        assert h in header, h
    by = {(r[0], r[1]): dict(zip(header, r)) for r in rows}
    assert [r[0] for r in rows[::2]] == ["ベースライン", "(A) オーバーサンプリング"], rows
    b = by[("ベースライン", "test")]
    assert b["n"] == "3"
    assert b["cable tower P"] == "0.800 ± 0.200", b   # 平均 0.8、標本標準偏差 0.2
    assert b["cable tower TP"] == "8.0 ± 2.0", b
    assert b["cable tower FN"] == "4.0 ± 0.0", b
    assert b["turbine P"] == "0.800 ± 0.000", b
    assert b["AP50"] == "0.780 ± 0.020", b             # 0.76 / 0.78 / 0.80
    assert b["Δcable tower P"] == "+0.0" and b["ΔAP50"] == "+0.0", b
    a = by[("(A) オーバーサンプリング", "test")]
    assert a["n"] == "1" and a["cable tower P"] == "0.900", a   # n=1 は ± なし
    assert a["Δcable tower P"] == "+10.0", a                    # 平均の差（ポイント）
    assert a["Δcable tower TP"] == "+1.0", a                    # 件数の差
    assert a["ΔAP50"] == "+2.0", a
    # 結果が無い split は n=0 で「未実施」、差分は空欄
    v = by[("ベースライン", "val")]
    assert v["n"] == "0" and v["cable tower P"] == "未実施" and v["Δcable tower P"] == "", v
    md = to_markdown(header, rows)
    assert "| ベースライン | test | 3 | 0.800 ± 0.200 |" in md, md
    got = list(csv.reader(io.StringIO(to_csv(header, rows))))
    assert got[0] == header and len(got) == 1 + 4

    # 既存の表（--aggregate-seeds なし）は、seed 付きの名前が増えても変わらない（seedless の 1 本だけを使う）
    header0, rows0 = build_table(d)
    assert len(rows0) == 8 and header0[:2] == ["実験", "split"] and header0[2] == "cable tower P"
    by0 = {(r[0], r[1]): dict(zip(header0, r)) for r in rows0}
    assert by0[("ベースライン", "test")]["cable tower R"] == "0.600 (6/10)", by0[("ベースライン", "test")]
print("ok")
