import csv
import io
import json
import tempfile
from pathlib import Path

from summarize_results import build_table, parse_cocomap, to_csv, to_markdown

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
print("ok")
