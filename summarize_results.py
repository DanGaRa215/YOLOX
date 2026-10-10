"""結果集約スクリプト: eval_pr.py の JSON と YOLOX tools/eval.py の cocomap ログから比較表を作る。

読むファイル（--results-dir 以下）:
  <exp名>_{val,test}.json          eval_pr.py の出力（クラス別 P/R と最後の "all (micro)"）
  <exp名>_{val,test}_cocomap.txt   YOLOX tools/eval.py の標準出力（pycocotools の表を含む）
出力: Markdown 表と CSV。ベースラインからの差分（ポイント）の列も付ける。
"""
import argparse
import csv
import io
import json
import re
from pathlib import Path

# 実験名 -> 表示名（この順に表を並べる）。先頭がベースライン。
EXPERIMENTS = {
    "yolox_s_windfarm": "ベースライン",
    "yolox_s_windfarm_noaug12": "(C) close-mosaic 延長",
    "yolox_s_windfarm_clsweight": "(B) クラス重み",
    "yolox_s_windfarm_oversample": "(A) オーバーサンプリング",
}
BASELINE = "yolox_s_windfarm"
SPLITS = ("val", "test")
MISSING = "未実施"

# 指標キー -> 見出し。差分列は同じ指標を同じ順で付ける。
METRICS = [
    ("tower_p", "cable tower P"), ("tower_r", "cable tower R"),
    ("turbine_p", "turbine P"), ("turbine_r", "turbine R"),
    ("all_p", "全体 P"), ("all_r", "全体 R"),
    ("ap50", "AP50"), ("ap5095", "AP50:95"),
]

_AP = r"Average Precision\s+\(AP\)\s+@\[\s*IoU=%s\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?\d+(?:\.\d+)?)"
_RE_AP5095 = re.compile(_AP % r"0\.50:0\.95")
_RE_AP50 = re.compile(_AP % r"0\.50(?![\d:])")


def parse_cocomap(text):
    """cocomap ログから (AP50, AP50:95) を返す。見つからなければ None。

    同じ行が複数回出る場合は最後のもの（最終評価）を採る。クラス別 AP 表（tabulate 形式）は
    上記の正規表現に一致しないので影響しない。
    """
    def last(rx):
        m = rx.findall(text)
        return float(m[-1]) if m else None
    return last(_RE_AP50), last(_RE_AP5095)


def load_pr(path):
    """eval_pr.py の JSON から P/R 指標（と TP, TP+FN の件数）を返す。"""
    rows = json.loads(Path(path).read_text())["results"]
    out = {"counts": {}}
    for r in rows:
        name = r["class"].lower()
        if name.startswith("all"):
            key = "all"
        elif name == "cable tower":
            key = "tower"
        elif name == "turbine":
            key = "turbine"
        else:
            continue
        out[key + "_p"], out[key + "_r"] = r["precision"], r["recall"]
        out["counts"][key] = (r["tp"], r["tp"] + r["fn"])
    return out


def load_cell(results_dir, exp, split):
    """1 つの実験 × split の指標 dict。ファイルが無い（または抽出できない）指標は含めない。"""
    d = {"counts": {}}
    jp = Path(results_dir) / f"{exp}_{split}.json"
    if jp.exists():
        d.update(load_pr(jp))
    cp = Path(results_dir) / f"{exp}_{split}_cocomap.txt"
    if cp.exists():
        ap50, ap5095 = parse_cocomap(cp.read_text(errors="replace"))
        if ap50 is not None:
            d["ap50"] = ap50
        if ap5095 is not None:
            d["ap5095"] = ap5095
    return d


def fmt_diff(v, base):
    """ベースラインとの差をポイント表記（例 +3.2）にする。どちらかが無ければ空欄。"""
    if v is None or base is None:
        return ""
    return f"{(v - base) * 100:+.1f}"


def build_table(results_dir):
    """(ヘッダ, 行のリスト) を返す。値が無いセルは「未実施」、差分は空欄。"""
    data = {(e, s): load_cell(results_dir, e, s) for e in EXPERIMENTS for s in SPLITS}
    header = ["実験", "split"] + [h for _, h in METRICS] + ["Δ" + h for _, h in METRICS]
    rows = []
    for exp, label in EXPERIMENTS.items():
        for split in SPLITS:
            d, base = data[(exp, split)], data[(BASELINE, split)]
            vals = []
            for k, _ in METRICS:
                if k not in d:
                    vals.append(MISSING)
                    continue
                s = f"{d[k]:.3f}"
                if k == "tower_r" and "tower" in d["counts"]:
                    tp, n = d["counts"]["tower"]
                    s += f" ({tp}/{n})"
                vals.append(s)
            diffs = [fmt_diff(d.get(k), base.get(k)) for k, _ in METRICS]
            rows.append([label, split] + vals + diffs)
    return header, rows


def to_markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="eval_pr / cocomap の結果を集約して比較表を出力する")
    ap.add_argument("--results-dir", default="results", help="結果ファイルのディレクトリ（既定 results）")
    ap.add_argument("--out", help="出力プレフィックス（拡張子なし。.md と .csv を書く）")
    a = ap.parse_args()
    header, rows = build_table(a.results_dir)
    md = to_markdown(header, rows)
    print(md, end="")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out + ".md").write_text(md, encoding="utf-8")
        Path(a.out + ".csv").write_text(to_csv(header, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
