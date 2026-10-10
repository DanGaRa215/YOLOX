"""結果集約スクリプト: eval_pr.py の JSON と YOLOX tools/eval.py の cocomap ログから比較表を作る。

読むファイル（--results-dir 以下）:
  <exp名>_{val,test}.json          eval_pr.py の出力（クラス別 P/R と最後の "all (micro)"）
  <exp名>_{val,test}_cocomap.txt   YOLOX tools/eval.py の標準出力（pycocotools の表を含む）
出力: Markdown 表と CSV。ベースラインからの差分（ポイント）の列も付ける。

--aggregate-seeds を付けると、seed 違いの結果をまとめて平均 ± 標準偏差（本数 n つき）の表にする。
実験名の末尾の `_s<数字>` を取り除いた名前でグループ化し、末尾にそれが無い名前（例 yolox_s_windfarm）は
seed 1 として同じグループに入れる（yolox_s_windfarm と yolox_s_windfarm_s2 は同じグループ）。
標準偏差は標本標準偏差（n-1 で割る）。n=1 のときは標準偏差を出さず平均だけを示す。
"""
import argparse
import csv
import io
import json
import re
import statistics
import sys
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
        out[key + "_tp"], out[key + "_fp"], out[key + "_fn"] = r["tp"], r["fp"], r["fn"]
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


# ---- seed をまたいだ集計（--aggregate-seeds） ----

_RE_SEED = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")  # 末尾の _s<数字> だけ（名前の途中の _s_ には反応しない）
_RE_RESULT = re.compile(r"^(?P<exp>.+)_(?P<split>val|test)(?:_cocomap\.txt|\.json)$")  # *_preds.json は対象外

# 集計する指標: クラス別（cable tower / turbine / 全体）の P・R・TP・FP・FN と AP50・AP50:95。種類は fraction / count
_CLASSES = (("tower", "cable tower"), ("turbine", "turbine"), ("all", "全体"))
_PER_CLASS = (("p", "P", "fraction"), ("r", "R", "fraction"),
              ("tp", "TP", "count"), ("fp", "FP", "count"), ("fn", "FN", "count"))
SEED_METRICS = [(f"{c}_{m}", f"{cn} {mn}", kind) for c, cn in _CLASSES for m, mn, kind in _PER_CLASS] + [
    ("ap50", "AP50", "fraction"), ("ap5095", "AP50:95", "fraction")]


def split_seed(name):
    """実験名 -> (グループ名, seed)。末尾の _s<数字> を取り除く。無ければ seed 1 のグループとして扱う。"""
    m = _RE_SEED.match(name)
    if m:
        return m.group("base"), int(m.group("seed"))
    return name, 1


def discover_runs(results_dir):
    """results-dir にある実験名（eval_pr の json か cocomap がある名前）を、グループ名 -> [実験名, ...] にする。"""
    names = set()
    for p in Path(results_dir).glob("*"):
        m = _RE_RESULT.match(p.name)
        if m:
            names.add(m.group("exp"))
    groups = {}
    for n in sorted(names):
        groups.setdefault(split_seed(n)[0], []).append(n)
    for g, runs in groups.items():
        seeds = [split_seed(n)[1] for n in runs]
        if len(set(seeds)) != len(seeds):
            print(f"警告: グループ {g} に seed が重複する実験があります: {runs}（どちらも集計に含めます）", file=sys.stderr)
    return groups


def mean_std(values):
    """(平均, 標本標準偏差, n)。n=1 の標準偏差は None、n=0 は (None, None, 0)。"""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    return mean, (statistics.stdev(values) if n >= 2 else None), n


def fmt_stat(mean, std, n, kind, row_n):
    """平均 ± 標準偏差の文字列。n が行の本数 row_n より少ないときは (n=k) を付ける。値が無ければ「未実施」。"""
    if mean is None:
        return MISSING
    digits = 3 if kind == "fraction" else 1
    s = f"{mean:.{digits}f}" if std is None else f"{mean:.{digits}f} ± {std:.{digits}f}"
    return s + (f" (n={n})" if n != row_n else "")


def fmt_mean_diff(mean, base_mean, kind):
    """平均の差。fraction はポイント（×100、例 +3.2）、count は件数の差（例 +1.5）。どちらか無ければ空欄。"""
    if mean is None or base_mean is None:
        return ""
    return f"{(mean - base_mean) * (100 if kind == 'fraction' else 1):+.1f}"


def build_seed_table(results_dir):
    """seed 違いをまとめた (ヘッダ, 行のリスト)。行は グループ × split。ベースラインのグループとの平均の差も付ける。"""
    groups = discover_runs(results_dir)
    order = [g for g in EXPERIMENTS if g in groups] + sorted(g for g in groups if g not in EXPERIMENTS)
    # グループ × split -> 指標 -> 値のリスト、と本数
    stats, counts = {}, {}
    for g in order:
        for sp in SPLITS:
            cells = [load_cell(results_dir, n, sp) for n in groups[g]]
            cells = [c for c in cells if len(c) > 1]  # "counts" 以外の指標が 1 つも無い（未実施）の実験は数えない
            counts[(g, sp)] = len(cells)
            stats[(g, sp)] = {k: mean_std([c[k] for c in cells if k in c]) for k, _, _ in SEED_METRICS}
    header = (["実験", "split", "n"] + [h for _, h, _ in SEED_METRICS]
              + ["Δ" + h for _, h, _ in SEED_METRICS])
    rows = []
    for g in order:
        for sp in SPLITS:
            st, n = stats[(g, sp)], counts[(g, sp)]
            base = stats.get((BASELINE, sp))
            vals = [fmt_stat(*st[k], kind, n) for k, _, kind in SEED_METRICS]
            diffs = [fmt_mean_diff(st[k][0], base[k][0] if base else None, kind) for k, _, kind in SEED_METRICS]
            rows.append([EXPERIMENTS.get(g, g), sp, str(n)] + vals + diffs)
    return header, rows


def main():
    ap = argparse.ArgumentParser(description="eval_pr / cocomap の結果を集約して比較表を出力する")
    ap.add_argument("--results-dir", default="results", help="結果ファイルのディレクトリ（既定 results）")
    ap.add_argument("--out", help="出力プレフィックス（拡張子なし。.md と .csv を書く）")
    ap.add_argument("--aggregate-seeds", action="store_true",
                    help="seed 違い（末尾 _s<数字>、無ければ seed 1）をまとめ、平均 ± 標準偏差（n つき）の表を出す")
    a = ap.parse_args()
    header, rows = build_seed_table(a.results_dir) if a.aggregate_seeds else build_table(a.results_dir)
    md = to_markdown(header, rows)
    print(md, end="")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out + ".md").write_text(md, encoding="utf-8")
        Path(a.out + ".csv").write_text(to_csv(header, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
