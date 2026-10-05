#!/usr/bin/env python3
"""Convert YOLO txt dataset (Roboflow) to COCO format for YOLOX.

Output layout (under --out):
  train2017/ val2017/ test2017/           images (copy or symlink)
  annotations/instances_{train,val,test}2017.json

Label rows: 5 columns (cls cx cy w h) = bbox; >5 columns (cls x1 y1 x2 y2 ...) = polygon
-> converted to bbox by min/max of vertices. Decided by column count.
category_id is 1-based (1: cable tower, 2: turbine). YOLOX COCODataset maps via
sorted(coco.getCatIds()).index(category_id), so either base works; 1-based is the COCO convention.
"""
import argparse, json, os, shutil, sys
from collections import Counter
from pathlib import Path
from PIL import Image

SPLITS = {"train": "train2017", "valid": "val2017", "test": "test2017"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_label(path):
    """Yield (cls, x_min, y_min, x_max, y_max, is_polygon) normalized."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        cls = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(p) == 5:
            cx, cy, w, h = v
            out.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, False))
        elif len(p) >= 7 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            out.append((cls, min(xs), min(ys), max(xs), max(ys), True))
        else:
            raise ValueError(f"unexpected column count {len(p)} in {path}")
    return out


def convert_split(src, split, dst_name, out, names, link, stats_all):
    img_dir, lbl_dir = src / split / "images", src / split / "labels"
    dst_img = out / dst_name
    dst_img.mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    st = Counter()
    per_cls = Counter()
    ann_id = 1
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    for img_id, f in enumerate(files, 1):
        with Image.open(f) as im:
            W, H = im.size
        dst = dst_img / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            os.symlink(f.resolve(), dst)
        else:
            shutil.copy2(f, dst)
        images.append({"id": img_id, "file_name": f.name, "width": W, "height": H})
        for cls, x1, y1, x2, y2, poly in parse_label(lbl_dir / (f.stem + ".txt")):
            st["rows"] += 1
            st["polygon_rows"] += poly
            if not (0 <= cls < len(names)):
                st["bad_class"] += 1
                continue
            px1, py1, px2, py2 = x1 * W, y1 * H, x2 * W, y2 * H
            if px1 < 0 or py1 < 0 or px2 > W or py2 > H:
                st["clipped"] += 1
            px1, px2 = min(max(px1, 0), W), min(max(px2, 0), W)
            py1, py2 = min(max(py1, 0), H), min(max(py2, 0), H)
            w, h = px2 - px1, py2 - py1
            if w <= 0 or h <= 0:
                st["dropped_degenerate"] += 1
                continue
            anns.append({"id": ann_id, "image_id": img_id, "category_id": cls + 1,
                         "bbox": [round(px1, 2), round(py1, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0,
                         "segmentation": [], "polygon_src": bool(poly)})
            ann_id += 1
            per_cls[names[cls]] += 1
            st["kept_polygon"] += poly
    coco = {"info": {}, "licenses": [], "images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(names)]}
    jp = out / "annotations" / f"instances_{dst_name}.json"
    jp.write_text(json.dumps(coco))
    st.update(images=len(images), annotations=len(anns))
    print(f"[{split} -> {dst_name}] images={len(images)} annotations={len(anns)} per_class={dict(per_cls)}")
    print(f"   label_rows={st['rows']} polygon_rows={st['polygon_rows']} polygon_kept={st['kept_polygon']} "
          f"clipped={st['clipped']} dropped_degenerate={st['dropped_degenerate']} bad_class={st['bad_class']}")
    stats_all[split] = {**st, "per_class": dict(per_cls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root containing train/valid/test and data.yaml")
    ap.add_argument("--out", required=True, help="output root, e.g. datasets/windfarm")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--names", nargs="+", default=["cable tower", "turbine"])
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    stats = {}
    for s, d in SPLITS.items():
        if (src / s).exists():
            convert_split(src, s, d, out, a.names, a.mode == "symlink", stats)
        else:
            print(f"skip missing split {s}", file=sys.stderr)
    (out / "annotations" / "conversion_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
