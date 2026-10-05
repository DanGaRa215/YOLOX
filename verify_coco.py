"""Validate converted COCO JSON and draw sample boxes into vis_check/."""
import json, random, sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/windfarm")
out = Path("vis_check"); out.mkdir(exist_ok=True)
COL = {1: "red", 2: "lime"}
for split in ["train2017", "val2017", "test2017"]:
    d = json.load(open(root / "annotations" / f"instances_{split}.json"))
    ids = {i["id"]: i for i in d["images"]}
    assert len(ids) == len(d["images"])
    c = Counter(a["category_id"] for a in d["annotations"])
    bad = 0
    for a in d["annotations"]:
        im = ids[a["image_id"]]; x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > im["width"] + 1e-6 or y + h > im["height"] + 1e-6: bad += 1
    sizes = Counter((i["width"], i["height"]) for i in d["images"])
    areas = {k: sorted(a["area"] for a in d["annotations"] if a["category_id"] == k) for k in (1, 2)}
    med = {k: round(v[len(v)//2] ** .5, 1) for k, v in areas.items() if v}
    print(split, "images", len(ids), "anns", len(d["annotations"]), dict(c), "out_of_bounds", bad, "sizes", dict(sizes), "median sqrt-area(px)", med)
    by = {}
    for a in d["annotations"]: by.setdefault(a["image_id"], []).append(a)
    random.seed(0)
    ct = [i for i, v in by.items() if any(a["category_id"] == 1 for a in v)]
    pick = random.sample(ct, 3) + random.sample(list(ids), 1)
    for i in pick:
        im = Image.open(root / split / ids[i]["file_name"]).convert("RGB"); dr = ImageDraw.Draw(im)
        for a in by.get(i, []):
            x, y, w, h = a["bbox"]; dr.rectangle([x, y, x + w, y + h], outline=COL[a["category_id"]], width=4)
        im.thumbnail((1280, 720)); im.save(out / f"{split}_{i}.jpg")
