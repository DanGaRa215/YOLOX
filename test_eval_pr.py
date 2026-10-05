from eval_pr import compute_pr
names = ["a", "b"]
gts = {1: [(0, [0, 0, 10, 10]), (1, [20, 20, 30, 30])], 2: [(1, [0, 0, 10, 10])]}
preds = {1: [(0, 0.9, [0, 0, 10, 10]),        # TP a
             (1, 0.8, [20, 20, 30, 30]),      # TP b
             (1, 0.7, [20, 20, 30, 30]),      # duplicate -> FP b
             (0, 0.2, [50, 50, 60, 60])],     # below conf, ignored
         2: [(0, 0.9, [0, 0, 10, 10])]}       # wrong class: FP a, FN b
r = {x["class"]: x for x in compute_pr(gts, preds, names, 0.5, 0.3)}
assert (r["a"]["tp"], r["a"]["fp"], r["a"]["fn"]) == (1, 1, 0), r["a"]
assert (r["b"]["tp"], r["b"]["fp"], r["b"]["fn"]) == (1, 1, 1), r["b"]
o = r["all (micro)"]; assert (o["tp"], o["fp"], o["fn"]) == (2, 2, 1)
assert abs(o["precision"] - 0.5) < 1e-9 and abs(o["recall"] - 2 / 3) < 1e-9
print("ok")
