from eval_pr import compute_pr
names = ["a", "b"]
gts = {1: [(0, [0, 0, 10, 10]), (1, [20, 20, 30, 30])], 2: [(1, [0, 0, 10, 10])]}
preds = {1: [(0, 0.9, [0, 0, 10, 10]),        # a の TP
             (1, 0.8, [20, 20, 30, 30]),      # b の TP
             (1, 0.7, [20, 20, 30, 30]),      # 重複 -> b の FP
             (0, 0.2, [50, 50, 60, 60])],     # conf 未満のため無視
         2: [(0, 0.9, [0, 0, 10, 10])]}       # クラス違い: a の FP, b の FN
r = {x["class"]: x for x in compute_pr(gts, preds, names, 0.5, 0.3)}
assert (r["a"]["tp"], r["a"]["fp"], r["a"]["fn"]) == (1, 1, 0), r["a"]
assert (r["b"]["tp"], r["b"]["fp"], r["b"]["fn"]) == (1, 1, 1), r["b"]
o = r["all (micro)"]; assert (o["tp"], o["fp"], o["fn"]) == (2, 2, 1)
assert abs(o["precision"] - 0.5) < 1e-9 and abs(o["recall"] - 2 / 3) < 1e-9
print("ok")
