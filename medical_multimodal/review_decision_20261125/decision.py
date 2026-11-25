from collections import Counter
from statistics import mean

POLICIES = ("stop", "review", "small_box", "large_box", "near_edge", "visual")


def decide(policy, row):
    if not row["first_valid"]:
        return False
    cues = row["cues"]
    return {"stop": False, "review": True, "small_box": cues["area"] <= 0.15,
            "large_box": cues["area"] >= 0.4, "near_edge": cues["margin"] <= 0.02,
            "visual": cues["visual"] in ("LOOSE", "CLIPPED", "MISSED")}[policy]


def metrics(rows, choices, policies):
    scores = [r["review" if c else "stop"] for r, c in zip(rows, choices)]
    calls = [1 + int(p == "visual" and r["first_valid"]) + (r["review_calls"] if c else 0)
             for r, c, p in zip(rows, choices, policies)]
    return {"mean_iou": mean(scores), "success": sum(s >= 0.5 for s in scores),
            "reviews": sum(choices), "mean_vlm_calls": mean(calls),
            "valid": sum(r["review_valid" if c else "first_valid"] for r,c in zip(rows,choices))}


def policy_metrics(rows, policy):
    return metrics(rows, [decide(policy, r) for r in rows], [policy] * len(rows))


def select(rows):
    def rank(policy):
        m = policy_metrics(rows, policy)
        return round(m["mean_iou"], 12), -m["mean_vlm_calls"]
    return max(POLICIES, key=rank)


def summarize(rows):
    arms = {p: policy_metrics(rows, p) for p in POLICIES}
    delta = [r["review"] - r["stop"] for r in rows]
    oracle = mean(max(r["stop"], r["review"]) for r in rows)
    return {"n": len(rows), "policies": arms, "gain_at_least_005": sum(d >= .05 for d in delta),
            "loss_at_least_005": sum(d <= -.05 for d in delta),
            "within_005": sum(abs(d) < .05 for d in delta),
            "success_gains": sum(r["stop"] < .5 <= r["review"] for r in rows),
            "success_losses": sum(r["review"] < .5 <= r["stop"] for r in rows),
            "oracle_iou": oracle, "headroom": oracle - max(arms[p]["mean_iou"] for p in ("stop", "review")),
            "visual_labels": dict(Counter(r["cues"]["visual"] for r in rows if r["first_valid"]))}


def training_summary(rows):
    info = summarize(rows)
    loo = [select(rows[:i] + rows[i+1:]) for i in range(len(rows))]
    info.update(frozen_policy=select(rows), loo_choices=dict(Counter(loo)),
                loo_metrics=metrics(rows, [decide(p,r) for p,r in zip(loo,rows)], loo))
    info["open_development"] = bool(info["gain_at_least_005"] and info["loss_at_least_005"] and info["headroom"] >= .02)
    return info
