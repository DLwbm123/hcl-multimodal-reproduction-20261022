from collections import Counter
from statistics import mean

RULES = ("default", "review", "low_margin", "disagree", "both")


def choose(rule, r):
    if not r["available"]:
        return False
    margin = r["margin"] <= .05
    disagree = r["agreement"] <= .75
    return {"default": False, "review": True, "low_margin": margin, "disagree": disagree, "both": margin and disagree}[rule]


def evaluate(rows, rules):
    flags = [choose(p,r) for p,r in zip(rules,rows)]
    results = [r["review" if f else "default"] for r,f in zip(rows,flags)]
    return {"dice": mean(v["score"] for v in results), "success": sum(v["success"] for v in results),
            "reviews": sum(flags), "valid": sum(v["valid"] for v in results),
            "mean_vlm_calls": 1 + mean(flags), "mean_sam_calls": mean(r["available"] for r in rows)}


def fit(rows):
    results = {p:evaluate(rows,[p]*len(rows)) for p in RULES}
    return max(RULES,key=lambda p:(round(results[p]["dice"],12),-results[p]["reviews"]))


def summarize(rows):
    policies = {p:evaluate(rows,[p]*len(rows)) for p in RULES}
    diff = [r["review"]["score"]-r["default"]["score"] for r in rows]
    fixed = max(policies[p]["dice"] for p in ("default","review"))
    oracle = mean(max(r["review"]["score"],r["default"]["score"]) for r in rows)
    return {"n":len(rows), "policies":policies,"two_action_oracle":oracle,"headroom":oracle-fixed,
            "three_mask_oracle":mean(r["oracle"] for r in rows), "material_gain":sum(v>=.05 for v in diff),
            "material_loss":sum(v<=-.05 for v in diff), "within_band":sum(abs(v)<.05 for v in diff),
            "success_gain":sum(r["review"]["success"] and not r["default"]["success"] for r in rows),
            "success_loss":sum(r["default"]["success"] and not r["review"]["success"] for r in rows),
            "switches":sum(r["switched"] for r in rows),"review_parse":dict(Counter(r["parse"] for r in rows))}
