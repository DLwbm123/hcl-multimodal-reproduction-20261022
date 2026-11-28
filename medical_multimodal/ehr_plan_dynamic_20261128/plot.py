"""Render the development trade-off and the guarded holdout comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    folder = Path(__file__).resolve().parent
    result = json.loads((folder / "RESULT.json").read_text())
    history = json.loads((folder.parent / "ehr_plan_reward_20261128" / "RESULT.json").read_text())
    historical_dev = next(row for row in history["higher_cost"]["training"]["dev"]
                          if row["step"] == 384)
    base, age = result["dynamic"], result["dynamic_with_age"]
    blue, orange, slate, green = "#16699f", "#b94e25", "#4c5364", "#067960"

    fig, (dev, guard) = plt.subplots(1, 2, figsize=(13.6, 5.1),
                                    gridspec_kw={"width_ratios": (1.45, 1)},
                                    layout="constrained")
    for arm, color, name in ((base, blue, "Dynamic · original receipt"),
                             (age, orange, "Dynamic · explicit age")):
        training = arm["training"]
        rows = training["dev"]
        missed = [row["counts"]["premature"] for row in rows]
        excess = [row["counts"]["unnecessary"] for row in rows]
        dev.plot(missed, excess, "o-", ms=4.5, lw=1.25, color=color, alpha=.82, label=name)
        chosen = next(i for i, row in enumerate(rows) if row["step"] == training["best_step"])
        for i, marker, offset in ((chosen, "D", (8, -11)), (-1, "s", (7, 4))):
            dev.scatter(missed[i], excess[i], marker=marker, color=color, edgecolor="white",
                        linewidth=.8, s=66, zorder=5)
            dev.annotate(str(rows[i]["step"]), (missed[i], excess[i]), xytext=offset,
                         textcoords="offset points", color=color, fontsize=9, weight="bold")
        dev.annotate("0", (missed[0], excess[0]), xytext=(0, 6), ha="center",
                     textcoords="offset points", color=color, fontsize=8)

    counts = historical_dev["counts"]
    dev.scatter(counts["premature"], counts["unnecessary"], marker="x", color=slate,
                s=76, linewidth=1.7, zorder=6, label="Historical 0.20 · step 384")
    dev.scatter(0, 0, marker="*", color=green, s=118, zorder=6, label="Deterministic rule")
    dev.set(title="Development checkpoints · without guard", xlabel="Missed necessary reads (of 96)",
            ylabel="Unnecessary reads (of 96)", xlim=(-4, 102), ylim=(-4, 54),
            xticks=[0, 10, 20, 30, 50, 70, 90])
    handles, labels = dev.get_legend_handles_labels()
    for marker, label in (("D", "Original dev choice"), ("s", "Training endpoint")):
        handles.append(Line2D([], [], color=slate, marker=marker, linestyle="None", markersize=6))
        labels.append(label)
    dev.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.4)

    rule = base["test"]["rule"]["counts"]
    rule_calls = rule["tool_calls"] + rule["extra_reads"]
    policies = (
        ("Frozen + guard", base["test"]["frozen"], "#828995"),
        ("Historical 0.20 + guard", result["historical_cost_020"]["test"]["endpoint"], slate),
        ("Dynamic endpoint + guard\nOriginal receipt", base["test"]["endpoint"], blue),
        ("Dynamic endpoint + guard\nExplicit age", age["test"]["endpoint"], orange),
        ("Deterministic rule", base["test"]["rule"], green),
    )
    for i, (name, policy, color) in enumerate(policies):
        counts = policy["counts"]
        calls = counts["tool_calls"] + counts["extra_reads"]
        excess = calls - rule_calls
        assert counts["guarded_success"] == 512 and excess == counts["unnecessary"]
        guard.barh(i, excess, height=.52, color=color)
        guard.scatter(excess, i, s=24, color=color, zorder=3)
        guard.text(excess + 4, i, f"{calls} total", va="center", color=color, fontsize=9, weight="bold")
    guard.set(yticks=range(len(policies)), yticklabels=[name for name, _, _ in policies],
              title="Holdout · all systems solve 512/512", xlabel=f"Calls above the rule ({rule_calls} calls)",
              xlim=(-4, 159), xticks=[0, 32, 64, 96, 128])
    guard.invert_yaxis()
    guard.tick_params(axis="y", length=0)
    for axis, direction in ((dev, "both"), (guard, "x")):
        axis.set_axisbelow(True)
        axis.grid(axis=direction, color="#d8dfe5", linewidth=.65)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "pagination_tradeoff.png", dpi=190, facecolor="white")


if __name__ == "__main__":
    main()
