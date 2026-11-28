"""Plot the two arms' sampled rewards and held-back development metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(input_path, output_path):
    data = json.loads(Path(input_path).read_text())
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    colors = ("#175b8a", "#c05237")
    for arm, color in zip(("original", "higher_cost"), colors):
        rec = data[arm]
        label = f"Penalty {rec['penalty']:.2f}"
        curve = rec["training"]["curve"]
        dev = rec["training"]["dev"]
        steps = [x["step"] for x in curve]
        development_steps = [x["step"] for x in dev]
        axes[0].plot(steps, [x["mean_reward"] for x in curve], color=color, marker="o",
                     ms=3, label=label)
        if arm == "higher_cost":
            axes[0].plot(steps, [x["mean_common_reward"] for x in curve], color=color,
                         linestyle="--", label="Penalty 0.20, rescored at 0.02")
        axes[1].plot(development_steps, [100*x["counts"]["success"]/x["prompts"] for x in dev],
                     color=color, marker="o", ms=3)
        axes[2].plot(development_steps, [x["counts"]["unnecessary"] for x in dev],
                     color=color, marker="o", ms=3)
        best = rec["training"]["best_step"]
        axes[1].axvline(best, color=color, linestyle=":", alpha=.65)
    axes[0].set(ylabel="Mean sampled reward", title="Training reward over each 32-step block")
    axes[1].set(ylabel="Development success (%)", title="Development success and selected checkpoints")
    axes[2].set(ylabel="Unneeded reads", title="Unnecessary reads on 96 sufficient development inputs",
                xlabel="Optimizer step")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
    limit = data["original"]["training"]["curve"][-1]["step"]
    axes[2].set_xticks(range(0,limit+1,64))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args=p.parse_args()
    main(args.input,args.output)
