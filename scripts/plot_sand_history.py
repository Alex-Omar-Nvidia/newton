# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot valid candidate errors and the best-so-far sand fit for one run."""

import argparse
import json
from pathlib import Path


def plot_history(run_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_directory = Path(run_directory)
    history = json.loads((run_directory / "history.json").read_text())
    if not isinstance(history, list) or not history:
        raise ValueError("Run history has no candidates")

    valid = [trial for trial in history if trial.get("valid") is True and trial.get("error") is not None]
    invalid = [trial["evaluation"] for trial in history if trial not in valid]
    best = min(valid, key=lambda trial: trial["error"], default=None)
    xs = [trial["evaluation"] for trial in valid]
    errors = [100 * trial["error"] for trial in valid]
    best_so_far = []
    current_best = float("inf")
    for trial in history:
        if trial in valid:
            current_best = min(current_best, 100 * trial["error"])
        best_so_far.append(current_best if current_best < float("inf") else float("nan"))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.2)
    ax.scatter(xs, errors, color="#8aa5bb", s=26, label="Valid candidate error", zorder=3)
    ax.step(
        [trial["evaluation"] for trial in history],
        best_so_far,
        where="post",
        color="#007d80",
        linewidth=2.2,
        label="Best valid error so far",
    )
    ax.axhline(10, color="#ba7428", linestyle="--", linewidth=1.3, label="10% per-case target reference")
    ax.scatter(
        invalid,
        [-0.08] * len(invalid),
        transform=ax.get_xaxis_transform(),
        marker="x",
        color="#b54749",
        clip_on=False,
        label="Invalid candidate",
    )
    if best:
        ax.scatter([best["evaluation"]], [100 * best["error"]], color="#007d80", s=100, edgecolor="white", zorder=4)
        ax.annotate(
            f"Best: #{best['evaluation']} · {100 * best['error']:.2f}%",
            (best["evaluation"], 100 * best["error"]),
            xytext=(8, 12),
            textcoords="offset points",
            color="#00696c",
        )
    ax.set(
        title=f"Sand system identification — {len(history)} candidates",
        xlabel="Candidate evaluation",
        ylabel="Combined relative RMS error (%)",
        xlim=(0.5, max(trial["evaluation"] for trial in history) + 0.5),
        ylim=(0, max([10, *errors]) * 1.15),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    output = run_directory / "plots" / "optimization_error.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(plot_history(args.run_directory))
