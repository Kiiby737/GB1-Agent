#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot M0-AL independent-test metrics as a one-page editable vector PDF.

Run:
    python plot_m0_al_performance.py

Optional:
    python plot_m0_al_performance.py --input /path/to/metrics.json \
        --output /path/to/output.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


DEFAULT_INPUT = Path(
    "/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/5.test/"
    "2.model_eval/m0_al_independent_test/m0_al_test_metrics.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/figure/"
    "m0_al_independent_test_performance.pdf"
)


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = ("n_test", "pearson", "spearman", "mse", "rmse", "topk")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing JSON fields: {missing}")
    if not isinstance(data["topk"], list) or not data["topk"]:
        raise ValueError("'topk' must be a non-empty list.")
    return data


def plot_metrics(data, output_path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    n_test = int(data["n_test"])
    corr_names = ["Pearson", "Spearman"]
    corr_values = [float(data["pearson"]), float(data["spearman"])]

    topk = sorted(data["topk"], key=lambda x: int(x["k"]))
    topk_labels = [f"Top-{int(x['k'])}" for x in topk]
    topk_rates = [float(x["topk_overlap_rate"]) for x in topk]
    topk_counts = [
        (int(x["topk_overlap_count"]), int(x["k"])) for x in topk
    ]

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.15, 0.85],
        hspace=0.48,
        wspace=0.30,
    )

    # A. Correlation metrics
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(corr_names))
    bars = ax.barh(y, corr_values, height=0.52)
    ax.set_yticks(y)
    ax.set_yticklabels(corr_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("A. Correlation")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, corr_values):
        ax.text(
            value + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
        )
    ax.text(
        0, -0.30,
        "Higher values indicate stronger correlation.",
        transform=ax.transAxes, fontsize=8.5, va="top",
    )

    # B. Error metrics as separate cards. MSE and RMSE have different units,
    # so they are deliberately not compared on a shared bar-chart axis.
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B. Prediction error")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for y0, name, value in [
        (0.54, "MSE", float(data["mse"])),
        (0.08, "RMSE", float(data["rmse"])),
    ]:
        ax.add_patch(Rectangle(
            (0.06, y0), 0.88, 0.34,
            fill=False, linewidth=1.0,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(
            0.12, y0 + 0.23, name,
            transform=ax.transAxes, fontsize=10, va="center",
        )
        ax.text(
            0.88, y0 + 0.16, f"{value:.4f}",
            transform=ax.transAxes, fontsize=19,
            fontweight="bold", ha="right", va="center",
        )
    ax.text(
        0.06, 0.00,
        "Lower values indicate smaller prediction errors.",
        transform=ax.transAxes, fontsize=8.5, va="top",
    )

    # C. Top-k overlap
    ax = fig.add_subplot(gs[1, :])
    y = np.arange(len(topk_labels))
    bars = ax.barh(y, topk_rates, height=0.48)
    ax.set_yticks(y)
    ax.set_yticklabels(topk_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Top-k overlap rate")
    ax.set_title("C. Top-k retrieval performance")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, rate, (count, k) in zip(bars, topk_rates, topk_counts):
        ax.text(
            rate + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}  ({count}/{k})",
            va="center",
            ha="left",
        )

    fig.suptitle(
        f"M0-AL independent test performance  |  Test samples: n = {n_test:,}",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5, 0.015,
        "Top-k overlap rate = number of reference top-k variants recovered / k.",
        ha="center", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"PDF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize M0-AL independent-test performance."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    metrics = load_metrics(args.input)
    plot_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
