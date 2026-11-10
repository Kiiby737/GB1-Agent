#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仅绘制方案 A：Original GB1 与 FLIP-GB1 sampled 的 Fitness 分段占比柱状图。

输出：
    distribution_visualization/gb1_fitness_distribution_A.pdf

PDF 为矢量格式，可在 Adobe Illustrator、Inkscape 等软件中编辑。

运行：
    python plot_gb1_fitness_scheme_A.py

也可以指定输入和输出路径：
    python plot_gb1_fitness_scheme_A.py \
        --sampled /path/to/sampled.csv \
        --full /path/to/gb1_2016_149361_seq.csv \
        --output /path/to/gb1_fitness_distribution_A.pdf
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_SAMPLED = Path(
    "/mnt/disk2/shiming/project/ZJU_agent_project/ver4.4/1.data/sampled.csv"
)
DEFAULT_FULL = Path(
    "/mnt/disk2/shiming/project/ZJU_agent_project/ver4.4/1.data/gb1_2016_149361_seq.csv"
)
DEFAULT_OUTPUT = (
    Path("/mnt/disk2/shiming/project/ZJU_agent_project/ver4.4/1.data")
    / "distribution_visualization"
    / "gb1_fitness_distribution_A.pdf"
)

# 分段边界：
# Fitness <= 0.5
# 0.5 < Fitness <= 1.0
# 1.0 < Fitness <= 2.0
# 2.0 < Fitness <= 4.0
# Fitness > 4.0
BIN_EDGES = [-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf]
BIN_LABELS = [
    "≤ 0.5",
    "(0.5, 1.0]",
    "(1.0, 2.0]",
    "(2.0, 4.0]",
    "> 4.0",
]


def find_fitness_column(df, file_path):
    """大小写不敏感地识别 Fitness 列。"""
    normalized = {
        str(column).strip().lower().replace("_", "").replace("-", ""): column
        for column in df.columns
    }

    candidates = ["fitness", "target", "score", "logfitness"]
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        f"无法在 {file_path} 中识别 Fitness 列。\n"
        f"当前 CSV 列名：{list(df.columns)}"
    )


def load_fitness(file_path, dataset_name):
    """读取并清理 Fitness 数据。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    df = pd.read_csv(file_path)
    fitness_column = find_fitness_column(df, file_path)

    fitness = pd.to_numeric(df[fitness_column], errors="coerce")
    fitness = fitness.replace([np.inf, -np.inf], np.nan).dropna()

    if fitness.empty:
        raise ValueError(f"{dataset_name} 没有有效的 Fitness 数值。")

    print(
        f"{dataset_name}: n={len(fitness):,}, "
        f"Fitness range=[{fitness.min():.4f}, {fitness.max():.4f}]"
    )
    return fitness


def calculate_bin_percentages(fitness):
    """计算每个 Fitness 区间的样本数和百分比。"""
    categories = pd.cut(
        fitness,
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=True,
        include_lowest=True,
        ordered=True,
    )

    counts = (
        categories.value_counts(sort=False)
        .reindex(BIN_LABELS, fill_value=0)
        .astype(int)
    )
    percentages = counts / len(fitness) * 100.0

    return counts, percentages


def plot_scheme_a(sampled_fitness, full_fitness, output_path):
    """只生成一张 Fitness 分段占比柱状图，并保存为矢量 PDF。"""
    full_counts, full_percentages = calculate_bin_percentages(full_fitness)
    sampled_counts, sampled_percentages = calculate_bin_percentages(sampled_fitness)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    x = np.arange(len(BIN_LABELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    bars_full = ax.bar(
        x - width / 2,
        full_percentages.to_numpy(),
        width,
        label=f"Original GB1 (n={len(full_fitness):,})",
    )
    bars_sampled = ax.bar(
        x + width / 2,
        sampled_percentages.to_numpy(),
        width,
        label=f"FLIP-GB1 sampled (n={len(sampled_fitness):,})",
    )

    # 在柱顶标注百分比；样本量放在图例中，避免标签过于拥挤。
    ax.bar_label(
        bars_full,
        labels=[f"{v:.1f}%" if v > 0 else "" for v in full_percentages],
        padding=3,
        fontsize=8,
    )
    ax.bar_label(
        bars_sampled,
        labels=[f"{v:.1f}%" if v > 0 else "" for v in sampled_percentages],
        padding=3,
        fontsize=8,
    )

    max_percentage = max(
        float(full_percentages.max()),
        float(sampled_percentages.max()),
    )
    ax.set_ylim(0, max(10, max_percentage * 1.20))

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Fitness range")
    ax.set_ylabel("Proportion of variants")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    ax.set_title("Fitness distribution of Original GB1 and FLIP-GB1")
    ax.legend(frameon=False, loc="upper right")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # 去除上、右边框，减少视觉干扰。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPDF saved: {output_path}")

    # 同时在终端打印分段统计，便于核对图中数据。
    summary = pd.DataFrame(
        {
            "Fitness range": BIN_LABELS,
            "Original GB1 count": full_counts.to_numpy(),
            "Original GB1 (%)": full_percentages.to_numpy(),
            "FLIP-GB1 sampled count": sampled_counts.to_numpy(),
            "FLIP-GB1 sampled (%)": sampled_percentages.to_numpy(),
        }
    )

    print("\nFitness-bin summary:")
    print(summary.to_string(index=False, formatters={
        "Original GB1 (%)": "{:.2f}".format,
        "FLIP-GB1 sampled (%)": "{:.2f}".format,
    }))


def main():
    parser = argparse.ArgumentParser(
        description="Plot only the Fitness-bin proportion chart (Scheme A)."
    )
    parser.add_argument("--sampled", type=Path, default=DEFAULT_SAMPLED)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sampled_fitness = load_fitness(args.sampled, "FLIP-GB1 sampled")
    full_fitness = load_fitness(args.full, "Original GB1")

    plot_scheme_a(
        sampled_fitness=sampled_fitness,
        full_fitness=full_fitness,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
