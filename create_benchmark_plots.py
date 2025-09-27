#!/usr/bin/env python3
"""
Create visualization plots for KERAG benchmark results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_benchmark_data():
    """Load the latest benchmark results."""
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        print("No benchmark results found!")
        return None

    # Find the latest detailed results file
    detailed_files = list(results_dir.glob("detailed_results_*.json"))
    if not detailed_files:
        print("No detailed results files found!")
        return None

    latest_file = max(detailed_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def create_accuracy_comparison_plot(df):
    """Create accuracy comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Group by method and calculate averages
    method_stats = (
        df.groupby("method")
        .agg(
            {
                "accuracy_score": "mean",
                "coverage_score": "mean",
                "hallucination_score": "mean",
                "response_time": "mean",
                "reasoning_quality": "mean",
            }
        )
        .round(3)
    )

    # Accuracy and Coverage comparison
    metrics = ["accuracy_score", "coverage_score"]
    x = np.arange(len(metrics))
    width = 0.35

    kerag_values = [
        method_stats.loc["KERAG", metric] if "KERAG" in method_stats.index else 0
        for metric in metrics
    ]
    traditional_values = [
        (
            method_stats.loc["Traditional_RAG", metric]
            if "Traditional_RAG" in method_stats.index
            else 0
        )
        for metric in metrics
    ]

    ax1.bar(
        x - width / 2, kerag_values, width, label="KERAG", alpha=0.8, color="#2E86AB"
    )
    ax1.bar(
        x + width / 2,
        traditional_values,
        width,
        label="Traditional RAG",
        alpha=0.8,
        color="#A23B72",
    )

    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy & Coverage Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Accuracy", "Coverage"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (kerag, trad) in enumerate(zip(kerag_values, traditional_values)):
        ax1.text(i - width / 2, kerag + 0.01, f"{kerag:.3f}", ha="center", va="bottom")
        ax1.text(i + width / 2, trad + 0.01, f"{trad:.3f}", ha="center", va="bottom")

    # Response Time comparison
    response_times = [
        (
            method_stats.loc["KERAG", "response_time"]
            if "KERAG" in method_stats.index
            else 0
        ),
        (
            method_stats.loc["Traditional_RAG", "response_time"]
            if "Traditional_RAG" in method_stats.index
            else 0
        ),
    ]
    methods = ["KERAG", "Traditional RAG"]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax2.bar(methods, response_times, color=colors, alpha=0.8)
    ax2.set_ylabel("Response Time (seconds)")
    ax2.set_title("Response Time Comparison")
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, time in zip(bars, response_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "benchmark_results/accuracy_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def create_question_analysis_plot(df):
    """Create question-by-question analysis plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter for Traditional RAG results (since KERAG had errors)
    trad_df = df[df["method"] == "Traditional_RAG"].copy()

    if len(trad_df) == 0:
        print("No Traditional RAG results to plot!")
        return

    # Create question complexity categories
    trad_df["question_length"] = trad_df["question"].str.len()
    trad_df["complexity"] = pd.cut(
        trad_df["question_length"],
        bins=[0, 50, 100, 200],
        labels=["Simple", "Medium", "Complex"],
    )

    # Plot accuracy by complexity
    complexity_stats = (
        trad_df.groupby("complexity")["accuracy_score"].agg(["mean", "std"]).fillna(0)
    )

    x = np.arange(len(complexity_stats))
    bars = ax.bar(
        x,
        complexity_stats["mean"],
        yerr=complexity_stats["std"],
        capsize=5,
        alpha=0.8,
        color="#A23B72",
    )

    ax.set_xlabel("Question Complexity")
    ax.set_ylabel("Average Accuracy Score")
    ax.set_title("Traditional RAG Performance by Question Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels(complexity_stats.index)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, complexity_stats["mean"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "benchmark_results/question_complexity_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_metrics_radar_plot(df):
    """Create radar plot comparing all metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Group by method
    method_stats = (
        df.groupby("method")
        .agg(
            {
                "accuracy_score": "mean",
                "coverage_score": "mean",
                "reasoning_quality": "mean",
            }
        )
        .fillna(0)
    )

    # Calculate hallucination as inverse (lower is better)
    method_stats["hallucination_inverse"] = (
        1 - df.groupby("method")["hallucination_score"].mean()
    )

    # Normalize response time (inverse, lower is better)
    response_times = df.groupby("method")["response_time"].mean()
    max_time = response_times.max()
    method_stats["response_time_inverse"] = 1 - (response_times / max_time)

    metrics = [
        "accuracy_score",
        "coverage_score",
        "reasoning_quality",
        "hallucination_inverse",
        "response_time_inverse",
    ]
    metric_labels = [
        "Accuracy",
        "Coverage",
        "Reasoning",
        "Low Hallucination",
        "Fast Response",
    ]

    # Convert to radians
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ["#2E86AB", "#A23B72"]
    methods = ["KERAG", "Traditional_RAG"]

    for i, method in enumerate(methods):
        if method in method_stats.index:
            values = [method_stats.loc[method, metric] for metric in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Comparison Radar Chart", size=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results/radar_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_summary_table(df):
    """Create a summary table of results."""
    method_stats = (
        df.groupby("method")
        .agg(
            {
                "accuracy_score": ["mean", "std"],
                "coverage_score": ["mean", "std"],
                "hallucination_score": ["mean", "std"],
                "response_time": ["mean", "std"],
                "reasoning_quality": ["mean", "std"],
            }
        )
        .round(3)
    )

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Metric':<20} {'KERAG':<15} {'Traditional RAG':<15} {'Improvement':<15}")
    print("-" * 80)

    for metric in [
        "accuracy_score",
        "coverage_score",
        "hallucination_score",
        "response_time",
        "reasoning_quality",
    ]:
        kerag_val = (
            method_stats.loc["KERAG", (metric, "mean")]
            if "KERAG" in method_stats.index
            else 0
        )
        trad_val = (
            method_stats.loc["Traditional_RAG", (metric, "mean")]
            if "Traditional_RAG" in method_stats.index
            else 0
        )

        if metric == "hallucination_score":
            # For hallucination, lower is better
            improvement = (
                ((trad_val - kerag_val) / trad_val * 100) if trad_val > 0 else 0
            )
        elif metric == "response_time":
            # For response time, lower is better
            improvement = (
                ((trad_val - kerag_val) / trad_val * 100) if trad_val > 0 else 0
            )
        else:
            # For other metrics, higher is better
            improvement = (
                ((kerag_val - trad_val) / trad_val * 100) if trad_val > 0 else 0
            )

        print(
            f"{metric.replace('_', ' ').title():<20} {kerag_val:<15.3f} {trad_val:<15.3f} {improvement:>+13.1f}%"
        )

    print("=" * 80)


def main():
    """Main function to create all plots."""
    print("📊 Creating Benchmark Visualization Plots")
    print("=" * 50)

    # Load data
    df = load_benchmark_data()
    if df is None:
        return

    print(f"Loaded {len(df)} benchmark results")
    print(f"Methods: {df['method'].unique()}")

    # Create plots directory
    Path("benchmark_results").mkdir(exist_ok=True)

    # Create all plots
    print("\n📈 Creating accuracy comparison plot...")
    create_accuracy_comparison_plot(df)

    print("📈 Creating question complexity analysis...")
    create_question_analysis_plot(df)

    print("📈 Creating radar comparison chart...")
    create_metrics_radar_plot(df)

    # Create summary table
    create_summary_table(df)

    print(f"\n✅ All plots saved to: benchmark_results/")
    print("   - accuracy_comparison.png")
    print("   - question_complexity_analysis.png")
    print("   - radar_comparison.png")


if __name__ == "__main__":
    main()


