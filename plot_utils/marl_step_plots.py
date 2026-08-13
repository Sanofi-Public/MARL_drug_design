#!/usr/bin/env python3
"""
Step-wise plotting functions for MARL component analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marl_plot_utils import rolling_mean, ensure_dir


def plot_unique_molecules_by_step(unique_df, window, save_dir):
    """
    Plot cumulative unique targets and non-targets over steps.
    Uses dual y-axes for better readability (targets on left, non-targets on right).
    """
    ensure_dir(save_dir)
    
    if unique_df.empty:
        print("No unique molecule data available for step-based plotting.")
        return None
    
    x = unique_df["Step"].to_numpy()
    final_step = x[-1] if len(x) > 0 else 0
    
    # Create single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Left axis: Targets (green)
    y_targets = unique_df["unique_targets"].to_numpy()
    y_targets_ma = rolling_mean(y_targets, window)
    
    line1 = ax1.plot(x, y_targets, alpha=0.2, color="green", linewidth=1)
    line2, = ax1.plot(x, y_targets_ma, alpha=0.9, label=f"Unique Targets (MA {window})", color="green", linewidth=2.5)
    
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Unique Targets", color="green", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="green")
    ax1.spines['left'].set_color('green')
    ax1.spines['left'].set_linewidth(2)
    
    # Right axis: Non-targets (red)
    ax2 = ax1.twinx()
    
    y_non_targets = unique_df["unique_non_targets"].to_numpy()
    y_non_targets_ma = rolling_mean(y_non_targets, window)
    
    line3 = ax2.plot(x, y_non_targets, alpha=0.2, color="red", linewidth=1)
    line4, = ax2.plot(x, y_non_targets_ma, alpha=0.9, label=f"Unique Non-Targets (MA {window})", color="red", linewidth=2.5, linestyle='--')
    
    ax2.set_ylabel("Unique Non-Targets", color="red", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.spines['right'].set_color('red')
    ax2.spines['right'].set_linewidth(2)
    
    # Annotate final counts
    final_targets = int(y_targets[-1]) if len(y_targets) > 0 else 0
    final_non_targets = int(y_non_targets[-1]) if len(y_non_targets) > 0 else 0
    
    if final_targets > 0:
        ax1.annotate(
            f"{final_targets}",
            xy=(final_step, final_targets),
            xytext=(final_step - len(x) * 0.12, final_targets + max(y_targets) * 0.08),
            fontsize=12,
            fontweight="bold",
            color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5)
        )
    
    if final_non_targets > 0:
        ax2.annotate(
            f"{final_non_targets}",
            xy=(final_step, final_non_targets),
            xytext=(final_step - len(x) * 0.12, final_non_targets - max(y_non_targets) * 0.08),
            fontsize=12,
            fontweight="bold",
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
        )
    
    # Combined legend
    lines = [line2, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11)
    
    ax1.set_title("Cumulative Unique Molecules Over Steps", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, f"unique_molecules_by_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_component_per_agent_by_step(agent_step_components, component, window, save_dir):
    """Line plots per agent for a single component by step with min/max bounds."""
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 7))
    
    has_data = False
    for agent, stepdf in agent_step_components.items():
        comp_mean = component + "_mean" if component + "_mean" in stepdf.columns else component
        comp_min = component + "_min"
        comp_max = component + "_max"
        
        if comp_mean not in stepdf.columns:
            print(f"      {agent}: Column '{comp_mean}' not found (step), skipping.")
            continue
        
        has_data = True
        x = stepdf["Step"].to_numpy()
        y = stepdf[comp_mean].to_numpy()
        y_ma = rolling_mean(y, window)
        
        plt.plot(x, y, alpha=0.25, label=f"{agent} (steps)")
        line, = plt.plot(x, y_ma, linewidth=2.0, label=f"{agent} (MA {window})")
        
        if comp_min in stepdf.columns and comp_max in stepdf.columns:
            y_min = rolling_mean(stepdf[comp_min].to_numpy(), window)
            y_max = rolling_mean(stepdf[comp_max].to_numpy(), window)
            plt.plot(x, y_min, linestyle='--', alpha=0.3, color=line.get_color())
            plt.plot(x, y_max, linestyle='--', alpha=0.3, color=line.get_color())

    if not has_data:
        plt.close()
        print(f"      No data for component '{component}' (step), skipping plot.")
        return None
        
    plt.title(f"{component} Evolution per Agent (by Step)")
    plt.xlabel("Step")
    plt.ylabel(component)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{component.replace(' ', '_')}_per_agent_by_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_components_grid_by_step(agent_step_components, components, window, save_dir):
    """Small multiples grid by step: each subplot is a component showing all agents."""
    ensure_dir(save_dir)
    n = len(components)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), squeeze=False)
    
    for idx, comp in enumerate(components):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for agent, stepdf in agent_step_components.items():
            comp_mean = comp + "_mean" if comp + "_mean" in stepdf.columns else comp
            if comp_mean not in stepdf.columns:
                continue
            x = stepdf["Step"].to_numpy()
            y = stepdf[comp_mean].to_numpy()
            y_ma = rolling_mean(y, window)
            ax.plot(x, y, alpha=0.2, label=f"{agent} (step)")
            ax.plot(x, y_ma, linewidth=2.0, label=f"{agent} (MA)")
        ax.set_title(comp)
        ax.set_xlabel("Step")
        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        fig.delaxes(axes[r][c])
    
    fig.suptitle(f"Reward Components per Agent by Step (MA {window})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(save_dir, f"components_grid_by_step_ma{window}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_noop_per_step(noop_stats, window, save_dir):
    """
    Plot no-op and invalid action statistics per step.
    Creates four plots:
        1. No-op count per step (per agent + combined)
        2. No-op rate per step (per agent + combined)
        3. Invalid count per step (per agent + combined)
        4. Invalid rate per step (per agent + combined)
    """
    ensure_dir(save_dir)
    outputs = []
    
    per_agent = noop_stats.get("per_agent", {})
    combined = noop_stats.get("combined", pd.DataFrame())
    
    if combined.empty:
        print("No action flag data available for step plotting.")
        return outputs
    
    # Plot 1: No-op count per step
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Step"].to_numpy()
        y = stats["noop_count"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    x_comb = combined["Step"].to_numpy()
    y_comb = combined["noop_count"].to_numpy()
    y_comb_ma = rolling_mean(y_comb, window)
    ax.plot(x_comb, y_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("No-op Count", fontsize=12)
    ax.set_title(f"No-op Count per Step (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"noop_count_per_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 2: No-op rate per step
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Step"].to_numpy()
        y = stats["noop_rate"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_rate_comb = combined["noop_rate"].to_numpy()
    y_rate_comb_ma = rolling_mean(y_rate_comb, window)
    ax.plot(x_comb, y_rate_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("No-op Rate", fontsize=12)
    ax.set_title(f"No-op Rate per Step (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"noop_rate_per_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 3: Invalid count per step
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Step"].to_numpy()
        y = stats["invalid_count"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_inv_comb = combined["invalid_count"].to_numpy()
    y_inv_comb_ma = rolling_mean(y_inv_comb, window)
    ax.plot(x_comb, y_inv_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Invalid Action Count", fontsize=12)
    ax.set_title(f"Invalid Action Count per Step (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"invalid_count_per_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 4: Invalid rate per step
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Step"].to_numpy()
        y = stats["invalid_rate"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_inv_rate_comb = combined["invalid_rate"].to_numpy()
    y_inv_rate_comb_ma = rolling_mean(y_inv_rate_comb, window)
    ax.plot(x_comb, y_inv_rate_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Invalid Action Rate", fontsize=12)
    ax.set_title(f"Invalid Action Rate per Step (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"invalid_rate_per_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    return outputs


def plot_validity_by_step(validity_df, window, save_dir):
    """
    Plot valid/invalid ratio over steps.
    Shows cumulative validity ratio with separate lines for valid and invalid.
    """
    ensure_dir(save_dir)
    
    if validity_df.empty:
        print("No validity data available for step-based plotting.")
        return None
    
    x = validity_df["Step"].to_numpy()
    final_step = x[-1] if len(x) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Valid ratio (green)
    y_valid = validity_df["valid_ratio"].to_numpy()
    y_valid_ma = rolling_mean(y_valid, window)
    
    ax.plot(x, y_valid, alpha=0.2, color="green", linewidth=1)
    ax.plot(x, y_valid_ma, alpha=0.9, label=f"Valid Ratio (MA {window})", color="green", linewidth=2.5)
    
    # Invalid ratio (red)
    y_invalid = validity_df["invalid_ratio"].to_numpy()
    y_invalid_ma = rolling_mean(y_invalid, window)
    
    ax.plot(x, y_invalid, alpha=0.2, color="red", linewidth=1)
    ax.plot(x, y_invalid_ma, alpha=0.9, label=f"Invalid Ratio (MA {window})", color="red", linewidth=2.5, linestyle='--')
    
    # Annotate final values
    final_valid = y_valid[-1] if len(y_valid) > 0 else 0
    final_invalid = y_invalid[-1] if len(y_invalid) > 0 else 0
    
    ax.annotate(
        f"{final_valid:.2%}",
        xy=(final_step, final_valid),
        xytext=(final_step - len(x) * 0.12, final_valid + 0.05),
        fontsize=12,
        fontweight="bold",
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5)
    )
    
    ax.annotate(
        f"{final_invalid:.2%}",
        xy=(final_step, final_invalid),
        xytext=(final_step - len(x) * 0.12, final_invalid - 0.05),
        fontsize=12,
        fontweight="bold",
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
    )
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Ratio", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Cumulative Valid/Invalid Ratio Over Steps", fontsize=14, fontweight='bold')
    ax.legend(loc="center right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, f"validity_ratio_by_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_binary_mpo_by_step(binary_mpo_df, window, save_dir, n_properties=None):
    """
    Plot cumulative average MPO score over steps using the existing MPO_Score column.
    
    Args:
        binary_mpo_df: DataFrame with 'Step' and 'cum_avg_props_satisfied' columns
        window: Moving average window size
        save_dir: Directory to save the plot
        n_properties: Total number of properties (for y-axis scaling), optional
    """
    ensure_dir(save_dir)
    
    if binary_mpo_df.empty or "cum_avg_props_satisfied" not in binary_mpo_df.columns:
        print("No binary MPO data available for step-based plotting.")
        return None
    
    x = binary_mpo_df["Step"].to_numpy()
    final_step = x[-1] if len(x) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    y = binary_mpo_df["cum_avg_props_satisfied"].to_numpy()
    y_ma = rolling_mean(y, window)
    
    ax.plot(x, y, alpha=0.2, color="blue", linewidth=1)
    ax.plot(x, y_ma, alpha=0.9, label=f"Avg MPO Score (MA {window})", color="blue", linewidth=2.5)
    
    # Annotate final value
    final_value = y[-1] if len(y) > 0 else 0
    
    ax.annotate(
        f"{final_value:.2f}",
        xy=(final_step, final_value),
        xytext=(final_step - len(x) * 0.12, final_value + max(y) * 0.08 if len(y) > 0 else 0.5),
        fontsize=12,
        fontweight="bold",
        color="blue",
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5)
    )
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Average MPO Score", fontsize=12)
    
    if n_properties is not None:
        ax.set_ylim(0, n_properties + 0.5)
        # Add horizontal line at max possible
        ax.axhline(y=n_properties, color='gray', linestyle=':', alpha=0.5, label=f"Max ({n_properties} properties)")
    
    ax.set_title("Cumulative Average MPO Score Over Steps\n(Sum of Satisfied Properties)", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, f"binary_mpo_by_step_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
