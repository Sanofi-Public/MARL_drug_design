#!/usr/bin/env python3
"""
Episode-wise plotting functions for MARL component analysis.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marl_plot_utils import (
    HAS_RDKIT,
    rolling_mean,
    ensure_dir,
    _get_property_score,
)

if HAS_RDKIT:
    from rdkit import Chem
    from rdkit.Chem import Draw


def plot_target_quality(unique_df, window, save_dir):
    """Plot average target quality scores over episodes."""
    ensure_dir(save_dir)
    
    if unique_df.empty:
        print("No quality score data available for plotting.")
        return None
    
    # Check if quality columns exist and have valid data
    has_target_quality = "avg_target_quality" in unique_df.columns and unique_df["avg_target_quality"].notna().any()
    has_non_target_quality = "avg_non_target_quality" in unique_df.columns and unique_df["avg_non_target_quality"].notna().any()
    
    if not has_target_quality and not has_non_target_quality:
        print("No valid quality score data available for plotting.")
        return None
    
    x = unique_df["Episode_number"].to_numpy()
    final_episode = x[-1] if len(x) > 0 else 0
    
    plt.figure(figsize=(12, 7))
    
    # Plot target quality scores if available
    if has_target_quality:
        y_target_quality = unique_df["avg_target_quality"].ffill().bfill().fillna(0).to_numpy()
        y_target_quality_ma = rolling_mean(y_target_quality, window)
        
        plt.plot(x, y_target_quality, alpha=0.3, label="Avg Target Quality (raw)", color="darkgreen", linewidth=1)
        plt.plot(x, y_target_quality_ma, alpha=0.8, label=f"Avg Target Quality (MA {window})", 
                 color="darkgreen", linewidth=2)
        
        # Label final value for targets
        final_target_quality = y_target_quality_ma[-1] if len(y_target_quality_ma) > 0 else 0
        
        if final_target_quality > 0:
            plt.annotate(
                f"{final_target_quality:.3f}",
                xy=(final_episode, final_target_quality),
                xytext=(final_episode - len(x) * 0.08, final_target_quality + 0.03),
                fontsize=11,
                fontweight="bold",
                color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5)
            )
    
    # Plot non-target quality scores if available
    if has_non_target_quality:
        y_non_target_quality = unique_df["avg_non_target_quality"].ffill().bfill().fillna(0).to_numpy()
        y_non_target_quality_ma = rolling_mean(y_non_target_quality, window)
        
        plt.plot(x, y_non_target_quality, alpha=0.3, label="Avg Non-Target Quality (raw)", 
                 color="steelblue", linewidth=1, linestyle='--')
        plt.plot(x, y_non_target_quality_ma, alpha=0.8, label=f"Avg Non-Target Quality (MA {window})", 
                 color="steelblue", linewidth=2, linestyle='--')
    
    plt.title("Average Target Quality Scores Over Episodes\n(Gaussian-weighted distance from property bounds)")
    plt.xlabel("Episode")
    plt.ylabel("Average Quality Score")
    plt.ylim(0, 1.05)  # Quality scores are between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, f"target_quality_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_random_episodes(agent_frames, save_dir, n_episodes=15, seed=None, property_names=None):
    """
    Plot N random episodes showing step-by-step molecule transformations for ALL agents.
    """
    ensure_dir(save_dir)
    outputs = []
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Filter agents that have valid SMILES columns
    valid_agents = {}
    for agent_label, df in agent_frames.items():
        if "Start_Smiles" not in df.columns or "End_Smiles" not in df.columns:
            print(f"Warning: {agent_label} missing SMILES columns, skipping for episode plots.")
            continue
        valid_agents[agent_label] = df
    
    if not valid_agents:
        print("No valid agents found for plotting.")
        return outputs
    
    # Create agent label to index mapping (for extracting correct score from Scores list)
    # If property_names provided, use that order; otherwise use sorted agent labels
    if property_names:
        agent_to_index = {name: idx for idx, name in enumerate(property_names) if name in valid_agents}
    else:
        agent_to_index = {label: idx for idx, label in enumerate(sorted(valid_agents.keys()))}
    
    # Get unique (Env_Index, Episode_number) pairs that exist across all agents
    episode_sets = []
    for agent_label, df in valid_agents.items():
        episode_keys = set(zip(df["Env_Index"], df["Episode_number"]))
        episode_sets.append(episode_keys)
    
    # Find episodes common to all agents
    common_episodes = episode_sets[0]
    for es in episode_sets[1:]:
        common_episodes = common_episodes.intersection(es)
    
    common_episodes = list(common_episodes)
    
    if not common_episodes:
        print("No common episodes found across all agents.")
        return outputs
    
    # Select random episodes
    n_to_sample = min(n_episodes, len(common_episodes))
    selected_episodes = random.sample(common_episodes, n_to_sample)
    print(f"Plotting {n_to_sample} random episodes with {len(valid_agents)} agents each...")
    
    agent_labels = sorted(valid_agents.keys())
    
    for idx, (env_idx, episode_num) in enumerate(selected_episodes):
        # Collect episode data for all agents
        agent_episode_data = {}
        for agent_label in agent_labels:
            df = valid_agents[agent_label]
            episode_df = df[(df["Env_Index"] == env_idx) & (df["Episode_number"] == episode_num)]
            episode_df = episode_df.sort_values("Episode_Step").reset_index(drop=True)
            if not episode_df.empty:
                agent_episode_data[agent_label] = episode_df
        
        if not agent_episode_data:
            continue
        
        # Determine agent order based on first Episode_Step value
        agent_first_steps = {}
        for agent_label, ep_df in agent_episode_data.items():
            first_step = ep_df["Episode_Step"].min()
            agent_first_steps[agent_label] = first_step
        
        # Sort agents by their first episode step to get correct execution order
        ordered_agent_labels = sorted(agent_first_steps.keys(), key=lambda x: agent_first_steps[x])
        
        # Create figure for this episode with all agents
        if HAS_RDKIT:
            fig = _plot_episode_with_molecules_all_agents(agent_episode_data, env_idx, episode_num, ordered_agent_labels, agent_to_index)
        #else:
        #    fig = _plot_episode_with_text_all_agents(agent_episode_data, env_idx, episode_num, ordered_agent_labels, agent_to_index)
        
        if fig is not None:
            out_path = os.path.join(save_dir, f"episode_{idx+1}_env{env_idx}_ep{episode_num}.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            outputs.append(out_path)
            print(f"  Saved: {out_path}")
    
    return outputs


def _plot_episode_with_molecules_all_agents(agent_episode_data, env_idx, episode_num, ordered_agent_labels=None, agent_to_index=None):
    """
    Plot episode with RDKit molecule visualizations for ALL agents.
    Shows the transformation chain: Input -> Agent1_Output -> Agent2_Output -> Agent3_Output
    Repeated targets are flagged in red.
    No-ops and invalid actions are indicated.
    Uses property-specific scores from Scores column based on agent index.
    Delta shows change from previous molecule (previous agent's output).
    Shows all property scores for each molecule.
    """
    # Use provided order or default to sorted
    if ordered_agent_labels is None:
        agent_labels = sorted(agent_episode_data.keys())
    else:
        agent_labels = ordered_agent_labels
    n_agents = len(agent_labels)
    
    if agent_to_index is None:
        agent_to_index = {label: idx for idx, label in enumerate(agent_labels)}
    
    # Create a list of all property names in their correct score index order
    # This ensures we display scores in the correct order regardless of agent execution order
    all_property_names = sorted(agent_to_index.keys(), key=lambda x: agent_to_index[x])
    
    # Find max steps across all agents
    max_steps_data = max(len(df) for df in agent_episode_data.values())
    
    # Limit steps to avoid overly large figures
    max_steps = 10
    truncated = max_steps_data > max_steps
    n_steps = min(max_steps_data, max_steps)
    
    # Track seen target SMILES to detect repeats
    seen_targets = set()
    
    # Create figure: (n_agents + 1) columns (input + output for each agent), n_steps rows
    n_cols = n_agents + 1  # 1 input + n_agents outputs
    fig, axes = plt.subplots(n_steps, n_cols, figsize=(3.5 * n_cols, 3 * n_steps))
    
    # Handle single step case
    if n_steps == 1:
        axes = axes.reshape(1, -1)
    
    title = f"Episode Trajectory: Env {env_idx} | Episode {episode_num}"
    if truncated:
        title += f" (showing {n_steps}/{max_steps_data} steps)"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Add column headers - positioned closer to the plots
    header_y = 0.96
    fig.text(0.5 / n_cols, header_y, "Input", ha='center', va='bottom', fontsize=11, fontweight='bold', color='gray')
    for agent_idx, agent_label in enumerate(agent_labels):
        fig.text((agent_idx + 1.5) / n_cols, header_y, agent_label, 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=plt.cm.tab10(agent_to_index.get(agent_label, agent_idx)))
    
    for step_idx in range(n_steps):
        # Get the input molecule (from first agent's Start_Smiles)
        first_agent_df = agent_episode_data[agent_labels[0]]
        
        # Get the previous molecule's scores (input to first agent = Start_Smiles scores)
        # This will be updated as we go through agents
        prev_mol_scores = None
        
        if step_idx < len(first_agent_df):
            row = first_agent_df.iloc[step_idx]
            episode_step = row.get("Episode_Step", step_idx)
            start_smiles = str(row.get("Start_Smiles", ""))
            
            # Plot input molecule (column 0)
            ax_in = axes[step_idx, 0]
            start_mol = Chem.MolFromSmiles(start_smiles) if start_smiles and start_smiles != "nan" else None
            if start_mol is not None:
                img = Draw.MolToImage(start_mol, size=(250, 200))
                ax_in.imshow(img)
            else:
                ax_in.text(0.5, 0.5, "Invalid", ha='center', va='center', fontsize=8)
            ax_in.axis('off')
            ax_in.set_title(f"Step {episode_step}", fontsize=10, fontweight='bold')
        else:
            axes[step_idx, 0].axis('off')
            axes[step_idx, 0].text(0.5, 0.5, "-", ha='center', va='center', fontsize=10, color='gray')
        
        # Plot each agent's output
        for agent_idx, agent_label in enumerate(agent_labels):
            episode_df = agent_episode_data[agent_label]
            ax_out = axes[step_idx, agent_idx + 1]
            
            if step_idx < len(episode_df):
                row = episode_df.iloc[step_idx]
                end_smiles = str(row.get("End_Smiles", ""))
                
                # Get all scores from this row
                current_scores = row.get("Scores", None)
                
                # Get the index of the current agent's property in the Scores list
                agent_prop_idx = agent_to_index.get(agent_label, None)
                
                # Build score display string with all properties in correct order
                score_parts = []
                
                if current_scores is not None and isinstance(current_scores, list):
                    # Iterate through properties in their correct score index order
                    for prop_name in all_property_names:
                        prop_idx = agent_to_index.get(prop_name)
                        if prop_idx is not None and prop_idx < len(current_scores):
                            score = current_scores[prop_idx]
                            if score is not None and not pd.isna(score):
                                # Calculate delta only for this agent's property
                                delta_str = ""
                                if prop_idx == agent_prop_idx and prev_mol_scores is not None:
                                    if isinstance(prev_mol_scores, list) and prop_idx < len(prev_mol_scores):
                                        prev_score = prev_mol_scores[prop_idx]
                                        if prev_score is not None and not pd.isna(prev_score):
                                            delta = score - prev_score
                                            delta_str = f"({delta:+.1f})"
                                
                                # Highlight the agent's own property with delta
                                if prop_idx == agent_prop_idx:
                                    score_parts.append(f"{prop_name}: {score:.1f}{delta_str}")
                                else:
                                    score_parts.append(f"{prop_name}: {score:.1f}")
                
                # Update prev_mol_scores for the next agent in the chain
                if current_scores is not None:
                    prev_mol_scores = current_scores
                    
                is_target = row.get("Target", False)
                is_noop = row.get("Is_Noop", False)
                is_invalid = row.get("Is_Invalid", False)
                
                # Check if this target is a repeat
                is_repeat = False
                if is_target and end_smiles and end_smiles != "nan":
                    if end_smiles in seen_targets:
                        is_repeat = True
                    else:
                        seen_targets.add(end_smiles)
                
                # Plot output molecule
                end_mol = Chem.MolFromSmiles(end_smiles) if end_smiles and end_smiles != "nan" else None
                if end_mol is not None:
                    img = Draw.MolToImage(end_mol, size=(250, 200))
                    ax_out.imshow(img)
                else:
                    ax_out.text(0.5, 0.5, "Invalid", ha='center', va='center', fontsize=8)
                ax_out.axis('off')
                
                # Build title with scores and flags
                title_lines = []
                
                # Add scores line
                if score_parts:
                    title_lines.append(" | ".join(score_parts))
                
                # Add flags on second line
                flag_parts = []
                if is_noop:
                    flag_parts.append("\u23f8NOOP")
                if is_invalid:
                    flag_parts.append("\u274cINVALID")
                if is_target:
                    if is_repeat:
                        flag_parts.append("\u26a0REPEAT")
                    else:
                        flag_parts.append("\u2605TARGET")
                
                if flag_parts:
                    title_lines.append(" | ".join(flag_parts))
                
                # Determine color: red for repeat/invalid, orange for noop, green for new target, black otherwise
                if is_repeat or is_invalid:
                    color = 'red'
                elif is_noop:
                    color = 'orange'
                elif is_target:
                    color = 'green'
                else:
                    color = 'black'
                
                title_text = "\n".join(title_lines) if title_lines else ""
                ax_out.set_title(title_text, fontsize=8,
                               color=color,
                               fontweight='bold' if is_target else 'normal')
            else:
                ax_out.axis('off')
                ax_out.text(0.5, 0.5, "-", ha='center', va='center', fontsize=10, color='gray')
        
        # Add arrows between columns
        if step_idx == 0:
            for col_idx in range(n_cols - 1):
                fig.text((col_idx + 1) / n_cols, 0.5, "\u2192", ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='gray', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _plot_episode_with_text_all_agents(agent_episode_data, env_idx, episode_num, ordered_agent_labels=None, agent_to_index=None):
    """
    Plot episode with SMILES text for ALL agents (fallback when RDKit is not available).
    Shows the transformation chain: Input -> Agent1_Output -> Agent2_Output -> ...
    Repeated targets are flagged in red.
    No-ops and invalid actions are indicated.
    Uses property-specific scores from Scores column based on agent index.
    Delta shows change from previous molecule (previous agent's output).
    """
    # Use provided order or default to sorted
    if ordered_agent_labels is None:
        agent_labels = sorted(agent_episode_data.keys())
    else:
        agent_labels = ordered_agent_labels
    n_agents = len(agent_labels)
    
    if agent_to_index is None:
        agent_to_index = {label: idx for idx, label in enumerate(agent_labels)}
    
    # Find max steps across all agents
    max_steps_data = max(len(df) for df in agent_episode_data.values())
    
    # Limit steps to avoid overly large figures
    max_steps = 10
    truncated = max_steps_data > max_steps
    n_steps = min(max_steps_data, max_steps)
    
    # Track seen target SMILES to detect repeats
    seen_targets = set()
    
    # Calculate figure width based on number of agents
    fig_width = 4 + 4 * (n_agents + 1)  # Input + agents
    fig, ax = plt.subplots(figsize=(fig_width, 0.8 * n_steps + 2))
    ax.axis('off')
    
    title = f"Episode Trajectory: Env {env_idx} | Episode {episode_num}"
    if truncated:
        title += f" (showing {n_steps}/{max_steps_data} steps)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Calculate column positions
    n_cols = n_agents + 1  # Input + agents
    col_width = 0.9 / n_cols
    step_col = 0.02
    
    y_pos = 0.92
    row_height = 0.85 / (n_steps + 1.5)
    
    # Header row
    ax.text(step_col, y_pos, "Step", fontsize=10, fontweight='bold', transform=ax.transAxes)
    ax.text(step_col + 0.04, y_pos, "Input", fontsize=10, fontweight='bold', transform=ax.transAxes, color='gray')
    
    for agent_idx, agent_label in enumerate(agent_labels):
        x_pos = step_col + 0.04 + (agent_idx + 1) * col_width
        ax.text(x_pos, y_pos, agent_label, fontsize=10, fontweight='bold', 
               transform=ax.transAxes, color=plt.cm.tab10(agent_idx))
    
    y_pos -= row_height
    
    # Data rows
    for step_idx in range(n_steps):
        first_agent_df = agent_episode_data[agent_labels[0]]
        
        # Track previous molecule's scores for delta calculation
        prev_mol_scores = None
        
        if step_idx < len(first_agent_df):
            row = first_agent_df.iloc[step_idx]
            step_num = row.get("Episode_Step", step_idx)
            start_smiles = str(row.get("Start_Smiles", ""))[:25]
            
            ax.text(step_col, y_pos, str(step_num), fontsize=9, fontweight='bold', transform=ax.transAxes)
            ax.text(step_col + 0.04, y_pos, start_smiles + ("..." if len(str(row.get("Start_Smiles", ""))) > 25 else ""),
                   fontsize=7, transform=ax.transAxes, family='monospace', color='gray')
        
        # Add arrow
        ax.text(step_col + 0.04 + col_width * 0.85, y_pos, "\u2192", fontsize=9, transform=ax.transAxes, color='gray')
        
        for agent_idx, agent_label in enumerate(agent_labels):
            x_pos = step_col + 0.04 + (agent_idx + 1) * col_width
            episode_df = agent_episode_data[agent_label]
            
            if step_idx < len(episode_df):
                row = episode_df.iloc[step_idx]
                end_smiles_full = str(row.get("End_Smiles", ""))
                end_smiles = end_smiles_full[:25]
                
                # Get property-specific score for this agent
                prop_score = _get_property_score(row, agent_label, agent_to_index)
                
                # Get all scores from this row to use as prev_mol_scores for next agent
                current_scores = row.get("Scores", None)
                
                # Calculate delta from previous molecule's score for this property
                delta_str = ""
                if not pd.isna(prop_score):
                    prop_idx = agent_to_index.get(agent_label, None)
                    if prev_mol_scores is not None and prop_idx is not None:
                        if isinstance(prev_mol_scores, list) and prop_idx < len(prev_mol_scores):
                            prev_score = prev_mol_scores[prop_idx]
                            if prev_score is not None and not pd.isna(prev_score):
                                delta = prop_score - prev_score
                                delta_str = f"({delta:+.1f})"
                
                # Update prev_mol_scores for the next agent in the chain
                if current_scores is not None:
                    prev_mol_scores = current_scores
                    
                is_target = row.get("Target", False)
                is_noop = row.get("Is_Noop", False)
                is_invalid = row.get("Is_Invalid", False)
                
                # Check if this target is a repeat
                is_repeat = False
                if is_target and end_smiles_full and end_smiles_full != "nan":
                    if end_smiles_full in seen_targets:
                        is_repeat = True
                    else:
                        seen_targets.add(end_smiles_full)
                
                # Determine color: red for repeat/invalid, orange for noop, green for new target, black otherwise
                if is_repeat or is_invalid:
                    color = 'red'
                elif is_noop:
                    color = 'orange'
                elif is_target:
                    color = 'green'
                else:
                    color = 'black'
                weight = 'bold' if is_target else 'normal'
                
                # Output SMILES
                ax.text(x_pos, y_pos, end_smiles + ("..." if len(end_smiles_full) > 25 else ""),
                       fontsize=7, transform=ax.transAxes, family='monospace', color=color, fontweight=weight)
                
                # Property name, score with delta and flags below SMILES
                score_text = f"{agent_label}: {prop_score:.1f}{delta_str}" if not pd.isna(prop_score) else ""
                if is_noop:
                    score_text += " \u23f8NOOP"
                if is_invalid:
                    score_text += " \u274cINV"
                if is_target:
                    if is_repeat:
                        score_text += " \u26a0REPEAT"
                    else:
                        score_text += " \u2605TARGET"
                ax.text(x_pos, y_pos - row_height * 0.35, score_text, fontsize=7, 
                       transform=ax.transAxes, color=color, fontweight=weight)
                
                # Add arrow to next (if not last agent)
                if agent_idx < n_agents - 1:
                    ax.text(x_pos + col_width * 0.85, y_pos, "\u2192", fontsize=9, transform=ax.transAxes, color='gray')
            else:
                ax.text(x_pos, y_pos, "-", fontsize=9, transform=ax.transAxes, color='gray')
        
        y_pos -= row_height * 1.5
    
    plt.tight_layout()
    return fig


def plot_component_per_agent(agent_components, component, window, save_dir):
    """Line plots per agent for a single component with min/max bounds."""
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 7))
    
    has_data = False
    for agent, epdf in agent_components.items():
        comp_mean = component + "_mean" if component + "_mean" in epdf.columns else component
        comp_min = component + "_min"
        comp_max = component + "_max"
        
        if comp_mean not in epdf.columns:
            print(f"      {agent}: Column '{comp_mean}' not found, skipping.")
            continue
        
        has_data = True
        y = epdf[comp_mean].to_numpy()
        x = np.arange(1, len(y) + 1)
        y_ma = rolling_mean(y, window)
        
        plt.plot(x, y, alpha=0.25, label=f"{agent} (episodes)")
        line, = plt.plot(x, y_ma, linewidth=2.0, label=f"{agent} (MA {window})")
        
        if comp_min in epdf.columns and comp_max in epdf.columns:
            y_min = rolling_mean(epdf[comp_min].to_numpy(), window)
            y_max = rolling_mean(epdf[comp_max].to_numpy(), window)
            plt.plot(x, y_min, linestyle='--', alpha=0.3, color=line.get_color())
            plt.plot(x, y_max, linestyle='--', alpha=0.3, color=line.get_color())

    if not has_data:
        plt.close()
        print(f"      No data for component '{component}', skipping plot.")
        return None
        
    plt.title(f"{component} Evolution per Agent")
    plt.xlabel("Episode")
    plt.ylabel(component)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{component.replace(' ', '_')}_per_agent_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_components_grid(agent_components, components, window, save_dir):
    """Small multiples grid: each subplot is a component showing all agents."""
    ensure_dir(save_dir)
    n = len(components)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), squeeze=False)
    
    for idx, comp in enumerate(components):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for agent, epdf in agent_components.items():
            comp_mean = comp + "_mean" if comp + "_mean" in epdf.columns else comp
            if comp_mean not in epdf.columns:
                continue
            y = epdf[comp_mean].to_numpy()
            x = np.arange(1, len(y) + 1)
            y_ma = rolling_mean(y, window)
            ax.plot(x, y, alpha=0.2, label=f"{agent} (ep)")
            ax.plot(x, y_ma, linewidth=2.0, label=f"{agent} (MA)")
        ax.set_title(comp)
        ax.set_xlabel("Episode")
        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        fig.delaxes(axes[r][c])
    
    fig.suptitle(f"Reward Components per Agent (MA {window})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(save_dir, f"components_grid_ma{window}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_noop_per_episode(noop_stats, window, save_dir):
    """
    Plot no-op and invalid action statistics per episode.
    Creates four plots:
        1. No-op count per episode (per agent + combined)
        2. No-op rate per episode (per agent + combined)
        3. Invalid count per episode (per agent + combined)
        4. Invalid rate per episode (per agent + combined)
    """
    ensure_dir(save_dir)
    outputs = []
    
    per_agent = noop_stats.get("per_agent", {})
    combined = noop_stats.get("combined", pd.DataFrame())
    
    if combined.empty:
        print("No action flag data available for episode plotting.")
        return outputs
    
    # Plot 1: No-op count per episode
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Episode_number"].to_numpy()
        y = stats["noop_count"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    x_comb = combined["Episode_number"].to_numpy()
    y_comb = combined["noop_count"].to_numpy()
    y_comb_ma = rolling_mean(y_comb, window)
    ax.plot(x_comb, y_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("No-op Count", fontsize=12)
    ax.set_title(f"No-op Count per Episode (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"noop_count_per_episode_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 2: No-op rate per episode
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Episode_number"].to_numpy()
        y = stats["noop_rate"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_rate_comb = combined["noop_rate"].to_numpy()
    y_rate_comb_ma = rolling_mean(y_rate_comb, window)
    ax.plot(x_comb, y_rate_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("No-op Rate", fontsize=12)
    ax.set_title(f"No-op Rate per Episode (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"noop_rate_per_episode_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 3: Invalid count per episode
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Episode_number"].to_numpy()
        y = stats["invalid_count"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_inv_comb = combined["invalid_count"].to_numpy()
    y_inv_comb_ma = rolling_mean(y_inv_comb, window)
    ax.plot(x_comb, y_inv_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Invalid Action Count", fontsize=12)
    ax.set_title(f"Invalid Action Count per Episode (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"invalid_count_per_episode_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    # Plot 4: Invalid rate per episode
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_label, stats in per_agent.items():
        x = stats["Episode_number"].to_numpy()
        y = stats["invalid_rate"].to_numpy()
        y_ma = rolling_mean(y, window)
        ax.plot(x, y_ma, alpha=0.7, linewidth=1.5, label=f"{agent_label}")
    
    # Combined
    y_inv_rate_comb = combined["invalid_rate"].to_numpy()
    y_inv_rate_comb_ma = rolling_mean(y_inv_rate_comb, window)
    ax.plot(x_comb, y_inv_rate_comb_ma, linewidth=2.5, color="black", linestyle="--", label="Combined")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Invalid Action Rate", fontsize=12)
    ax.set_title(f"Invalid Action Rate per Episode (MA {window})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, f"invalid_rate_per_episode_ma{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    outputs.append(out_path)
    
    return outputs


def plot_components_stacked(agent_components, components, window, save_dir):
    """Stacked area plot per agent showing component contributions."""
    ensure_dir(save_dir)
    outputs = []
    
    for agent, epdf in agent_components.items():
        series_list = []
        labels = []
        for comp in components:
            comp_mean = comp + "_mean" if comp + "_mean" in epdf.columns else comp
            if comp_mean in epdf.columns:
                y = rolling_mean(epdf[comp_mean].to_numpy(), window)
                series_list.append(y)
                labels.append(comp)
        if not series_list:
            continue
        
        min_len = min(len(s) for s in series_list)
        series_list = [s[:min_len] for s in series_list]
        x = np.arange(1, min_len + 1)
        
        plt.figure(figsize=(12, 7))
        plt.stackplot(x, series_list, labels=labels, alpha=0.8)
        plt.title(f"Stacked Components (MA {window}) \u2014 {agent}")
        plt.xlabel("Episode")
        plt.ylabel("Component magnitude")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{agent}_components_stacked_ma{window}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        outputs.append(out_path)
    
    return outputs
