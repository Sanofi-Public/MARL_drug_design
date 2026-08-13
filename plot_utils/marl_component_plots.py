#!/usr/bin/env python3
"""
Plot per-agent reward components and unique targets from CSV logs.

CLI example:
    python marl_component_plots.py --data_dir ./logs --save_dir ./plots \
        --config configs/my_config.yaml \
        --components "Base_Reward" "Novelty_Reward" "Repeat_Penalty" "Total_Reward" \
        --window 100
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marl_plot_utils import (
    DEFAULT_COMPONENTS,
    load_config,
    get_property_names_from_config,
    get_property_bounds_from_config,
    get_agent_label,
    load_agent_csv,
    ensure_dir,
    compute_episode_components,
    compute_step_components,
    compute_unique_molecules_over_steps,
    compute_noop_per_episode,
    compute_noop_per_step,
    detect_agent_files,
    extract_molecule_properties,
    compute_validity_over_steps,
    compute_binary_mpo_over_steps,
)

from marl_episode_plots import (
    plot_component_per_agent,
    plot_components_grid,
    plot_components_stacked,
    plot_noop_per_episode,
    plot_random_episodes,
)

from marl_step_plots import (
    plot_unique_molecules_by_step,
    plot_component_per_agent_by_step,
    plot_components_grid_by_step,
    plot_noop_per_step,
    plot_validity_by_step,
    plot_binary_mpo_by_step,
)


def main():
    parser = argparse.ArgumentParser(description="Plot reward components per agent from CSVs.")
    parser.add_argument("--data_dir", type=str, default="./logs", help="Folder containing CSVs")
    parser.add_argument("--save_dir", type=str, default="./plots", help="Output folder for plots")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (yaml/json) for property names")
    parser.add_argument("--agents", nargs="*", default=None, help="Agent CSV filenames (auto-detected if not specified)")
    parser.add_argument("--components", nargs="*", default=DEFAULT_COMPONENTS, help="Components to plot")
    parser.add_argument("--window", type=int, default=100, help="Moving average window")
    parser.add_argument("--agg_map", type=str, default=None, help="JSON string: component -> agg mode")
    parser.add_argument("--eval", type=bool, default=True, help="Whether to plot evaluation data")
    parser.add_argument("--plot_mode", type=str, default="both", choices=["episode", "step", "both"],
                        help="Plot by 'episode', 'step', or 'both'")
    args = parser.parse_args()
    
    # Resolve symbolic component name
    if args.components and len(args.components) == 1 and args.components[0] == "DEFAULT_COMPONENTS":
        args.components = DEFAULT_COMPONENTS
    
    # Store original paths
    base_data_dir = args.data_dir.rstrip("/")
    base_save_dir = args.save_dir.rstrip("/")
    
    # Load config and extract property names and bounds
    config = load_config(args.config)
    property_names = get_property_names_from_config(config)
    property_bounds = get_property_bounds_from_config(config)
    
    if property_names:
        print(f"Using property names from config: {property_names}")
    else:
        print("No property names found in config, using default agent labels.")
    
    if property_bounds:
        print(f"Using property bounds from config: {property_bounds}")
    else:
        print("No property bounds found in config, quality scores will not be computed.")
    
    # Derive subdirectory from config filename
    config_name = None
    if args.config:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # Auto-resolve data_dir to data_dir/config_name if needed
        if not detect_agent_files(base_data_dir):
            candidate = os.path.join(base_data_dir, config_name)
            if os.path.isdir(candidate) and detect_agent_files(candidate):
                print(f"Using config-derived data path: {candidate}")
                base_data_dir = candidate
        # Save results under save_dir/config_name
        base_save_dir = os.path.join(base_save_dir, config_name)
    
    # Auto-detect agent files if not specified
    if args.agents is None:
        agent_files = detect_agent_files(base_data_dir)
        if not agent_files:
            raise FileNotFoundError(f"No agent_*.csv files found in {base_data_dir}")
        print(f"Auto-detected {len(agent_files)} agent files: {agent_files}")
    else:
        agent_files = args.agents
    
    train_agent_frames = {}
    for idx, fname in enumerate(agent_files):
        path = os.path.join(base_data_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        label = get_agent_label(fname, idx, property_names)
        df = load_agent_csv(path)
        train_agent_frames[label] = df
    
    eval_agent_frames = {}
    if args.eval:
        print("Loading evaluation data...")
        eval_data_dir = base_data_dir + "_eval"
        
        # Auto-detect eval agent files too
        if args.agents is None:
            eval_agent_files = detect_agent_files(eval_data_dir)
            if not eval_agent_files:
                print(f"No agent_*.csv files found in {eval_data_dir}, skipping eval.")
        else:
            eval_agent_files = args.agents
        
        for idx, fname in enumerate(eval_agent_files):
            path = os.path.join(eval_data_dir, fname)
            if not os.path.isfile(path):
                print(f"Evaluation file not found: {path}, skipping.")
                continue
            label = get_agent_label(fname, idx, property_names)
            df = load_agent_csv(path)
            eval_agent_frames[label] = df
    
    def make_plots(agent_frames, save_dir, plot_mode, property_names=None, property_bounds=None):
        """Generate all plots for given agent frames."""
        agg_map = json.loads(args.agg_map) if args.agg_map else None
        ensure_dir(save_dir)
        
        # Episode-based plots
        if plot_mode in ["episode", "both"]:
            print("  Generating episode-based plots...")
            episode_save_dir = save_dir if plot_mode == "episode" else os.path.join(save_dir, "by_episode")
            ensure_dir(episode_save_dir)
            
            agent_components = {}
            for label, df in agent_frames.items():
                ep_comp = compute_episode_components(df, components=args.components, agg_map=agg_map)
                agent_components[label] = ep_comp
                print(f"    {label}: {len(ep_comp)} episodes aggregated.")

            # Per-component plots
            for comp in args.components:
                out = plot_component_per_agent(agent_components, comp, args.window, episode_save_dir)
                print(f"    Saved: {out}")

            # Grid plot
            grid_out = plot_components_grid(agent_components, args.components, args.window, episode_save_dir)
            print(f"    Saved: {grid_out}")

            # Stacked plots
            stacked_outs = plot_components_stacked(agent_components, args.components, args.window, episode_save_dir)
            for p in stacked_outs:
                print(f"    Saved: {p}")
            
            # No-op and invalid action plots by episode
            noop_episode_stats = compute_noop_per_episode(agent_frames)
            noop_episode_plots = plot_noop_per_episode(noop_episode_stats, args.window, episode_save_dir)
            for p in noop_episode_plots:
                print(f"    Saved: {p}")
            
            # Random episode trajectory plots - pass property_names for score extraction
            episode_plots_dir = os.path.join(episode_save_dir, "episode_trajectories")
            episode_plots = plot_random_episodes(agent_frames, episode_plots_dir, n_episodes=5, property_names=property_names)
            for p in episode_plots:
                print(f"    Saved: {p}")
        
        # Step-based plots
        if plot_mode in ["step", "both"]:
            print("  Generating step-based plots...")
            step_save_dir = save_dir if plot_mode == "step" else os.path.join(save_dir, "by_step")
            ensure_dir(step_save_dir)
            
            agent_step_components = {}
            for label, df in agent_frames.items():
                step_comp = compute_step_components(df, components=args.components)
                agent_step_components[label] = step_comp
                print(f"    {label}: {len(step_comp)} steps aggregated.")

            # Per-component plots by step
            for comp in args.components:
                out = plot_component_per_agent_by_step(agent_step_components, comp, 10*args.window, step_save_dir)
                print(f"    Saved: {out}")

            # Grid plot by step
            grid_out = plot_components_grid_by_step(agent_step_components, args.components, 10*args.window, step_save_dir)
            print(f"    Saved: {grid_out}")

            # Unique molecules plot by step (overall stats, quality computed only for targets)
            unique_step_df = compute_unique_molecules_over_steps(
                agent_frames, 
                save_dir=step_save_dir,
                property_names=property_names,
                property_bounds=property_bounds
            )
            if not unique_step_df.empty:
                molecules_out = plot_unique_molecules_by_step(unique_step_df, args.window, step_save_dir)
                if molecules_out:
                    # Save summary CSV
                    unique_step_df.to_csv(os.path.join(step_save_dir, "unique_molecules_by_step_summary.csv"), index=False)
                    print(f"    Saved: {molecules_out}")
                    print(f"    Final unique targets count (by step): {unique_step_df['unique_targets'].iloc[-1]}")
                    print(f"    Final unique non-targets count (by step): {unique_step_df['unique_non_targets'].iloc[-1]}")
                    if unique_step_df['avg_target_quality'].notna().any():
                        print(f"    Final avg target quality: {unique_step_df['avg_target_quality'].iloc[-1]:.3f}")
            
            # No-op and invalid action plots by step
            noop_step_stats = compute_noop_per_step(agent_frames)
            noop_step_plots = plot_noop_per_step(noop_step_stats, 10*args.window, step_save_dir)
            for p in noop_step_plots:
                print(f"    Saved: {p}")
            
            # Validity ratio plot by step
            validity_df = compute_validity_over_steps(agent_frames)
            if not validity_df.empty:
                validity_out = plot_validity_by_step(validity_df, args.window, step_save_dir)
                if validity_out:
                    validity_df.to_csv(os.path.join(step_save_dir, "validity_by_step_summary.csv"), index=False)
                    print(f"    Saved: {validity_out}")
                    print(f"    Final valid ratio: {validity_df['valid_ratio'].iloc[-1]:.2%}")
            
            # Binary MPO plot by step (properties satisfied count)
            binary_mpo_df = compute_binary_mpo_over_steps(agent_frames, property_bounds=property_bounds)
            if not binary_mpo_df.empty:
                n_props = len(property_bounds) if property_bounds else None
                binary_mpo_out = plot_binary_mpo_by_step(binary_mpo_df, args.window, step_save_dir, n_properties=n_props)
                if binary_mpo_out:
                    binary_mpo_df.to_csv(os.path.join(step_save_dir, "binary_mpo_by_step_summary.csv"), index=False)
                    print(f"    Saved: {binary_mpo_out}")
                    print(f"    Final avg properties satisfied: {binary_mpo_df['cum_avg_props_satisfied'].iloc[-1]:.2f}")
    
    print(f"Processing training data (mode: {args.plot_mode})...")
    make_plots(train_agent_frames, save_dir=base_save_dir, plot_mode=args.plot_mode, 
               property_names=property_names, property_bounds=property_bounds)
    
    # Extract molecule properties table
    print("\nExtracting molecule properties...")
    mol_out = os.path.join(base_save_dir, "molecules_with_properties.csv")
    extract_molecule_properties(base_data_dir, property_names=property_names, output_path=mol_out)
    
    if args.eval and eval_agent_frames:
        print(f"\nProcessing evaluation data (mode: {args.plot_mode})...")
        eval_save_dir = os.path.join(base_save_dir, "eval")
        make_plots(eval_agent_frames, save_dir=eval_save_dir, plot_mode=args.plot_mode,
                   property_names=property_names, property_bounds=property_bounds)
        
        eval_data_dir = base_data_dir + "_eval"
        eval_mol_out = os.path.join(eval_save_dir, "molecules_with_properties.csv")
        extract_molecule_properties(eval_data_dir, property_names=property_names, output_path=eval_mol_out)
    elif args.eval and not eval_agent_frames:
        print("\nNo evaluation data loaded, skipping eval plots.")

if __name__ == "__main__":
    main()

