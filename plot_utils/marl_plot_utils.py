#!/usr/bin/env python3
"""
Utility functions, constants, and data-processing helpers for MARL component plots.
"""

import os
import ast
import json
import pandas as pd
import numpy as np
import yaml
from scipy.stats import norm

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available. Episode molecule plotting will use SMILES text only.")

DEFAULT_COMPONENTS = [
    "No_Op_Reward",
    "Base_Reward",
    "Invalid_Penalty",
    "Target_Bonus",
    "Repeat_Penalty",
    "Novelty_Reward",
    "Deferred_Bonus_Collected",
    "Total_Reward",
    "Episode_Reward",
]

DEFAULT_AGG_MAP = {
    "No_Op_Reward": "sum",
    "Base_Reward": "sum",
    "Invalid_Penalty": "sum",
    "Target_Bonus": "sum",
    "Repeat_Penalty": "sum",
    "Novelty_Reward": "sum",
    "Deferred_Bonus_Collected": "sum",
    "Total_Reward": "sum",
    "Episode_Reward": "last",
    "Prop_Indicator": "last",
    "Is_Noop": "sum",
    "Is_Invalid": "sum",
    "Is_Prop_Improved": "sum",
}

# Column name mappings (old -> new)
COLUMN_MAPPINGS = {
    # Old format -> New format
    "Base Reward": "Base_Reward",
    "Novelty reward": "Novelty_Reward",
    "Repeat penalty": "Repeat_Penalty",
    "Step reward": "Total_Reward",
    "Indicator": "Prop_Indicator",
    "Episode Reward": "Episode_Reward",
    "Episode step": "Episode_Step",
    "Start_Smiles": "Start_Smiles",
    "End_Smiles": "End_Smiles",
    "Start Smiles": "Start_Smiles",
    "End Smiles": "End_Smiles",
    "MPO_Score": "MPO_Score",
    "MPO Score": "MPO_Score",
    "MPO_score": "MPO_Score",
    # Handle space variants
    "No Op Reward": "No_Op_Reward",
    "No_Op Reward": "No_Op_Reward",
    "Invalid Penalty": "Invalid_Penalty",
    "Target Bonus": "Target_Bonus",
    "Total Reward": "Total_Reward",
    "Novelty Reward": "Novelty_Reward",
    "Repeat Penalty": "Repeat_Penalty",
}


def load_config(config_path):
    """Load configuration file and extract property names."""
    if config_path is None:
        print(f"Config path is None")
        return None
    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}")
        return None
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            print(f"Warning: Unknown config format for {config_path}")
            return None
    
    print(f"Config top-level keys: {list(config.keys())}")
    return config


def get_property_names_from_config(config):
    """Extract property names from config."""
    if config is None:
        return None
    
    # Try different possible config structures
    # Direct "properties" key at root level
    if "properties" in config:
        prop_dict = config["properties"]
        if "names" in prop_dict:
            return prop_dict["names"]
    
    # Check if nested under common parent keys
    for parent_key in ["algorithm", "env", "training", "experiment"]:
        if parent_key in config and isinstance(config[parent_key], dict):
            if "properties" in config[parent_key]:
                prop_dict = config[parent_key]["properties"]
                if "names" in prop_dict:
                    return prop_dict["names"]
    
    if "prop_list" in config:
        return config["prop_list"]
    
    # Check for prop_names directly
    if "prop_names" in config:
        return config["prop_names"]
    
    return None


def get_property_bounds_from_config(config):
    """Extract property bounds from config."""
    if config is None:
        return None
    
    # Try different possible config structures
    # Direct "properties" key at root level
    if "properties" in config:
        prop_dict = config["properties"]
        if "bounds" in prop_dict:
            return prop_dict["bounds"]
    
    # Check if nested under common parent keys
    for parent_key in ["algorithm", "env", "training", "experiment"]:
        if parent_key in config and isinstance(config[parent_key], dict):
            if "properties" in config[parent_key]:
                prop_dict = config[parent_key]["properties"]
                if "bounds" in prop_dict:
                    return prop_dict["bounds"]
    
    if "prop_bounds" in config:
        return config["prop_bounds"]
    
    if "prop_score_bounds" in config:
        return config["prop_score_bounds"]
    
    return None


def get_agent_label(agent_filename, agent_index, property_names):
    """
    Get display label for an agent.
    If property_names is available, use the property name.
    Otherwise, use the filename.
    """
    if property_names is not None and agent_index < len(property_names):
        return property_names[agent_index]
    
    # Fallback to filename-based label
    return os.path.splitext(os.path.basename(agent_filename))[0]


def safe_to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def normalize_columns(df):
    """Normalize column names to new format."""
    df = df.copy()
    for old_name, new_name in COLUMN_MAPPINGS.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    return df


def load_agent_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    # Print available columns for debugging
    print(f"  Columns in {os.path.basename(path)}: {list(df.columns)[:15]}...")
    
    # Normalize column names
    df = normalize_columns(df)
    
    required = {"Step", "Env_Index", "Episode_number"}
    # Check for Episode_Step or Episode step
    if "Episode_Step" not in df.columns and "Episode step" not in df.columns:
        required.add("Episode_Step")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    
    df["Env_Index"] = df["Env_Index"].astype(int)
    df["Episode_number"] = df["Episode_number"].astype(int)
    df["Step"] = df["Step"].astype(int)
    
    # Handle Episode_Step column
    if "Episode_Step" in df.columns:
        df["Episode_Step"] = df["Episode_Step"].astype(int)
    elif "Episode step" in df.columns:
        df["Episode_Step"] = df["Episode step"].astype(int)
    
    df["Target"] = df["Target"].astype(bool) if "Target" in df.columns else False
    
    # Handle SMILES columns
    if "Start_Smiles" in df.columns:
        df["Start_Smiles"] = df["Start_Smiles"].astype(str)
    else:
        df["Start_Smiles"] = ""
    
    if "End_Smiles" in df.columns:
        df["End_Smiles"] = df["End_Smiles"].astype(str)
    else:
        df["End_Smiles"] = ""
    
    # Handle MPO_Score column
    if "MPO_Score" in df.columns:
        df["MPO_Score"] = pd.to_numeric(df["MPO_Score"], errors='coerce')
    else:
        df["MPO_Score"] = np.nan
    
    # Handle Is_Noop column (use directly if available, otherwise infer)
    if "Is_Noop" in df.columns:
        df["Is_Noop"] = df["Is_Noop"].astype(bool).astype(int)
    else:
        # Fallback: infer from SMILES comparison
        df["Is_Noop"] = (df["Start_Smiles"] == df["End_Smiles"]).astype(int)
    
    # Handle Is_Invalid column
    if "Is_Invalid" in df.columns:
        df["Is_Invalid"] = df["Is_Invalid"].astype(bool).astype(int)
    else:
        df["Is_Invalid"] = 0
    
    # Handle Is_Prop_Improved column
    if "Is_Prop_Improved" in df.columns:
        df["Is_Prop_Improved"] = df["Is_Prop_Improved"].astype(bool).astype(int)
    else:
        df["Is_Prop_Improved"] = 0
    
    # Handle Scores column (list of all property scores)
    if "Scores" in df.columns:
        df["Scores"] = df["Scores"].apply(safe_to_list)
    else:
        df["Scores"] = None
    
    for col in ["Fragments_Removed", "Fragments_Added", "Scores"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_to_list)
    
    # Print which reward components are available
    reward_cols = [c for c in df.columns if any(x in c.lower() for x in ['reward', 'penalty', 'bonus', 'total'])]
    print(f"  Reward columns found: {reward_cols}")
    
    return df


def compute_episode_components(df, components, agg_map=None):
    """Aggregate reward components per episode across all environments."""
    if agg_map is None:
        agg_map = DEFAULT_AGG_MAP
    
    df = df.sort_values(["Env_Index", "Episode_number", "Episode_Step"])
    gb = df.groupby(["Env_Index", "Episode_number"], as_index=False)
    
    # Check which components exist
    available_components = [c for c in components if c in df.columns]
    missing_components = [c for c in components if c not in df.columns]
    if missing_components:
        print(f"    Warning: Components not found in data: {missing_components}")
    if not available_components:
        print(f"    Error: No requested components found. Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=["Episode_number", "num_envs"])
    
    comp_frames = []
    for comp in available_components:
        mode = agg_map.get(comp, "sum")
        if mode == "sum":
            agg = gb[comp].sum()
        elif mode == "mean":
            agg = gb[comp].mean()
        elif mode == "last":
            agg = df.groupby(["Env_Index", "Episode_number"], as_index=False)[comp].last()
        else:
            raise ValueError(f"Unsupported aggregation mode '{mode}' for '{comp}'.")
        comp_frames.append(agg)
    
    if comp_frames:
        merged = comp_frames[0][["Env_Index", "Episode_number"]].copy()
        for cf in comp_frames:
            merged = pd.merge(merged, cf, on=["Env_Index", "Episode_number"], how="outer")
    else:
        return pd.DataFrame(columns=["Episode_number", "num_envs"])
    
    agg_dict = {"Env_Index": "count"}
    for comp in available_components:
        if comp in merged.columns:
            agg_dict[comp] = ["mean", "min", "max"]
    
    episode_agg = merged.groupby("Episode_number", as_index=False).agg(agg_dict)
    episode_agg.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in episode_agg.columns.values]
    episode_agg = episode_agg.rename(columns={"Env_Index_count": "num_envs"})
    episode_agg = episode_agg.sort_values("Episode_number").reset_index(drop=True)
    return episode_agg

def compute_step_components(df, components):
    """Aggregate reward components per global step across all environments."""
    df = df.sort_values(["Step", "Env_Index"])
    
    # Check which components exist
    available_components = [c for c in components if c in df.columns]
    missing_components = [c for c in components if c not in df.columns]
    if missing_components:
        print(f"    Warning: Components not found in data (step): {missing_components}")
    if not available_components:
        print(f"    Error: No requested components found. Available columns: {list(df.columns)}")
        return pd.DataFrame(columns=["Step", "num_envs"])
    
    agg_dict = {"Env_Index": "count"}
    for comp in available_components:
        agg_dict[comp] = ["mean", "min", "max"]
    
    step_agg = df.groupby("Step", as_index=False).agg(agg_dict)
    
    # Flatten multi-level column names
    new_columns = []
    for col in step_agg.columns:
        if isinstance(col, tuple):
            if col[1]:  # Has aggregation suffix
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col[0])
        else:
            new_columns.append(col)
    step_agg.columns = new_columns
    
    step_agg = step_agg.rename(columns={"Env_Index_count": "num_envs"})
    step_agg = step_agg.sort_values("Step").reset_index(drop=True)
    return step_agg


def compute_unique_molecules_over_episodes(agent_frames, save_dir=None, property_names=None, property_bounds=None):
    """
    Compute cumulative unique targets and non-targets found across all agents and environments.
    Also tracks average MPO scores and target quality scores for unique molecules.
    
    Uses 'End_Smiles' column to track unique molecules.
    Returns DataFrame with columns: 
        ['Episode_number', 'unique_targets', 'unique_non_targets', 
         'avg_target_mpo', 'avg_non_target_mpo',
         'avg_target_quality', 'avg_non_target_quality']
    """
    target_rows = []
    non_target_rows = []
    
    for agent_label, df in agent_frames.items():
        if "End_Smiles" not in df.columns or "Target" not in df.columns:
            continue
        
        # Compute quality scores if bounds are available
        if property_bounds is not None and "Scores" in df.columns:
            df = df.copy()
            df["Quality_Score"] = compute_target_quality_for_dataframe(df, property_names, property_bounds)
        else:
            df = df.copy()
            df["Quality_Score"] = np.nan
        
        target_df = df[df["Target"] == True][["Step", "Episode_number", "End_Smiles", "MPO_Score", "Scores", "Quality_Score"]].copy()
        print(f"{agent_label}: Found {len(target_df)} target entries.")
        target_rows.append(target_df)
        
        non_target_df = df[df["Target"] == False][["Step", "Episode_number", "End_Smiles", "MPO_Score", "Scores", "Quality_Score"]].copy()
        print(f"{agent_label}: Found {len(non_target_df)} non-target entries.")
        non_target_rows.append(non_target_df)
    
    if not target_rows and not non_target_rows:
        return pd.DataFrame(columns=["Episode_number", "unique_targets", "unique_non_targets", 
                                     "avg_target_mpo", "avg_non_target_mpo",
                                     "avg_target_quality", "avg_non_target_quality"])
    
    def process_molecules(rows, label="molecules", save_dir=None):
        """Vectorized processing of molecule data."""
        if not rows:
            return pd.DataFrame(columns=["Episode_number", f"unique_{label}", f"avg_{label}_mpo", f"avg_{label}_quality"])
        
        combined = pd.concat(rows, ignore_index=True)
        combined = combined.sort_values("Step").reset_index(drop=True)
        # Filter out invalid SMILES entries
        combined = combined[
            combined["End_Smiles"].notna() & 
            (combined["End_Smiles"] != "") &
            (combined["End_Smiles"] != "nan") &
            (combined["End_Smiles"].str.strip() != "")
        ]
        
        if combined.empty:
            return pd.DataFrame(columns=["Episode_number", f"unique_{label}", f"avg_{label}_mpo", f"avg_{label}_quality"])
        
        # Keep first occurrence of each SMILES (already sorted by Step)
        # This ensures uniqueness - each SMILES is counted only once
        first_occurrences = combined.drop_duplicates(subset="End_Smiles", keep="first")
        
        # Group by episode to get cumulative counts and rolling averages
        first_occurrences = first_occurrences.sort_values("Step")
        first_occurrences["cumulative_unique"] = range(1, len(first_occurrences) + 1)
        
        # MPO score tracking
        first_occurrences["cumulative_mpo_sum"] = first_occurrences["MPO_Score"].fillna(0).cumsum()
        first_occurrences["cumulative_mpo_count"] = first_occurrences["MPO_Score"].notna().cumsum()
        first_occurrences["avg_mpo"] = (
            first_occurrences["cumulative_mpo_sum"] / 
            first_occurrences["cumulative_mpo_count"].replace(0, np.nan)
        )
        
        # Quality score tracking
        first_occurrences["cumulative_quality_sum"] = first_occurrences["Quality_Score"].fillna(0).cumsum()
        first_occurrences["cumulative_quality_count"] = first_occurrences["Quality_Score"].notna().cumsum()
        first_occurrences["avg_quality"] = (
            first_occurrences["cumulative_quality_sum"] / 
            first_occurrences["cumulative_quality_count"].replace(0, np.nan)
        )
        
        # Get last value per episode
        episode_agg = first_occurrences.groupby("Episode_number", as_index=False).agg({
            "cumulative_unique": "max",
            "avg_mpo": "last",
            "avg_quality": "last"
        })
        
        episode_agg = episode_agg.rename(columns={
            "cumulative_unique": f"unique_{label}",
            "avg_mpo": f"avg_{label}_mpo",
            "avg_quality": f"avg_{label}_quality"
        })
        
        print(f"Total unique {label} found: {len(first_occurrences)}")
        
        # Print quality stats if available
        valid_quality = first_occurrences["Quality_Score"].dropna()
        if len(valid_quality) > 0:
            print(f"  Quality scores - mean: {valid_quality.mean():.3f}, min: {valid_quality.min():.3f}, max: {valid_quality.max():.3f}")

        # Save unique smiles to csv in the save directory
        if save_dir is not None:
            ensure_dir(save_dir)
            unique_mols = first_occurrences[["End_Smiles", "MPO_Score", "Quality_Score"]].copy()
            unique_mols = unique_mols.rename(columns={"End_Smiles": "SMILES"})
            csv_path = os.path.join(save_dir, f"unique_{label}_molecules.csv")
            unique_mols.to_csv(csv_path, index=False)
            print(f"Saved unique {label} SMILES to: {csv_path}")

        return episode_agg
    
    episode_targets = process_molecules(target_rows, "targets", save_dir)
    episode_non_targets = process_molecules(non_target_rows, "non_targets", save_dir)
    
    # Merge and forward fill
    max_episode = max(
        episode_targets["Episode_number"].max() if not episode_targets.empty else 0,
        episode_non_targets["Episode_number"].max() if not episode_non_targets.empty else 0
    )
    if max_episode == 0:
        return pd.DataFrame(columns=["Episode_number", "unique_targets", "unique_non_targets", 
                                     "avg_target_mpo", "avg_non_target_mpo",
                                     "avg_target_quality", "avg_non_target_quality"])
    
    all_episodes = pd.DataFrame({"Episode_number": range(1, int(max_episode) + 1)})
    result = pd.merge(all_episodes, episode_targets, on="Episode_number", how="left")
    result = pd.merge(result, episode_non_targets, on="Episode_number", how="left")
    
    # Ensure columns exist before filling
    for col, default in [
        ("unique_targets", 0), ("unique_non_targets", 0),
        ("avg_target_mpo", np.nan), ("avg_non_target_mpo", np.nan),
        ("avg_target_quality", np.nan), ("avg_non_target_quality", np.nan)
    ]:
        if col in result.columns:
            if "unique" in col:
                result[col] = result[col].ffill().fillna(0).astype(int)
            else:
                result[col] = result[col].ffill()
        else:
            result[col] = default
    
    return result


def compute_unique_molecules_over_steps(agent_frames, save_dir=None, property_names=None, property_bounds=None):
    """
    Compute cumulative unique targets and non-targets found across ALL agents combined (overall stats).
    Quality scores are computed only for targets, after all unique molecules are collected.
    
    Args:
        agent_frames: Dict of {label: DataFrame} for each agent
        save_dir: Directory to save CSV outputs
        property_names: List of property names from config
        property_bounds: List of [lower, upper] bounds for each property
    
    Returns:
        DataFrame with columns: ['Step', 'unique_targets', 'unique_non_targets', 
                                 'avg_target_mpo', 'avg_target_quality']
    """
    target_rows = []
    non_target_rows = []
    
    # Collect all target and non-target rows from all agents
    for agent_label, df in agent_frames.items():
        if "End_Smiles" not in df.columns or "Target" not in df.columns:
            continue
        
        df = df.copy()
        
        target_df = df[df["Target"] == True][["Step", "End_Smiles", "MPO_Score", "Scores"]].copy()
        target_rows.append(target_df)
        
        non_target_df = df[df["Target"] == False][["Step", "End_Smiles", "MPO_Score"]].copy()
        non_target_rows.append(non_target_df)
    
    if not target_rows and not non_target_rows:
        return pd.DataFrame(columns=["Step", "unique_targets", "unique_non_targets", 
                                     "avg_target_mpo", "avg_target_quality"])
    
    # Process targets with quality scores
    def process_targets(rows, save_dir=None):
        if not rows:
            return pd.DataFrame(columns=["Step", "unique_targets", "avg_target_mpo", "avg_target_quality"]), None
        
        combined = pd.concat(rows, ignore_index=True)
        combined = combined.sort_values("Step").reset_index(drop=True)
        
        # Filter out invalid SMILES entries
        combined = combined[
            combined["End_Smiles"].notna() & 
            (combined["End_Smiles"] != "") &
            (combined["End_Smiles"] != "nan") &
            (combined["End_Smiles"].str.strip() != "")
        ]
        
        if combined.empty:
            return pd.DataFrame(columns=["Step", "unique_targets", "avg_target_mpo", "avg_target_quality"]), None
        
        # Keep first occurrence of each SMILES (already sorted by Step)
        first_occurrences = combined.drop_duplicates(subset="End_Smiles", keep="first")
        first_occurrences = first_occurrences.sort_values("Step").reset_index(drop=True)
        
        # Compute quality scores for targets AFTER collecting all unique molecules
        if property_bounds is not None and "Scores" in first_occurrences.columns:
            first_occurrences["Quality_Score"] = first_occurrences["Scores"].apply(
                lambda scores: compute_target_quality_score(scores, property_names, property_bounds)
            )
        else:
            first_occurrences["Quality_Score"] = np.nan
        
        # Cumulative counts
        first_occurrences["cumulative_unique"] = range(1, len(first_occurrences) + 1)
        
        # MPO score tracking
        first_occurrences["cumulative_mpo_sum"] = first_occurrences["MPO_Score"].fillna(0).cumsum()
        first_occurrences["cumulative_mpo_count"] = first_occurrences["MPO_Score"].notna().cumsum()
        first_occurrences["avg_mpo"] = (
            first_occurrences["cumulative_mpo_sum"] / 
            first_occurrences["cumulative_mpo_count"].replace(0, np.nan)
        )
        
        # Quality score tracking
        first_occurrences["cumulative_quality_sum"] = first_occurrences["Quality_Score"].fillna(0).cumsum()
        first_occurrences["cumulative_quality_count"] = first_occurrences["Quality_Score"].notna().cumsum()
        first_occurrences["avg_quality"] = (
            first_occurrences["cumulative_quality_sum"] / 
            first_occurrences["cumulative_quality_count"].replace(0, np.nan)
        )
        
        # Get last value per step
        step_agg = first_occurrences.groupby("Step", as_index=False).agg({
            "cumulative_unique": "max",
            "avg_mpo": "last",
            "avg_quality": "last"
        })
        
        step_agg = step_agg.rename(columns={
            "cumulative_unique": "unique_targets",
            "avg_mpo": "avg_target_mpo",
            "avg_quality": "avg_target_quality"
        })
        
        print(f"Total unique targets found: {len(first_occurrences)}")
        
        # Print quality stats if available
        valid_quality = first_occurrences["Quality_Score"].dropna()
        if len(valid_quality) > 0:
            print(f"  Target quality scores - mean: {valid_quality.mean():.3f}, min: {valid_quality.min():.3f}, max: {valid_quality.max():.3f}")
        
        # Return unique molecules for saving
        unique_mols = first_occurrences[["End_Smiles", "MPO_Score", "Quality_Score"]].copy()
        unique_mols = unique_mols.rename(columns={"End_Smiles": "SMILES"})
        
        return step_agg, unique_mols
    
    # Process non-targets (no quality scores needed)
    def process_non_targets(rows, save_dir=None):
        if not rows:
            return pd.DataFrame(columns=["Step", "unique_non_targets"]), None
        
        combined = pd.concat(rows, ignore_index=True)
        combined = combined.sort_values("Step").reset_index(drop=True)
        
        # Filter out invalid SMILES entries
        combined = combined[
            combined["End_Smiles"].notna() & 
            (combined["End_Smiles"] != "") &
            (combined["End_Smiles"] != "nan") &
            (combined["End_Smiles"].str.strip() != "")
        ]
        
        if combined.empty:
            return pd.DataFrame(columns=["Step", "unique_non_targets"]), None
        
        # Keep first occurrence of each SMILES
        first_occurrences = combined.drop_duplicates(subset="End_Smiles", keep="first")
        first_occurrences = first_occurrences.sort_values("Step").reset_index(drop=True)
        
        # Cumulative counts
        first_occurrences["cumulative_unique"] = range(1, len(first_occurrences) + 1)
        
        # Get last value per step
        step_agg = first_occurrences.groupby("Step", as_index=False).agg({
            "cumulative_unique": "max"
        })
        
        step_agg = step_agg.rename(columns={"cumulative_unique": "unique_non_targets"})
        
        print(f"Total unique non-targets found: {len(first_occurrences)}")
        
        # Return unique molecules for saving
        unique_mols = first_occurrences[["End_Smiles", "MPO_Score"]].copy()
        unique_mols = unique_mols.rename(columns={"End_Smiles": "SMILES"})
        
        return step_agg, unique_mols
    
    step_targets, unique_target_mols = process_targets(target_rows, save_dir)
    step_non_targets, unique_non_target_mols = process_non_targets(non_target_rows, save_dir)
    
    # Save unique molecules to CSV
    if save_dir is not None:
        ensure_dir(save_dir)
        if unique_target_mols is not None and not unique_target_mols.empty:
            csv_path = os.path.join(save_dir, "unique_target_molecules.csv")
            unique_target_mols.to_csv(csv_path, index=False)
            print(f"Saved unique target molecules to: {csv_path}")
        if unique_non_target_mols is not None and not unique_non_target_mols.empty:
            csv_path = os.path.join(save_dir, "unique_non_target_molecules.csv")
            unique_non_target_mols.to_csv(csv_path, index=False)
            print(f"Saved unique non-target molecules to: {csv_path}")
    
    # Merge and forward fill
    max_step = max(
        step_targets["Step"].max() if not step_targets.empty else 0,
        step_non_targets["Step"].max() if not step_non_targets.empty else 0
    )
    if max_step == 0:
        return pd.DataFrame(columns=["Step", "unique_targets", "unique_non_targets", 
                                     "avg_target_mpo", "avg_target_quality"])
    
    all_steps = pd.DataFrame({"Step": range(1, int(max_step) + 1)})
    result = pd.merge(all_steps, step_targets, on="Step", how="left")
    result = pd.merge(result, step_non_targets, on="Step", how="left")
    
    # Ensure columns exist before filling
    for col, default in [
        ("unique_targets", 0), ("unique_non_targets", 0),
        ("avg_target_mpo", np.nan), ("avg_target_quality", np.nan)
    ]:
        if col in result.columns:
            if "unique" in col:
                result[col] = result[col].ffill().fillna(0).astype(int)
            else:
                result[col] = result[col].ffill()
        else:
            result[col] = default
    
    return result


def compute_target_quality_score(scores, property_names, property_bounds):
    """
    Compute target quality score based on Gaussian distributions centered on property bounds.
    
    For each property:
    - The Gaussian is centered at the midpoint of the bounds: mean = (lower + upper) / 2
    - The standard deviation is set so that the bounds represent ~2 sigma (95% of distribution)
      i.e., std = (upper - lower) / 4
    - The score for each property is the PDF value normalized by the max PDF (at the mean)
    
    Args:
        scores: List of property scores for a molecule, OR a pandas Series/dict with property values
        property_names: List of property names (used to extract values when scores is a Series/dict)
        property_bounds: List of [lower, upper] bounds for each property
    
    Returns:
        float: Quality score between 0 and 1, where 1 is perfect (all properties at their ideal values)
    """
    if scores is None or property_bounds is None:
        return np.nan
    
    # Handle case where scores is a pandas Series or dict (individual columns)
    if isinstance(scores, (pd.Series, dict)):
        score_values = []
        for prop_name in property_names:
            if prop_name in scores:
                score_values.append(scores[prop_name])
            else:
                score_values.append(None)
        scores = score_values
    
    if not isinstance(scores, list) or len(scores) == 0:
        return np.nan
    
    n_props = min(len(scores), len(property_bounds))
    if n_props == 0:
        return np.nan
    
    total_quality = 0.0
    valid_props = 0
    
    for i in range(n_props):
        score = scores[i]
        bounds = property_bounds[i]
        
        if score is None or pd.isna(score):
            continue
        
        if bounds is None or len(bounds) < 2:
            continue
        
        lower, upper = bounds[0], bounds[1]
        
        # Handle edge case where bounds are equal
        if lower == upper:
            # Perfect score if exactly at the bound, 0 otherwise
            quality = 1.0 if abs(score - lower) < 1e-6 else 0.0
        else:
            # Gaussian centered at midpoint of bounds
            mean = (lower + upper) / 2.0
            # Set std so bounds are approximately at 2 sigma (95% coverage)
            std = (upper - lower) / 4.0
            
            # Compute the Gaussian PDF at the score
            pdf_at_score = norm.pdf(score, loc=mean, scale=std)
            # Normalize by max PDF (which occurs at mean)
            pdf_at_mean = norm.pdf(mean, loc=mean, scale=std)
            
            quality = pdf_at_score / pdf_at_mean if pdf_at_mean > 0 else 0.0
        
        total_quality += quality
        valid_props += 1
    
    if valid_props == 0:
        return np.nan
    
    # Average quality across all valid properties (normalized to 0-1)
    return total_quality / valid_props


def compute_target_quality_for_dataframe(df, property_names, property_bounds):
    """
    Compute target quality scores for all rows in a DataFrame.
    
    Args:
        df: DataFrame with either a 'Scores' column containing lists of property scores,
            OR individual columns for each property (column names matching property_names)
        property_names: List of property names
        property_bounds: List of [lower, upper] bounds for each property
    
    Returns:
        Series of quality scores
    """
    if "Scores" in df.columns:
        # Original behavior: use the Scores column containing lists
        return df["Scores"].apply(
            lambda scores: compute_target_quality_score(scores, property_names, property_bounds)
        )
    else:
        # New behavior: extract values from individual property columns
        return df.apply(
            lambda row: compute_target_quality_score(row, property_names, property_bounds),
            axis=1
        )


def rolling_mean(y, window):
    if len(y) == 0:
        return np.array([])
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _get_property_score(row, agent_label, agent_to_index):
    """
    Extract the property-specific score for an agent from the Scores column.
    Falls back to MPO_Score if Scores is not available or parsing fails.
    """
    scores = row.get("Scores", None)
    agent_idx = agent_to_index.get(agent_label, None)
    
    if scores is not None and agent_idx is not None:
        if isinstance(scores, list) and agent_idx < len(scores):
            score = scores[agent_idx]
            if score is not None and not pd.isna(score):
                return score
    
    # Fallback to MPO_Score
    return row.get("MPO_Score", np.nan)


def compute_noop_per_episode(agent_frames):
    """
    Compute no-op and invalid action statistics per episode across all agents.
    Uses Is_Noop and Is_Invalid flags directly from the data.
    
    Returns a dict with:
        - 'per_agent': {agent_label: DataFrame with Episode_number, noop_count, invalid_count, total_steps, noop_rate, invalid_rate}
        - 'combined': DataFrame with aggregated stats across all agents
    """
    per_agent_stats = {}
    combined_rows = []
    
    for agent_label, df in agent_frames.items():
        # Group by episode
        episode_stats = df.groupby("Episode_number").agg({
            "Is_Noop": "sum",
            "Is_Invalid": "sum",
            "Step": "count"
        }).reset_index()
        episode_stats = episode_stats.rename(columns={
            "Is_Noop": "noop_count",
            "Is_Invalid": "invalid_count",
            "Step": "total_steps"
        })
        episode_stats["noop_rate"] = episode_stats["noop_count"] / episode_stats["total_steps"]
        episode_stats["invalid_rate"] = episode_stats["invalid_count"] / episode_stats["total_steps"]
        episode_stats["agent"] = agent_label
        
        per_agent_stats[agent_label] = episode_stats
        combined_rows.append(episode_stats)
    
    # Combine all agents
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_agg = combined.groupby("Episode_number").agg({
            "noop_count": "sum",
            "invalid_count": "sum",
            "total_steps": "sum"
        }).reset_index()
        combined_agg["noop_rate"] = combined_agg["noop_count"] / combined_agg["total_steps"]
        combined_agg["invalid_rate"] = combined_agg["invalid_count"] / combined_agg["total_steps"]
    else:
        combined_agg = pd.DataFrame(columns=["Episode_number", "noop_count", "invalid_count", "total_steps", "noop_rate", "invalid_rate"])
    
    return {
        "per_agent": per_agent_stats,
        "combined": combined_agg
    }


def compute_noop_per_step(agent_frames):
    """
    Compute no-op and invalid action statistics per global step across all agents.
    Uses Is_Noop and Is_Invalid flags directly from the data.
    
    Returns a dict with:
        - 'per_agent': {agent_label: DataFrame with Step, noop_count, invalid_count, total_count, noop_rate, invalid_rate}
        - 'combined': DataFrame with aggregated stats across all agents
    """
    per_agent_stats = {}
    combined_rows = []
    
    for agent_label, df in agent_frames.items():
        # Group by step
        step_stats = df.groupby("Step").agg({
            "Is_Noop": "sum",
            "Is_Invalid": "sum",
            "Episode_number": "count"
        }).reset_index()
        step_stats = step_stats.rename(columns={
            "Is_Noop": "noop_count",
            "Is_Invalid": "invalid_count",
            "Episode_number": "total_count"
        })
        step_stats["noop_rate"] = step_stats["noop_count"] / step_stats["total_count"]
        step_stats["invalid_rate"] = step_stats["invalid_count"] / step_stats["total_count"]
        step_stats["agent"] = agent_label
        
        per_agent_stats[agent_label] = step_stats
        combined_rows.append(step_stats)
    
    # Combine all agents
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_agg = combined.groupby("Step").agg({
            "noop_count": "sum",
            "invalid_count": "sum",
            "total_count": "sum"
        }).reset_index()
        combined_agg["noop_rate"] = combined_agg["noop_count"] / combined_agg["total_count"]
        combined_agg["invalid_rate"] = combined_agg["invalid_count"] / combined_agg["total_count"]
    else:
        combined_agg = pd.DataFrame(columns=["Step", "noop_count", "invalid_count", "total_count", "noop_rate", "invalid_rate"])
    
    return {
        "per_agent": per_agent_stats,
        "combined": combined_agg
    }


def detect_agent_files(data_dir):
    """
    Dynamically detect agent CSV files in the data directory.
    Looks for files matching pattern 'agent_*.csv' and returns them sorted.
    """
    if not os.path.isdir(data_dir):
        return []
    
    agent_files = []
    for fname in os.listdir(data_dir):
        if fname.startswith("agent_") and fname.endswith(".csv"):
            agent_files.append(fname)
    
    # Sort by agent number (agent_0.csv, agent_1.csv, etc.)
    def extract_agent_num(filename):
        try:
            # Extract number from 'agent_X.csv'
            num_str = filename.replace("agent_", "").replace(".csv", "")
            return int(num_str)
        except ValueError:
            return float('inf')  # Put non-numeric ones at the end
    
    agent_files.sort(key=extract_agent_num)
    return agent_files


def extract_molecule_properties(data_dir, config_path=None, property_names=None, 
                                 output_path=None, agents=None):
    """
    Extract unique molecules with their properties from agent CSV files.
    
    Parses the 'Scores' column (stored as a list string) into separate property
    columns, and produces a deduplicated table of molecules.
    
    Args:
        data_dir: Folder containing agent_*.csv files
        config_path: Path to config file (to get property names)
        property_names: Explicit property names (overrides config)
        output_path: Where to save the result CSV. If None, saved to data_dir/molecules_with_properties.csv
        agents: List of agent filenames. Auto-detected if None.
    
    Returns:
        DataFrame with columns: smiles, target, agent, prop1, prop2, ..., mpo_score
    """
    # Resolve property names from config if not given
    if property_names is None and config_path is not None:
        config = load_config(config_path)
        property_names = get_property_names_from_config(config)
    
    # Auto-detect agent files
    if agents is None:
        agents = detect_agent_files(data_dir)
    if not agents:
        raise FileNotFoundError(f"No agent_*.csv files found in {data_dir}")
    
    all_rows = []
    for fname in agents:
        path = os.path.join(data_dir, fname)
        df = load_agent_csv(path)
        
        # Determine the SMILES column (prefer End_Smiles as that's the result)
        smiles_col = "End_Smiles" if "End_Smiles" in df.columns else "Start_Smiles"
        
        for _, row in df.iterrows():
            smi = row.get(smiles_col)
            if pd.isna(smi) or not str(smi).strip():
                continue
            
            entry = {
                "smiles": str(smi).strip(),
                "target": row.get("Target", False),
                "agent": fname.replace(".csv", ""),
            }
            
            # Parse Scores column
            scores_raw = row.get("Scores")
            try:
                if scores_raw is not None and str(scores_raw).strip() not in ('', 'nan'):
                    scores = ast.literal_eval(str(scores_raw))
                    if isinstance(scores, (list, tuple)):
                        for i, val in enumerate(scores):
                            col_name = property_names[i] if property_names and i < len(property_names) else f"prop_{i}"
                            entry[col_name] = val
            except (ValueError, SyntaxError):
                pass
            
            # Include MPO_Score if present
            mpo = row.get("MPO_Score")
            try:
                if mpo is not None and str(mpo).strip() not in ('', 'nan'):
                    entry["mpo_score"] = float(mpo)
            except (ValueError, TypeError):
                pass
            
            all_rows.append(entry)
    
    result_df = pd.DataFrame(all_rows)
    
    # Deduplicate: keep first occurrence per unique SMILES
    result_df = result_df.drop_duplicates(subset=["smiles"], keep="first")
    result_df = result_df.sort_values("smiles").reset_index(drop=True)
    
    if output_path is None:
        output_path = os.path.join(data_dir, "molecules_with_properties.csv")
    
    ensure_dir(os.path.dirname(output_path))
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} unique molecules to {output_path}")
    
    return result_df


def compute_validity_over_steps(agent_frames):
    """
    Compute cumulative valid/invalid counts and ratios across all agents by step.
    
    Args:
        agent_frames: Dict of {label: DataFrame} for each agent
    
    Returns:
        DataFrame with columns: ['Step', 'total_actions', 'valid_count', 'invalid_count', 
                                 'valid_ratio', 'invalid_ratio']
    """
    all_rows = []
    
    for agent_label, df in agent_frames.items():
        if "Step" not in df.columns or "Is_Invalid" not in df.columns:
            continue
        
        df_subset = df[["Step", "Is_Invalid"]].copy()
        all_rows.append(df_subset)
    
    if not all_rows:
        return pd.DataFrame(columns=["Step", "total_actions", "valid_count", "invalid_count", 
                                     "valid_ratio", "invalid_ratio"])
    
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("Step").reset_index(drop=True)
    
    # Group by step and compute counts
    step_agg = combined.groupby("Step", as_index=False).agg({
        "Is_Invalid": ["count", "sum"]
    })
    step_agg.columns = ["Step", "total_actions", "invalid_count"]
    step_agg["valid_count"] = step_agg["total_actions"] - step_agg["invalid_count"]
    
    # Compute cumulative sums
    step_agg["cum_total"] = step_agg["total_actions"].cumsum()
    step_agg["cum_valid"] = step_agg["valid_count"].cumsum()
    step_agg["cum_invalid"] = step_agg["invalid_count"].cumsum()
    
    # Compute ratios
    step_agg["valid_ratio"] = step_agg["cum_valid"] / step_agg["cum_total"]
    step_agg["invalid_ratio"] = step_agg["cum_invalid"] / step_agg["cum_total"]
    
    # Fill in missing steps
    max_step = int(step_agg["Step"].max())
    all_steps = pd.DataFrame({"Step": range(1, max_step + 1)})
    result = pd.merge(all_steps, step_agg, on="Step", how="left")
    
    # Forward fill ratios for steps with no data
    for col in ["valid_ratio", "invalid_ratio", "cum_total", "cum_valid", "cum_invalid"]:
        result[col] = result[col].ffill().fillna(0)
    
    return result


def compute_binary_mpo_over_steps(agent_frames, property_bounds=None):
    """
    Compute cumulative average MPO score over steps using the existing MPO_Score column.
    
    Args:
        agent_frames: Dict of {label: DataFrame} for each agent
        property_bounds: List of [lower, upper] bounds for each property (for n_properties count only)
    
    Returns:
        DataFrame with columns: ['Step', 'avg_mpo_score', 'n_molecules', 'cum_avg_props_satisfied']
    """
    all_rows = []
    
    for agent_label, df in agent_frames.items():
        if "Step" not in df.columns or "MPO_Score" not in df.columns:
            continue
        
        cols_to_use = ["Step", "MPO_Score"]
        if "Is_Invalid" in df.columns:
            cols_to_use.append("Is_Invalid")
        
        df_subset = df[cols_to_use].copy()
        df_subset["agent"] = agent_label
        all_rows.append(df_subset)
    
    if not all_rows:
        return pd.DataFrame(columns=["Step", "avg_mpo_score", "n_molecules", "cum_avg_props_satisfied"])
    
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("Step").reset_index(drop=True)
    
    # Only consider valid molecules if Is_Invalid column exists
    if "Is_Invalid" in combined.columns:
        combined = combined[combined["Is_Invalid"] == False]
    
    # Filter out rows with missing MPO_Score
    combined = combined[combined["MPO_Score"].notna()]
    
    if combined.empty:
        return pd.DataFrame(columns=["Step", "avg_mpo_score", "n_molecules", "cum_avg_props_satisfied"])
    
    # Group by step and compute statistics
    step_agg = combined.groupby("Step", as_index=False).agg({
        "MPO_Score": ["mean", "count", "sum"]
    })
    step_agg.columns = ["Step", "avg_mpo_score", "n_molecules", "total_mpo_score"]
    
    # Compute cumulative statistics
    step_agg["cum_molecules"] = step_agg["n_molecules"].cumsum()
    step_agg["cum_mpo_sum"] = step_agg["total_mpo_score"].cumsum()
    step_agg["cum_avg_props_satisfied"] = step_agg["cum_mpo_sum"] / step_agg["cum_molecules"]
    
    # Fill in missing steps
    max_step = int(step_agg["Step"].max()) if not step_agg.empty else 0
    if max_step == 0:
        return pd.DataFrame(columns=["Step", "avg_mpo_score", "n_molecules", "cum_avg_props_satisfied"])
    
    all_steps = pd.DataFrame({"Step": range(1, max_step + 1)})
    result = pd.merge(all_steps, step_agg, on="Step", how="left")
    
    # Forward fill for missing steps
    for col in ["cum_avg_props_satisfied", "cum_molecules", "cum_mpo_sum"]:
        result[col] = result[col].ffill().fillna(0)
    
    return result
