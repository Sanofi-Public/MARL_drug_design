#!/usr/bin/env python3
"""
Property Analysis: KDE plots, Property Distributions, and Scaffold Novelty Analysis.

Property Distribution Plots:
    Creates individual histograms for each property column, with target/non-target separation.

KDE Plots:
    Reads molecules_with_properties.csv and creates pairwise KDE plots
    with target molecules overlaid as scatter points.

Scaffold Novelty Analysis:
    Compares lead molecules (from train_file in config) to generated targets:
    - Computes Murcko scaffolds for both sets
    - Reports new scaffold discovery rate and diversity stats
    - Identifies most novel targets (lowest Tanimoto to any lead)
    - Outputs summary CSV and scaffold frequency plots

Usage:
    # Property distributions only
    python property_analysis.py --csv molecules.csv --config config.json --mode dist

    # KDE plots only
    python property_analysis.py --csv molecules.csv --config config.json --mode kde

    # Scaffold analysis only
    python property_analysis.py --csv molecules.csv --config config.json --mode scaffold

    # KDE + Scaffold (default)
    python property_analysis.py --csv molecules.csv --config config.json --mode both

    # All analyses
    python property_analysis.py --csv molecules.csv --config config.json --mode all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from itertools import combinations
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_plot_utils import (
    load_config,
    get_property_names_from_config,
    get_property_bounds_from_config,
    ensure_dir,
)

from rdkit import Chem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from collections import Counter

_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


def _smi_to_image(smiles, size=(250, 180)):
    """Render SMILES as a PIL Image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def _add_mol_inset(fig, ax, smiles, position="lower left", label=""):
    """Add a molecule structure image as an inset on the axes."""
    img = _smi_to_image(smiles)
    if img is None:
        return

    # Position: lower left or lower right
    if position == "lower left":
        box_x, box_y = 0.02, 0.02
        loc = "lower left"
    else:
        box_x, box_y = 0.98, 0.02
        loc = "lower right"

    # Convert PIL to numpy array for OffsetImage
    img_arr = np.array(img)
    imagebox = OffsetImage(img_arr, zoom=0.45)
    ab = AnnotationBbox(imagebox, (box_x, box_y),
                        xycoords="axes fraction",
                        box_alignment=(0.0 if "left" in loc else 1.0, 0.0),
                        bboxprops=dict(boxstyle="round,pad=0.3", 
                                       facecolor="white", edgecolor="gray", alpha=0.9),
                        pad=0.3)
    ax.add_artist(ab)

    # Add label above the image
    label_x = box_x + (0.08 if "left" in loc else -0.08)
    ax.text(label_x, box_y + 0.22, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))


def find_closest_target(lead_smi, target_smiles_list):
    """
    Find the most similar target molecule to a lead by Tanimoto similarity.
    
    Returns:
        (best_target_smi, similarity_score) or (None, 0)
    """
    lead_mol = Chem.MolFromSmiles(lead_smi)
    if lead_mol is None:
        return None, 0.0
    lead_fp = _morgan_gen.GetFingerprint(lead_mol)

    best_smi = None
    best_sim = -1.0

    for tsmi in target_smiles_list:
        tmol = Chem.MolFromSmiles(tsmi)
        if tmol is None:
            continue
        tfp = _morgan_gen.GetFingerprint(tmol)
        sim = DataStructs.TanimotoSimilarity(lead_fp, tfp)
        if sim > best_sim:
            best_sim = sim
            best_smi = tsmi

    return best_smi, best_sim


def plot_kde_pair(df, prop_x, prop_y, bounds, save_dir, 
                  bound_x=None, bound_y=None, start_df=None,
                  lead_smi=None, lead_props=None,
                  target_smi=None, target_props=None):
    """
    Create a 2D KDE plot for a pair of properties with targets overlaid.
    
    Args:
        df: DataFrame with property columns and 'target' column
        prop_x: Name of x-axis property
        prop_y: Name of y-axis property
        bounds: Dict mapping property name -> [min, max]
        save_dir: Output directory
        bound_x: [min, max] for x property (optional)
        bound_y: [min, max] for y property (optional)
        start_df: DataFrame of starting molecules with same property columns (optional)
        lead_smi: SMILES of selected lead molecule (optional)
        lead_props: Dict of property values for lead molecule (optional)
        target_smi: SMILES of closest target molecule (optional)
        target_props: Dict of property values for closest target (optional)
    
    Returns:
        Path to saved plot
    """
    # Drop rows with missing values for these properties
    plot_df = df[[prop_x, prop_y, "target"]].dropna()
    if len(plot_df) < 3:
        print(f"  Skipping {prop_x} vs {prop_y}: too few data points ({len(plot_df)})")
        return None

    x = plot_df[prop_x].values.astype(float)
    y = plot_df[prop_y].values.astype(float)
    targets = plot_df["target"].values

    # Compute KDE (subsample if large for performance)
    try:
        max_kde_points = 500000
        if len(x) > max_kde_points:
            idx_sample = np.random.choice(len(x), size=max_kde_points, replace=False)
            xy = np.vstack([x[idx_sample], y[idx_sample]])
        else:
            xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
    except np.linalg.LinAlgError:
        print(f"  Skipping {prop_x} vs {prop_y}: KDE failed (singular matrix)")
        return None

    # Create evaluation grid — zoom to region of interest
    # Use actual data range (all molecules + starting molecules) for axis limits
    all_x_points = [x.min(), x.max()]
    all_y_points = [y.min(), y.max()]
    if start_df is not None:
        sp = start_df[[prop_x, prop_y]].dropna()
        if len(sp) > 0:
            all_x_points.extend([sp[prop_x].min(), sp[prop_x].max()])
            all_y_points.extend([sp[prop_y].min(), sp[prop_y].max()])

    if all_x_points:
        lo_x, hi_x = min(all_x_points), max(all_x_points)
        span_x = hi_x - lo_x or 1.0
        xmin, xmax = lo_x - 0.15 * span_x, hi_x + 0.15 * span_x
    else:
        xmin, xmax = np.percentile(x, [5, 95])
        pad_x = (xmax - xmin) * 0.2 or 1.0
        xmin, xmax = xmin - pad_x, xmax + pad_x

    if all_y_points:
        lo_y, hi_y = min(all_y_points), max(all_y_points)
        span_y = hi_y - lo_y or 1.0
        ymin, ymax = lo_y - 0.15 * span_y, hi_y + 0.15 * span_y
    else:
        ymin, ymax = np.percentile(y, [5, 95])
        pad_y = (ymax - ymin) * 0.2 or 1.0
        ymin, ymax = ymin - pad_y, ymax + pad_y
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions).reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # KDE contour fill (all molecules)
    cf = ax.contourf(xx, yy, zz, levels=30, cmap="Blues", alpha=0.8)
    cbar = plt.colorbar(cf, ax=ax, label="Density (all molecules)")
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    # Target KDE contour lines
    target_mask = targets.astype(bool)
    n_total_targets = int(target_mask.sum())
    if n_total_targets > 0:
        tx_all = x[target_mask]
        ty_all = y[target_mask]
        ax.scatter(tx_all, ty_all,
                   s=4, c="red", alpha=0.02, edgecolors="none",
                   label=f"Targets ({n_total_targets:,})", zorder=3)

    # Overlay starting molecules (all of them)
    if start_df is not None:
        start_plot = start_df[[prop_x, prop_y]].dropna()
        if len(start_plot) > 0:
            sx = start_plot[prop_x].values.astype(float)
            sy = start_plot[prop_y].values.astype(float)
            ax.scatter(sx, sy,
                       s=45, c="gold", alpha=0.9, marker="*",
                       label=f"Starting mols ({len(sx)})",
                       edgecolors="black", linewidths=0.4, zorder=5)

    # Draw property bounds as rectangle if available (clipped to axis limits)
    if bound_x is not None and bound_y is not None:
        from matplotlib.patches import Rectangle
        clip_x0 = max(bound_x[0], xmin)
        clip_x1 = min(bound_x[1], xmax)
        clip_y0 = max(bound_y[0], ymin)
        clip_y1 = min(bound_y[1], ymax)
        if clip_x1 > clip_x0 and clip_y1 > clip_y0:
            rect = Rectangle(
                (clip_x0, clip_y0),
                clip_x1 - clip_x0,
                clip_y1 - clip_y0,
                linewidth=2, edgecolor="green", facecolor="none",
                linestyle="--", label="Target bounds", zorder=4,
            )
            ax.add_patch(rect)

    ax.set_xlabel(prop_x, fontsize=12)
    ax.set_ylabel(prop_y, fontsize=12)
    ax.set_title(f"KDE: {prop_x} vs {prop_y}  (n={len(plot_df):,})", fontsize=13)

    # Molecule popout: highlight lead and closest target with structure images
    if lead_smi and target_smi and lead_props and target_props:
        lx = lead_props.get(prop_x)
        ly = lead_props.get(prop_y)
        ttx = target_props.get(prop_x)
        tty = target_props.get(prop_y)
        if all(v is not None for v in [lx, ly, ttx, tty]):
            # Mark positions
            ax.scatter([lx], [ly], s=120, c="gold", marker="*", edgecolors="black",
                       linewidths=1.0, zorder=10, label="Selected lead")
            ax.scatter([ttx], [tty], s=120, c="red", marker="D", edgecolors="black",
                       linewidths=1.0, zorder=10, label="Closest target")
            # Draw arrow from lead to target
            ax.annotate("", xy=(ttx, tty), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                        zorder=9)
            # Render molecule images as insets
            _add_mol_inset(fig, ax, lead_smi, position="lower left", label="Lead")
            _add_mol_inset(fig, ax, target_smi, position="lower right", label="Target")

    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"kde_{prop_x}_vs_{prop_y}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# =============================================================================
# Property Distribution Plots
# =============================================================================

def plot_property_distribution(
    df: pd.DataFrame,
    property_name: str,
    save_dir: str,
    bounds: tuple = None,
    lead_df: pd.DataFrame = None,
    bins: int = 50,
    figsize: tuple = (8, 5),
    show_stats: bool = True
) -> str:
    """
    Plot distribution of a single property with optional lead overlay on secondary axis.
    
    Args:
        df: DataFrame with property column (generated molecules)
        property_name: Name of the property column to plot
        save_dir: Output directory for the plot
        bounds: Optional tuple (min, max) for target bounds to draw (dynamically capped)
        lead_df: Optional DataFrame of lead molecules to overlay on secondary axis
        bins: Number of histogram bins
        figsize: Figure size tuple
        show_stats: Whether to show mean/std statistics on plot
    
    Returns:
        Path to saved plot
    """
    if property_name not in df.columns:
        print(f"  Warning: column '{property_name}' not in DataFrame, skipping.")
        return None
    
    values = df[property_name].dropna().values.astype(float)
    if len(values) < 2:
        print(f"  Skipping {property_name}: too few data points ({len(values)})")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dynamically determine x-axis range from data (include leads if available)
    data_min, data_max = np.min(values), np.max(values)
    
    # Get lead values if available
    lead_values = None
    if lead_df is not None and property_name in lead_df.columns:
        lead_values = lead_df[property_name].dropna().values.astype(float)
        if len(lead_values) > 0:
            data_min = min(data_min, np.min(lead_values))
            data_max = max(data_max, np.max(lead_values))
    
    data_range = data_max - data_min
    padding = data_range * 0.05 if data_range > 0 else 1.0
    xlim = (data_min - padding, data_max + padding)
    
    # Plot generated molecules
    ax.hist(values, bins=bins, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5, label=f'Generated ({len(values):,})')
    ax.set_ylabel('Count (Generated)', fontsize=11, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    # Plot leads on secondary axis if available
    if lead_values is not None and len(lead_values) > 0:
        ax2 = ax.twinx()
        ax2.hist(lead_values, bins=bins, alpha=0.5, color='#e74c3c',
                 edgecolor='darkred', linewidth=0.5, label=f'Leads ({len(lead_values):,})')
        ax2.set_ylabel('Count (Leads)', fontsize=11, color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    else:
        ax.legend(loc='upper right', fontsize=9)
    
    # Calculate % within bounds and draw bounds
    pct_in_bounds = None
    if bounds is not None and len(bounds) == 2:
        bound_min, bound_max = bounds
        
        # For visualization, cap to data range
        vis_min = max(bound_min, data_min - padding)
        vis_max = min(bound_max, data_max + padding)
        
        # Calculate % of molecules within bounds
        n_in_bounds = np.sum((values >= bound_min) & (values <= bound_max))
        pct_in_bounds = (n_in_bounds / len(values)) * 100
        
        # Draw bounds
        ax.axvline(vis_min, color='green', linestyle='--', linewidth=2, zorder=5)
        ax.axvline(vis_max, color='green', linestyle='--', linewidth=2, zorder=5)
        ax.axvspan(vis_min, vis_max, alpha=0.15, color='green', zorder=1)
    
    # Stats text
    stats_lines = []
    if show_stats:
        stats_lines.append(f"Gen: μ={np.mean(values):.2f}, σ={np.std(values):.2f}")
        if lead_values is not None and len(lead_values) > 0:
            stats_lines.append(f"Lead: μ={np.mean(lead_values):.2f}, σ={np.std(lead_values):.2f}")
    if pct_in_bounds is not None:
        stats_lines.append(f"In bounds: {pct_in_bounds:.1f}%")
    
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                zorder=10)
    
    ax.set_xlim(xlim)
    ax.set_xlabel(property_name, fontsize=11)
    ax.set_title(f'Distribution: {property_name}  (n={len(values):,})', fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"dist_{property_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path


def plot_all_property_distributions(
    df: pd.DataFrame,
    property_names: list,
    save_dir: str,
    bounds_dict: dict = None,
    lead_df: pd.DataFrame = None,
    bins: int = 50
) -> list:
    """
    Plot distributions for all specified properties, each in a separate file.
    
    Args:
        df: DataFrame with property columns
        property_names: List of property column names to plot
        save_dir: Output directory for plots
        bounds_dict: Optional dict mapping property_name -> (min, max) bounds
        lead_df: Optional DataFrame of lead molecules
        bins: Number of histogram bins
    
    Returns:
        List of paths to saved plots
    """
    ensure_dir(save_dir)
    saved_paths = []
    
    print(f"\nGenerating {len(property_names)} property distribution plots...")
    
    for prop in property_names:
        bounds = bounds_dict.get(prop) if bounds_dict else None
        out_path = plot_property_distribution(
            df=df,
            property_name=prop,
            save_dir=save_dir,
            bounds=bounds,
            lead_df=lead_df,
            bins=bins
        )
        if out_path:
            print(f"  Saved: {out_path}")
            saved_paths.append(out_path)
    
    return saved_paths


def plot_property_grid(
    df: pd.DataFrame,
    property_names: list,
    save_dir: str,
    bounds_dict: dict = None,
    lead_df: pd.DataFrame = None,
    bins: int = 30,
    cols: int = 3
) -> str:
    """
    Plot all property distributions in a single grid figure with lead overlay.
    
    Args:
        df: DataFrame with property columns
        property_names: List of property column names to plot
        save_dir: Output directory
        bounds_dict: Optional dict mapping property_name -> (min, max) bounds
        lead_df: Optional DataFrame of lead molecules
        bins: Number of histogram bins
        cols: Number of columns in the grid
    
    Returns:
        Path to saved plot
    """
    ensure_dir(save_dir)
    
    n_props = len(property_names)
    if n_props == 0:
        return None
    
    rows = (n_props + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, prop in enumerate(property_names):
        ax = axes[i]
        
        if prop not in df.columns:
            ax.text(0.5, 0.5, f"'{prop}' not found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prop)
            continue
        
        values = df[prop].dropna().values.astype(float)
        if len(values) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prop)
            continue
        
        # Dynamically determine x-axis range (include leads if available)
        data_min, data_max = np.min(values), np.max(values)
        
        lead_values = None
        if lead_df is not None and prop in lead_df.columns:
            lead_values = lead_df[prop].dropna().values.astype(float)
            if len(lead_values) > 0:
                data_min = min(data_min, np.min(lead_values))
                data_max = max(data_max, np.max(lead_values))
        
        data_range = data_max - data_min
        padding = data_range * 0.05 if data_range > 0 else 1.0
        xlim = (data_min - padding, data_max + padding)
        
        # Plot generated molecules
        ax.hist(values, bins=bins, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.3, label='Generated')
        ax.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot leads on secondary axis if available
        if lead_values is not None and len(lead_values) > 0:
            ax2 = ax.twinx()
            ax2.hist(lead_values, bins=bins, alpha=0.5, color='#e74c3c',
                     edgecolor='darkred', linewidth=0.3, label='Leads')
            ax2.tick_params(axis='y', labelcolor='#e74c3c')
            ax2.set_ylabel('Leads', fontsize=7, color='#e74c3c')
        
        # Calculate % within bounds and draw bounds
        bounds = bounds_dict.get(prop) if bounds_dict else None
        pct_in_bounds = None
        if bounds is not None and len(bounds) == 2:
            bound_min, bound_max = bounds
            
            # For visualization, cap to data range
            vis_min = max(bound_min, data_min - padding)
            vis_max = min(bound_max, data_max + padding)
            
            # Calculate % of molecules within bounds
            n_in_bounds = np.sum((values >= bound_min) & (values <= bound_max))
            pct_in_bounds = (n_in_bounds / len(values)) * 100
            
            # Draw bounds
            ax.axvline(vis_min, color='green', linestyle='--', linewidth=1.5, zorder=5)
            ax.axvline(vis_max, color='green', linestyle='--', linewidth=1.5, zorder=5)
            ax.axvspan(vis_min, vis_max, alpha=0.15, color='green', zorder=1)
        
        # Add % text
        if pct_in_bounds is not None:
            ax.text(0.02, 0.98, f"In bounds: {pct_in_bounds:.1f}%", transform=ax.transAxes, 
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), zorder=10)
        
        ax.set_xlim(xlim)
        ax.set_xlabel(prop, fontsize=9)
        ax.set_ylabel('Generated', fontsize=7, color='steelblue')
        ax.set_title(f'{prop} (n={len(values):,})', fontsize=10)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, "property_distributions_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"  Saved grid: {out_path}")
    return out_path


# =============================================================================
# Scaffold Novelty & Diversity Analysis
# =============================================================================

def get_murcko_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Compute Murcko scaffold for a SMILES string.
    
    Args:
        smiles: Input SMILES string
        generic: If True, return generic scaffold (all atoms -> C, all bonds -> single)
    
    Returns:
        Scaffold SMILES or empty string if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def compute_scaffold_set(smiles_list: list, generic: bool = False) -> dict:
    """
    Compute scaffolds for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        generic: If True, use generic scaffolds
    
    Returns:
        Dict with 'scaffolds' (list), 'unique_scaffolds' (set), 
        'scaffold_counts' (Counter), 'smiles_to_scaffold' (dict)
    """
    scaffolds = []
    smiles_to_scaffold = {}
    
    for smi in smiles_list:
        scaf = get_murcko_scaffold(smi, generic=generic)
        scaffolds.append(scaf)
        smiles_to_scaffold[smi] = scaf
    
    unique_scaffolds = set(s for s in scaffolds if s)
    scaffold_counts = Counter(s for s in scaffolds if s)
    
    return {
        "scaffolds": scaffolds,
        "unique_scaffolds": unique_scaffolds,
        "scaffold_counts": scaffold_counts,
        "smiles_to_scaffold": smiles_to_scaffold
    }


def compute_pairwise_tanimoto(smiles_list: list, sample_size: int = 1000) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarities for internal diversity.
    
    Args:
        smiles_list: List of SMILES
        sample_size: Max number of molecules to sample (for performance)
    
    Returns:
        Array of pairwise similarities
    """
    if len(smiles_list) <= 1:
        return np.array([])
    
    # Sample if too large
    if len(smiles_list) > sample_size:
        smiles_list = list(np.random.choice(smiles_list, sample_size, replace=False))
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(_morgan_gen.GetFingerprint(mol))
    
    if len(fps) < 2:
        return np.array([])
    
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    return np.array(similarities)


def find_min_tanimoto_to_leads(target_smiles: str, lead_fps: list) -> float:
    """
    Find minimum Tanimoto similarity between a target and all lead molecules.
    Lower = more novel.
    """
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None or not lead_fps:
        return 1.0
    
    target_fp = _morgan_gen.GetFingerprint(mol)
    min_sim = min(DataStructs.TanimotoSimilarity(target_fp, lfp) for lfp in lead_fps)
    return min_sim


def scaffold_novelty_analysis(
    lead_file: str,
    target_csv: str,
    save_dir: str = None,
    smiles_col: str = "smiles",
    target_col: str = "target",
    top_novel: int = 20,
    generic_scaffolds: bool = False
) -> dict:
    """
    Comprehensive scaffold novelty and diversity analysis.
    
    Compares lead molecules to generated targets:
    - Computes Murcko scaffolds for both sets
    - Reports new scaffold discovery rate
    - Computes diversity statistics
    - Identifies most novel targets (lowest Tanimoto to any lead)
    - Outputs summary CSV and scaffold frequency plot
    
    Args:
        lead_file: Path to lead molecules file (CSV or SMI)
        target_csv: Path to generated molecules CSV with 'target' column
        save_dir: Output directory (default: same as target_csv directory)
        smiles_col: Name of SMILES column
        target_col: Name of target boolean column
        top_novel: Number of most novel targets to report
        generic_scaffolds: If True, use generic (atom-type-agnostic) scaffolds
    
    Returns:
        Dict with analysis results and output file paths
    """
    print("\n" + "=" * 70)
    print("SCAFFOLD NOVELTY & DIVERSITY ANALYSIS")
    print("=" * 70)
    
    # Setup output directory
    if save_dir is None:
        save_dir = os.path.dirname(target_csv)
    ensure_dir(save_dir)
    
    # -------------------------------------------------------------------------
    # 1. Load Lead Molecules
    # -------------------------------------------------------------------------
    print(f"\n[1] Loading lead molecules from: {lead_file}")
    
    if lead_file.endswith(".smi"):
        lead_smiles = []
        with open(lead_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    lead_smiles.append(parts[0])
    else:
        lead_df = pd.read_csv(lead_file)
        # Try common column names
        smi_col = None
        for c in [smiles_col, "smiles", "SMILES", "Smiles", "canonical_smiles"]:
            if c in lead_df.columns:
                smi_col = c
                break
        if smi_col is None:
            raise ValueError(f"Could not find SMILES column in {lead_file}")
        lead_smiles = lead_df[smi_col].dropna().tolist()
    
    lead_smiles = list(set(lead_smiles))  # Deduplicate
    print(f"    Loaded {len(lead_smiles)} unique lead molecules")
    
    # -------------------------------------------------------------------------
    # 2. Load Target Molecules
    # -------------------------------------------------------------------------
    print(f"\n[2] Loading target molecules from: {target_csv}")
    
    target_df = pd.read_csv(target_csv)
    
    # Find SMILES column
    smi_col = None
    for c in [smiles_col, "smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if c in target_df.columns:
            smi_col = c
            break
    if smi_col is None:
        raise ValueError(f"Could not find SMILES column in {target_csv}")
    
    # Filter to targets only if target column exists
    if target_col in target_df.columns:
        target_molecules_df = target_df[target_df[target_col] == True].copy()
    else:
        target_molecules_df = target_df.copy()
    
    target_smiles = target_molecules_df[smi_col].dropna().tolist()
    target_smiles = list(set(target_smiles))  # Deduplicate
    print(f"    Loaded {len(target_smiles)} unique target molecules")
    
    # -------------------------------------------------------------------------
    # 3. Compute Scaffolds
    # -------------------------------------------------------------------------
    scaffold_type = "generic" if generic_scaffolds else "Murcko"
    print(f"\n[3] Computing {scaffold_type} scaffolds...")
    
    lead_scaffold_data = compute_scaffold_set(lead_smiles, generic=generic_scaffolds)
    target_scaffold_data = compute_scaffold_set(target_smiles, generic=generic_scaffolds)
    
    lead_scaffolds = lead_scaffold_data["unique_scaffolds"]
    target_scaffolds = target_scaffold_data["unique_scaffolds"]
    
    print(f"    Lead scaffolds:   {len(lead_scaffolds)} unique")
    print(f"    Target scaffolds: {len(target_scaffolds)} unique")
    
    # -------------------------------------------------------------------------
    # 4. Scaffold Novelty Analysis
    # -------------------------------------------------------------------------
    print(f"\n[4] Scaffold novelty analysis...")
    
    # New scaffolds = in targets but not in leads
    new_scaffolds = target_scaffolds - lead_scaffolds
    shared_scaffolds = target_scaffolds & lead_scaffolds
    
    # Scaffold discovery rate
    if len(target_scaffolds) > 0:
        scaffold_novelty_rate = len(new_scaffolds) / len(target_scaffolds) * 100
    else:
        scaffold_novelty_rate = 0.0
    
    print(f"    New scaffolds discovered:    {len(new_scaffolds)}")
    print(f"    Shared scaffolds with leads: {len(shared_scaffolds)}")
    print(f"    Scaffold novelty rate:       {scaffold_novelty_rate:.1f}%")
    
    # Count targets per scaffold category
    n_targets_new_scaffold = sum(
        1 for smi in target_smiles 
        if target_scaffold_data["smiles_to_scaffold"].get(smi, "") in new_scaffolds
    )
    n_targets_shared_scaffold = len(target_smiles) - n_targets_new_scaffold
    
    print(f"    Targets with new scaffolds:    {n_targets_new_scaffold} ({n_targets_new_scaffold/len(target_smiles)*100:.1f}%)")
    print(f"    Targets with shared scaffolds: {n_targets_shared_scaffold} ({n_targets_shared_scaffold/len(target_smiles)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 5. Diversity Statistics
    # -------------------------------------------------------------------------
    print(f"\n[5] Computing diversity statistics...")
    
    # Internal diversity of targets
    target_pairwise = compute_pairwise_tanimoto(target_smiles)
    if len(target_pairwise) > 0:
        target_avg_sim = np.mean(target_pairwise)
        target_internal_diversity = 1 - target_avg_sim
        target_sim_std = np.std(target_pairwise)
    else:
        target_avg_sim = 0.0
        target_internal_diversity = 1.0
        target_sim_std = 0.0
    
    # Internal diversity of leads (for comparison)
    lead_pairwise = compute_pairwise_tanimoto(lead_smiles)
    if len(lead_pairwise) > 0:
        lead_avg_sim = np.mean(lead_pairwise)
        lead_internal_diversity = 1 - lead_avg_sim
    else:
        lead_avg_sim = 0.0
        lead_internal_diversity = 1.0
    
    print(f"    Target internal diversity: {target_internal_diversity:.3f} (avg Tanimoto: {target_avg_sim:.3f})")
    print(f"    Lead internal diversity:   {lead_internal_diversity:.3f} (avg Tanimoto: {lead_avg_sim:.3f})")
    
    # Scaffold diversity (unique scaffolds per molecule)
    target_scaffold_diversity = len(target_scaffolds) / len(target_smiles) if target_smiles else 0
    lead_scaffold_diversity = len(lead_scaffolds) / len(lead_smiles) if lead_smiles else 0
    
    print(f"    Target scaffold diversity: {target_scaffold_diversity:.3f} (scaffolds/mol)")
    print(f"    Lead scaffold diversity:   {lead_scaffold_diversity:.3f} (scaffolds/mol)")
    
    # -------------------------------------------------------------------------
    # 6. Find Most Novel Targets
    # -------------------------------------------------------------------------
    print(f"\n[6] Identifying {top_novel} most novel targets...")
    
    # Precompute lead fingerprints
    lead_fps = []
    for smi in lead_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            lead_fps.append(_morgan_gen.GetFingerprint(mol))
    
    # Compute min similarity to leads for each target
    novelty_scores = []
    for smi in target_smiles:
        min_sim = find_min_tanimoto_to_leads(smi, lead_fps)
        novelty_scores.append({
            "smiles": smi,
            "min_tanimoto_to_lead": min_sim,
            "novelty_score": 1 - min_sim,
            "scaffold": target_scaffold_data["smiles_to_scaffold"].get(smi, ""),
            "is_new_scaffold": target_scaffold_data["smiles_to_scaffold"].get(smi, "") in new_scaffolds
        })
    
    novelty_df = pd.DataFrame(novelty_scores)
    novelty_df = novelty_df.sort_values("novelty_score", ascending=False)
    
    # Top novel targets
    top_novel_df = novelty_df.head(top_novel)
    
    print(f"\n    Top {len(top_novel_df)} most novel targets:")
    print(f"    {'SMILES':<50} {'Min Tan.':<10} {'Novel?':<8}")
    print("    " + "-" * 68)
    for _, row in top_novel_df.head(10).iterrows():
        smi_short = row["smiles"][:47] + "..." if len(row["smiles"]) > 50 else row["smiles"]
        scaffold_flag = "NEW" if row["is_new_scaffold"] else ""
        print(f"    {smi_short:<50} {row['min_tanimoto_to_lead']:.4f}     {scaffold_flag}")
    
    # -------------------------------------------------------------------------
    # 7. Output Summary CSV
    # -------------------------------------------------------------------------
    print(f"\n[7] Writing output files...")
    
    # Full novelty results
    novelty_csv_path = os.path.join(save_dir, "scaffold_novelty_analysis.csv")
    novelty_df.to_csv(novelty_csv_path, index=False)
    print(f"    Saved: {novelty_csv_path}")
    
    # Summary statistics CSV
    summary_data = {
        "metric": [
            "n_lead_molecules",
            "n_target_molecules",
            "n_lead_scaffolds",
            "n_target_scaffolds",
            "n_new_scaffolds",
            "n_shared_scaffolds",
            "scaffold_novelty_rate_pct",
            "n_targets_with_new_scaffold",
            "pct_targets_with_new_scaffold",
            "target_internal_diversity",
            "lead_internal_diversity",
            "target_scaffold_diversity",
            "lead_scaffold_diversity",
            "target_avg_pairwise_tanimoto",
            "avg_min_tanimoto_to_lead",
            "most_novel_smiles",
            "most_novel_min_tanimoto"
        ],
        "value": [
            len(lead_smiles),
            len(target_smiles),
            len(lead_scaffolds),
            len(target_scaffolds),
            len(new_scaffolds),
            len(shared_scaffolds),
            f"{scaffold_novelty_rate:.2f}",
            n_targets_new_scaffold,
            f"{n_targets_new_scaffold/len(target_smiles)*100:.2f}" if target_smiles else "0.00",
            f"{target_internal_diversity:.4f}",
            f"{lead_internal_diversity:.4f}",
            f"{target_scaffold_diversity:.4f}",
            f"{lead_scaffold_diversity:.4f}",
            f"{target_avg_sim:.4f}",
            f"{novelty_df['min_tanimoto_to_lead'].mean():.4f}" if len(novelty_df) > 0 else "N/A",
            novelty_df.iloc[0]["smiles"] if len(novelty_df) > 0 else "N/A",
            f"{novelty_df.iloc[0]['min_tanimoto_to_lead']:.4f}" if len(novelty_df) > 0 else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, "scaffold_analysis_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"    Saved: {summary_csv_path}")
    
    # -------------------------------------------------------------------------
    # 8. Scaffold Frequency Plot
    # -------------------------------------------------------------------------
    print(f"\n[8] Generating scaffold frequency plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot A: Top scaffold frequencies (targets)
    ax1 = axes[0]
    top_scaffolds = target_scaffold_data["scaffold_counts"].most_common(20)
    if top_scaffolds:
        scaffolds_labels = [s[0][:30] + "..." if len(s[0]) > 30 else s[0] for s in top_scaffolds]
        scaffold_counts = [s[1] for s in top_scaffolds]
        colors = ["#2ecc71" if top_scaffolds[i][0] in new_scaffolds else "#3498db" 
                  for i in range(len(top_scaffolds))]
        
        y_pos = np.arange(len(scaffolds_labels))
        ax1.barh(y_pos, scaffold_counts, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(scaffolds_labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Frequency", fontsize=11)
        ax1.set_title("Top 20 Target Scaffolds\n(green = novel, blue = shared)", fontsize=12)
    
    # Plot B: Novelty distribution histogram
    ax2 = axes[1]
    if len(novelty_df) > 0:
        ax2.hist(novelty_df["min_tanimoto_to_lead"], bins=30, color="#9b59b6", 
                 edgecolor="black", alpha=0.7)
        ax2.axvline(novelty_df["min_tanimoto_to_lead"].mean(), color="red", 
                    linestyle="--", linewidth=2, label=f"Mean: {novelty_df['min_tanimoto_to_lead'].mean():.3f}")
        ax2.axvline(novelty_df["min_tanimoto_to_lead"].median(), color="orange", 
                    linestyle="--", linewidth=2, label=f"Median: {novelty_df['min_tanimoto_to_lead'].median():.3f}")
        ax2.set_xlabel("Min Tanimoto to Lead", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Target Novelty Distribution\n(lower = more novel)", fontsize=12)
        ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "scaffold_frequency_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {plot_path}")
    
    # -------------------------------------------------------------------------
    # 9. Optional: Top Novel Molecules Grid
    # -------------------------------------------------------------------------
    print(f"\n[9] Generating top novel molecules grid...")
    
    top_mols = []
    top_legends = []
    for _, row in top_novel_df.head(12).iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            top_mols.append(mol)
            scaffold_tag = " [NEW]" if row["is_new_scaffold"] else ""
            top_legends.append(f"Tan={row['min_tanimoto_to_lead']:.3f}{scaffold_tag}")
    
    if top_mols:
        img = Draw.MolsToGridImage(top_mols, molsPerRow=4, subImgSize=(300, 250),
                                   legends=top_legends)
        grid_path = os.path.join(save_dir, "top_novel_molecules.png")
        img.save(grid_path)
        print(f"    Saved: {grid_path}")
    else:
        grid_path = None
    
    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
Key Findings:
  - Scaffold novelty rate: {scaffold_novelty_rate:.1f}% ({len(new_scaffolds)} new scaffolds)
  - {n_targets_new_scaffold}/{len(target_smiles)} targets ({n_targets_new_scaffold/len(target_smiles)*100:.1f}%) have novel scaffolds
  - Target internal diversity: {target_internal_diversity:.3f}
  - Average min Tanimoto to leads: {novelty_df['min_tanimoto_to_lead'].mean():.3f}
  - Most novel target: {novelty_df.iloc[0]['smiles'][:50]}... (Tan={novelty_df.iloc[0]['min_tanimoto_to_lead']:.3f})

Output files:
  - {novelty_csv_path}
  - {summary_csv_path}
  - {plot_path}
""")
    
    return {
        "n_leads": len(lead_smiles),
        "n_targets": len(target_smiles),
        "n_lead_scaffolds": len(lead_scaffolds),
        "n_target_scaffolds": len(target_scaffolds),
        "n_new_scaffolds": len(new_scaffolds),
        "scaffold_novelty_rate": scaffold_novelty_rate,
        "target_internal_diversity": target_internal_diversity,
        "lead_internal_diversity": lead_internal_diversity,
        "novelty_df": novelty_df,
        "summary_csv": summary_csv_path,
        "novelty_csv": novelty_csv_path,
        "plot_path": plot_path,
        "grid_path": grid_path
    }


def scaffold_analysis_cli():
    """Command-line interface for scaffold novelty analysis."""
    parser = argparse.ArgumentParser(
        description="Scaffold novelty and diversity analysis for generated molecules."
    )
    parser.add_argument("--leads", type=str, required=True,
                        help="Path to lead molecules file (CSV or SMI)")
    parser.add_argument("--targets", type=str, required=True,
                        help="Path to generated molecules CSV with 'target' column")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: same as targets directory)")
    parser.add_argument("--smiles_col", type=str, default="smiles",
                        help="Name of SMILES column (default: smiles)")
    parser.add_argument("--target_col", type=str, default="target",
                        help="Name of target boolean column (default: target)")
    parser.add_argument("--top_novel", type=int, default=20,
                        help="Number of most novel targets to report (default: 20)")
    parser.add_argument("--generic", action="store_true",
                        help="Use generic scaffolds (atom-type agnostic)")
    
    args = parser.parse_args()
    
    scaffold_novelty_analysis(
        lead_file=args.leads,
        target_csv=args.targets,
        save_dir=args.save_dir,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        top_novel=args.top_novel,
        generic_scaffolds=args.generic
    )


def main():
    """Main entry point - supports KDE plots, scaffold analysis, or both."""
    parser = argparse.ArgumentParser(description="KDE plots and scaffold novelty analysis for molecules.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to molecules_with_properties.csv")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file (yaml/json) for property names and bounds")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: same as CSV directory)")
    parser.add_argument("--mode", type=str, default="both", choices=["kde", "scaffold", "dist", "both", "all"],
                        help="Analysis mode: kde, scaffold, dist (property distributions), both (kde+scaffold), or all (default: both)")
    parser.add_argument("--top_novel", type=int, default=20,
                        help="Number of most novel targets to report (default: 20)")
    parser.add_argument("--generic_scaffolds", action="store_true",
                        help="Use generic scaffolds (atom-type agnostic)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    property_names = get_property_names_from_config(config)
    property_bounds = get_property_bounds_from_config(config)

    if not property_names:
        print("Error: No property names found in config.")
        sys.exit(1)

    print(f"Properties: {property_names}")
    print(f"Bounds: {property_bounds}")

    # Build bounds dict
    bounds = {}
    if property_bounds:
        for i, name in enumerate(property_names):
            if i < len(property_bounds):
                bounds[name] = property_bounds[i]

    # Load CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} molecules from {args.csv}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Targets: {df['target'].sum()}")

    # Load starting molecules (leads) from train_file in config
    start_df = None
    train_file = None
    if config and "deepfmpo" in config:
        train_file = config["deepfmpo"].get("train_file")
        if train_file:
            if not os.path.isabs(train_file):
                train_file = os.path.join(os.getcwd(), train_file)
            if os.path.exists(train_file):
                start_df = pd.read_csv(train_file)
                # Normalise column names to lowercase for matching
                col_map = {c: c.lower() for c in start_df.columns}
                start_df = start_df.rename(columns=col_map)
                # Keep only property columns that exist
                avail = [p for p in property_names if p.lower() in start_df.columns]
                if avail:
                    start_df = start_df.rename(columns={p.lower(): p for p in property_names})
                else:
                    # Properties not in train file — merge from molecules_with_properties.csv
                    print(f"  No property columns in train file, merging from {args.csv}...")
                    start_smiles = start_df["smiles"].tolist()
                    matched = df[df["smiles"].isin(start_smiles)][["smiles"] + property_names].drop_duplicates(subset=["smiles"])
                    if len(matched) > 0:
                        start_df = start_df.merge(matched, on="smiles", how="inner")
                        avail = [p for p in property_names if p in start_df.columns]
                        print(f"  Matched {len(start_df)} starting molecules with properties: {avail}")
                    else:
                        print(f"  No matching SMILES found in CSV, skipping starting molecules.")
                        start_df = None

                if start_df is not None:
                    print(f"Loaded {len(start_df)} starting molecules from {train_file}")
                    avail = [p for p in property_names if p in start_df.columns]
                    print(f"  Available properties: {avail}")
            else:
                print(f"Warning: train_file not found: {train_file}")

    # Resolve save dir
    save_dir = args.save_dir or os.path.dirname(args.csv)
    ensure_dir(save_dir)

    # Pick a random lead molecule and find its closest target (for KDE visualization)
    lead_smi, lead_props, target_smi, target_props = None, None, None, None
    if start_df is not None and "smiles" in start_df.columns and "smiles" in df.columns:
        # Sample a random lead
        lead_row = start_df.sample(n=1, random_state=42).iloc[0]
        lead_smi = lead_row["smiles"]
        lead_props = {p: lead_row.get(p) for p in property_names if p in start_df.columns}

        # Get target SMILES list — search all targets
        target_df = df[df["target"] == True]
        if len(target_df) > 0:
            target_smi, sim_score = find_closest_target(lead_smi, target_df["smiles"].tolist())
            if target_smi:
                trow = df[df["smiles"] == target_smi].iloc[0]
                target_props = {p: trow.get(p) for p in property_names if p in df.columns}
                print(f"\nSelected lead: {lead_smi}")
                print(f"Closest target: {target_smi} (Tanimoto={sim_score:.3f})")

    # -------------------------------------------------------------------------
    # Run Property Distribution plots
    # -------------------------------------------------------------------------
    if args.mode in ["dist", "all"]:
        plot_all_property_distributions(
            df=df,
            property_names=property_names,
            save_dir=save_dir,
            bounds_dict=bounds,
            lead_df=start_df,
            bins=50
        )
        # Also plot grid view
        plot_property_grid(
            df=df,
            property_names=property_names,
            save_dir=save_dir,
            bounds_dict=bounds,
            lead_df=start_df
        )
    else:
        print("\nSkipping property distribution plots (use --mode dist or --mode all).")

    # -------------------------------------------------------------------------
    # Run KDE plots
    # -------------------------------------------------------------------------
    if args.mode in ["kde", "both", "all"]:
        pairs = list(combinations(property_names, 2))
        print(f"\nGenerating {len(pairs)} pairwise KDE plots...")

        for prop_x, prop_y in pairs:
            if prop_x not in df.columns:
                print(f"  Warning: column '{prop_x}' not in CSV, skipping pair.")
                continue
            if prop_y not in df.columns:
                print(f"  Warning: column '{prop_y}' not in CSV, skipping pair.")
                continue

            bound_x = bounds.get(prop_x)
            bound_y = bounds.get(prop_y)

            out = plot_kde_pair(df, prop_x, prop_y, bounds, save_dir,
                                bound_x=bound_x, bound_y=bound_y, start_df=start_df,
                                lead_smi=lead_smi, lead_props=lead_props,
                                target_smi=target_smi, target_props=target_props)
            if out:
                print(f"  Saved: {out}")
    else:
        print("\nSkipping KDE plots (use --mode kde, --mode both, or --mode all).")

    # -------------------------------------------------------------------------
    # Run Scaffold Novelty Analysis
    # -------------------------------------------------------------------------
    if args.mode in ["scaffold", "both", "all"]:
        if start_df is not None and "smiles" in start_df.columns:
            # Get lead SMILES from start_df
            lead_smiles = start_df["smiles"].dropna().unique().tolist()
            
            # Get target SMILES
            if "target" in df.columns:
                target_molecules_df = df[df["target"] == True].copy()
            else:
                target_molecules_df = df.copy()
            target_smiles = target_molecules_df["smiles"].dropna().unique().tolist()
            
            if lead_smiles and target_smiles:
                # Run scaffold analysis directly (inline version)
                print("\n" + "=" * 70)
                print("SCAFFOLD NOVELTY & DIVERSITY ANALYSIS")
                print("=" * 70)
                
                scaffold_type = "generic" if args.generic_scaffolds else "Murcko"
                print(f"\n[1] Computing {scaffold_type} scaffolds...")
                
                lead_scaffold_data = compute_scaffold_set(lead_smiles, generic=args.generic_scaffolds)
                target_scaffold_data = compute_scaffold_set(target_smiles, generic=args.generic_scaffolds)
                
                lead_scaffolds = lead_scaffold_data["unique_scaffolds"]
                target_scaffolds = target_scaffold_data["unique_scaffolds"]
                
                print(f"    Lead scaffolds:   {len(lead_scaffolds)} unique (from {len(lead_smiles)} molecules)")
                print(f"    Target scaffolds: {len(target_scaffolds)} unique (from {len(target_smiles)} molecules)")
                
                # Scaffold novelty
                new_scaffolds = target_scaffolds - lead_scaffolds
                shared_scaffolds = target_scaffolds & lead_scaffolds
                scaffold_novelty_rate = len(new_scaffolds) / len(target_scaffolds) * 100 if target_scaffolds else 0.0
                
                print(f"\n[2] Scaffold novelty:")
                print(f"    New scaffolds:    {len(new_scaffolds)} ({scaffold_novelty_rate:.1f}%)")
                print(f"    Shared scaffolds: {len(shared_scaffolds)}")
                
                # Targets with new scaffolds
                n_targets_new_scaffold = sum(
                    1 for smi in target_smiles 
                    if target_scaffold_data["smiles_to_scaffold"].get(smi, "") in new_scaffolds
                )
                print(f"    Targets with new scaffolds: {n_targets_new_scaffold}/{len(target_smiles)} ({n_targets_new_scaffold/len(target_smiles)*100:.1f}%)")
                
                # Diversity
                print(f"\n[3] Computing diversity...")
                target_pairwise = compute_pairwise_tanimoto(target_smiles)
                target_internal_diversity = 1 - np.mean(target_pairwise) if len(target_pairwise) > 0 else 1.0
                lead_pairwise = compute_pairwise_tanimoto(lead_smiles)
                lead_internal_diversity = 1 - np.mean(lead_pairwise) if len(lead_pairwise) > 0 else 1.0
                
                print(f"    Target internal diversity: {target_internal_diversity:.3f}")
                print(f"    Lead internal diversity:   {lead_internal_diversity:.3f}")
                
                # Novel targets
                print(f"\n[4] Finding {args.top_novel} most novel targets...")
                lead_fps = []
                for smi in lead_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        lead_fps.append(_morgan_gen.GetFingerprint(mol))
                
                novelty_scores = []
                for smi in target_smiles:
                    min_sim = find_min_tanimoto_to_leads(smi, lead_fps)
                    novelty_scores.append({
                        "smiles": smi,
                        "min_tanimoto_to_lead": min_sim,
                        "novelty_score": 1 - min_sim,
                        "scaffold": target_scaffold_data["smiles_to_scaffold"].get(smi, ""),
                        "is_new_scaffold": target_scaffold_data["smiles_to_scaffold"].get(smi, "") in new_scaffolds
                    })
                
                novelty_df = pd.DataFrame(novelty_scores).sort_values("novelty_score", ascending=False)
                
                # Save CSVs
                novelty_csv_path = os.path.join(save_dir, "scaffold_novelty_analysis.csv")
                novelty_df.to_csv(novelty_csv_path, index=False)
                print(f"    Saved: {novelty_csv_path}")
                
                # Summary CSV
                summary_data = {
                    "metric": [
                        "n_lead_molecules", "n_target_molecules", "n_lead_scaffolds", "n_target_scaffolds",
                        "n_new_scaffolds", "scaffold_novelty_rate_pct", "n_targets_with_new_scaffold",
                        "target_internal_diversity", "lead_internal_diversity", "avg_min_tanimoto_to_lead"
                    ],
                    "value": [
                        len(lead_smiles), len(target_smiles), len(lead_scaffolds), len(target_scaffolds),
                        len(new_scaffolds), f"{scaffold_novelty_rate:.2f}", n_targets_new_scaffold,
                        f"{target_internal_diversity:.4f}", f"{lead_internal_diversity:.4f}",
                        f"{novelty_df['min_tanimoto_to_lead'].mean():.4f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_csv_path = os.path.join(save_dir, "scaffold_analysis_summary.csv")
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"    Saved: {summary_csv_path}")
                
                # Plots
                print(f"\n[5] Generating plots...")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Top scaffolds
                ax1 = axes[0]
                top_scaffolds = target_scaffold_data["scaffold_counts"].most_common(20)
                if top_scaffolds:
                    scaffolds_labels = [s[0][:30] + "..." if len(s[0]) > 30 else s[0] for s in top_scaffolds]
                    scaffold_counts = [s[1] for s in top_scaffolds]
                    colors = ["#2ecc71" if top_scaffolds[i][0] in new_scaffolds else "#3498db" 
                              for i in range(len(top_scaffolds))]
                    y_pos = np.arange(len(scaffolds_labels))
                    ax1.barh(y_pos, scaffold_counts, color=colors, edgecolor="black", linewidth=0.5)
                    ax1.set_yticks(y_pos)
                    ax1.set_yticklabels(scaffolds_labels, fontsize=8)
                    ax1.invert_yaxis()
                    ax1.set_xlabel("Frequency", fontsize=11)
                    ax1.set_title("Top 20 Target Scaffolds\n(green = novel, blue = shared)", fontsize=12)
                
                # Novelty distribution
                ax2 = axes[1]
                ax2.hist(novelty_df["min_tanimoto_to_lead"], bins=30, color="#9b59b6", edgecolor="black", alpha=0.7)
                ax2.axvline(novelty_df["min_tanimoto_to_lead"].mean(), color="red", linestyle="--", linewidth=2,
                            label=f"Mean: {novelty_df['min_tanimoto_to_lead'].mean():.3f}")
                ax2.axvline(novelty_df["min_tanimoto_to_lead"].median(), color="orange", linestyle="--", linewidth=2,
                            label=f"Median: {novelty_df['min_tanimoto_to_lead'].median():.3f}")
                ax2.set_xlabel("Min Tanimoto to Lead", fontsize=11)
                ax2.set_ylabel("Count", fontsize=11)
                ax2.set_title("Target Novelty Distribution\n(lower = more novel)", fontsize=12)
                ax2.legend(loc="upper right")
                
                plt.tight_layout()
                plot_path = os.path.join(save_dir, "scaffold_frequency_plot.png")
                plt.savefig(plot_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"    Saved: {plot_path}")
                
                # Top novel molecules grid
                top_novel_df = novelty_df.head(args.top_novel)
                top_mols, top_legends = [], []
                for _, row in top_novel_df.head(12).iterrows():
                    mol = Chem.MolFromSmiles(row["smiles"])
                    if mol:
                        top_mols.append(mol)
                        scaffold_tag = " [NEW]" if row["is_new_scaffold"] else ""
                        top_legends.append(f"Tan={row['min_tanimoto_to_lead']:.3f}{scaffold_tag}")
                
                if top_mols:
                    img = Draw.MolsToGridImage(top_mols, molsPerRow=4, subImgSize=(300, 250), legends=top_legends)
                    grid_path = os.path.join(save_dir, "top_novel_molecules.png")
                    img.save(grid_path)
                    print(f"    Saved: {grid_path}")
                
                print("\n" + "=" * 70)
                print(f"Scaffold novelty rate: {scaffold_novelty_rate:.1f}% | Internal diversity: {target_internal_diversity:.3f}")
                print("=" * 70)
            else:
                print("\nWarning: No lead or target molecules found for scaffold analysis.")
        else:
            print("\nWarning: No lead molecules found in config (train_file). Skipping scaffold analysis.")
    else:
        print("\nSkipping scaffold analysis (--mode kde).")

    print("\nDone.")


if __name__ == "__main__":
    main()
