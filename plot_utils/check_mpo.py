import pandas as pd
import numpy as np
from rdkit import Chem 
import json
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.analyse_properties import *
from find_common_mols import *
from marl_plot_utils import compute_target_quality_score, compute_target_quality_for_dataframe

# Fallback colours per property (key = config property name)
prop_color_dict = {
    "fxa_pIC50": "#E63946",           # Red - primary activity
    "logD_74": "#457B9D",              # Blue - lipophilicity
    "molweight": "#2A9D8F",            # Teal - molecular weight
    "caco2_permeability": "#E9C46A",   # Gold - permeability
    "hERG_pIC50": "#F4A261",           # Orange - cardiac safety
    "hlm_clearance": "#9B5DE5",        # Purple - metabolic stability
}
plot_bounds=[[5,10],[-0.5,4],[380,660],[0,4],[3,7],[0,100]]


def create_violin_plots(df, prop_names, prop_bounds, output_dir):
    """Create violin plots for each property with bound markers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual plots for each property
    for i, prop in enumerate(prop_names):
        if prop not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(6, 5))
        color = prop_color_dict.get(prop, "#666666")
        
        # Add shaded rectangle for acceptable range (draw first so violin is on top)
        lower, upper = prop_bounds[i]
        ax.axhspan(lower, upper, alpha=0.2, color='green', label=f'Target range: [{lower}, {upper}]')
        ax.axhline(y=lower, color='green', linestyle='-', linewidth=1.5)
        ax.axhline(y=upper, color='green', linestyle='-', linewidth=1.5)
        
        # Violin plot
        sns.violinplot(y=df[prop], color=color, ax=ax, inner="box")
        
        # Use plot_bounds for y-axis limits
        ax.set_ylim(plot_bounds[i][0], plot_bounds[i][1])
        
        ax.set_ylabel(prop, fontsize=12)
        ax.set_title(f'Distribution of {prop}', fontsize=14)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violin_{prop.replace("/", "_")}.png'), dpi=150)
        plt.close()
    
    # Combined plot with all properties (normalized)
    fig, axes = plt.subplots(1, len(prop_names), figsize=(4 * len(prop_names), 5))
    if len(prop_names) == 1:
        axes = [axes]
    
    for i, (prop, ax) in enumerate(zip(prop_names, axes)):
        if prop not in df.columns:
            continue
        color = prop_color_dict.get(prop, "#666666")
        
        # Add shaded rectangle for acceptable range
        lower, upper = prop_bounds[i]
        ax.axhspan(lower, upper, alpha=0.2, color='green')
        ax.axhline(y=lower, color='green', linestyle='-', linewidth=1.5)
        ax.axhline(y=upper, color='green', linestyle='-', linewidth=1.5)
        
        sns.violinplot(y=df[prop], color=color, ax=ax, inner="box")
        
        # Use plot_bounds for y-axis limits
        ax.set_ylim(plot_bounds[i][0], plot_bounds[i][1])
        
        ax.set_ylabel(prop, fontsize=10)
        ax.set_title(prop.split('-')[0], fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violin_all_properties.png'), dpi=150)
    plt.close()
    
    print(f"Violin plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate molecular properties using RDKit and ML models.")
    parser.add_argument("--smiles_file", type=str, required=True, help="Path to the file containing SMILES strings.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots and results.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, 'r') as f:
        cfg = json.load(f)

    #load csv file with SMILES strings    smiles_df = pd.read_csv(args.smiles_file)
    input_df = pd.read_csv(args.smiles_file)
    if 'SMILES' in input_df.columns:
        smiles_list = input_df['SMILES'].tolist()
    else:
        smiles_list = input_df['smiles'].tolist()

    # Analyze SMILES and get scores
    results_df = smiles_analysis(smiles_list, cfg)
    prop_dict = cfg["properties"]
    prop_names=prop_dict["names"]
    prop_bounds=prop_dict["bounds"]
    out_name = args.smiles_file.split("/")[-1].split(".")[0]
    
    # Use output_dir if provided, else use input file directory
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "/".join(args.smiles_file.split("/")[:-1])
    
    results_df.to_csv(f"{out_dir}/{out_name}_analysis_results.csv", index=False)

    # Create violin plots for all molecules
   # create_violin_plots(results_df, prop_names, prop_bounds, out_dir)

    # Compute binary MPO: fraction of bounds satisfied per molecule
    k = len(prop_bounds)
    bounds_satisfied = np.zeros(len(results_df))
    for i, (prop, bound) in enumerate(zip(prop_names, prop_bounds)):
        if prop in results_df.columns:
            bounds_satisfied += ((results_df[prop] >= bound[0]) & (results_df[prop] <= bound[1])).astype(int)
    results_df['binary_mpo'] = bounds_satisfied / k

    # Diversity for all molecules
    all_smiles = results_df['smiles'].to_list()
    all_canonical = [canonicalize_smiles(x) for x in all_smiles]
    print(f"\n--- All molecules (n={len(results_df)}) ---")
    print(f"Average binary MPO (fraction of {k} bounds satisfied): {results_df['binary_mpo'].mean():.4f}")
    div_all = analyze_diversity(all_canonical, 'deepfmpo_all', out_dir)

    # Targets: molecules with all scores in bounds
    targ_df = results_df[results_df['all_scores_in_bounds'] == True]
    targ_df["Quality_Score"]=compute_target_quality_for_dataframe(targ_df,prop_names,prop_bounds)
    targ_smiles = targ_df['smiles'].to_list()
    targ_canonical = [canonicalize_smiles(x) for x in targ_smiles]
    targ_df['smiles'].to_csv(f"{out_dir}/{out_name}_targets.csv", index=False)

    print(f"\n--- Target molecules (n={len(targ_df)}, all {k} bounds satisfied) ---")
    print(f"Average target quality: {targ_df['Quality_Score'].mean():.4f}")
    div_targ = analyze_diversity(targ_canonical, 'deepfmpo_targets', out_dir)