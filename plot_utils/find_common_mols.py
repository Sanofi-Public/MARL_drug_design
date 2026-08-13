#!/usr/bin/env python3
"""
Compare two CSV files containing molecule SMILES and find common/unique molecules.
Also analyzes chemical diversity of each dataset.
Can compare against an approved/reference molecule set.

Usage:
    python find_common_mols.py --csv1 file1.csv --csv2 file2.csv --smiles_col SMILES --output_dir ./comparison
    python find_common_mols.py --csv1 file1.csv --csv2 file2.csv --approved approved.csv --top_k 10
"""

import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
import matplotlib.pyplot as plt

# Import smiles_analysis from utils
try:
    from utils.analyse_properties import smiles_analysis
    SMILES_ANALYSIS_AVAILABLE = True
except ImportError:
    SMILES_ANALYSIS_AVAILABLE = False
    print("Warning: utils.analyse_properties not available. Property filtering will be skipped.")


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form for consistent comparison."""
    if pd.isna(smiles) or smiles == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None


def load_molecules(csv_path, smiles_col="SMILES"):
    """Load CSV and extract canonicalized SMILES."""
    df = pd.read_csv(csv_path)
    
    # Try to find the SMILES column (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if smiles_col not in df.columns:
        if smiles_col.lower() in cols_lower:
            smiles_col = cols_lower[smiles_col.lower()]
        else:
            raise ValueError(f"Column '{smiles_col}' not found in {csv_path}. Available: {list(df.columns)}")
    
    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()]
    
    print(f"Loaded {len(df)} valid molecules from {os.path.basename(csv_path)}")
    return df


def compute_fingerprints(smiles_list, radius=2, n_bits=2048, names_list=None):
    """Compute Morgan fingerprints for a list of SMILES using MorganGenerator."""
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    valid_smiles = []
    valid_names = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            fps.append(fp)
            valid_smiles.append(smi)
            if names_list is not None:
                valid_names.append(names_list[i] if i < len(names_list) else "")
    if names_list is not None:
        return fps, valid_smiles, valid_names
    return fps, valid_smiles


def compute_tanimoto_matrix(fps, sample_size=1000):
    """Compute pairwise Tanimoto similarity matrix (sampled for large datasets)."""
    n = len(fps)
    if n == 0:
        return np.array([]), 0.0
    
    # Sample if too large
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        fps_sample = [fps[i] for i in indices]
    else:
        fps_sample = fps
    
    n_sample = len(fps_sample)
    similarities = []
    
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            sim = DataStructs.TanimotoSimilarity(fps_sample[i], fps_sample[j])
            similarities.append(sim)
    
    if similarities:
        avg_similarity = np.mean(similarities)
        return np.array(similarities), avg_similarity
    return np.array([]), 0.0


def get_scaffold(smiles):
    """Extract Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        pass
    return None


def compute_molecular_properties(smiles_list):
    """Compute molecular properties for a list of SMILES."""
    properties = {
        "MW": [], "LogP": [], "TPSA": [], "HBD": [], "HBA": [],
        "RotBonds": [], "Rings": [], "HeavyAtoms": []
    }
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            properties["MW"].append(Descriptors.MolWt(mol))
            properties["LogP"].append(Descriptors.MolLogP(mol))
            properties["TPSA"].append(Descriptors.TPSA(mol))
            properties["HBD"].append(rdMolDescriptors.CalcNumHBD(mol))
            properties["HBA"].append(rdMolDescriptors.CalcNumHBA(mol))
            properties["RotBonds"].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            properties["Rings"].append(rdMolDescriptors.CalcNumRings(mol))
            properties["HeavyAtoms"].append(mol.GetNumHeavyAtoms())
    
    return properties


def analyze_diversity(smiles_list, name="dataset", output_dir=None):
    """Analyze chemical diversity of a molecule set."""
    print(f"\n{'='*50}")
    print(f"Diversity Analysis: {name}")
    print(f"{'='*50}")
    
    n_mols = len(smiles_list)
    print(f"Number of molecules: {n_mols}")
    
    if n_mols == 0:
        return {}
    
    results = {"name": name, "n_molecules": n_mols}
    
    # 1. Fingerprint-based diversity (Tanimoto)
    print("Computing fingerprints...")
    fps, valid_smiles = compute_fingerprints(smiles_list)
    print(f"Valid fingerprints: {len(fps)}")
    
    if len(fps) > 1:
        print("Computing pairwise Tanimoto similarities (sampling if large)...")
        similarities, avg_sim = compute_tanimoto_matrix(fps)
        internal_diversity = 1.0 - avg_sim
        
        print(f"Average pairwise Tanimoto similarity: {avg_sim:.4f}")
        print(f"Internal diversity (1 - avg similarity): {internal_diversity:.4f}")
        
        results["avg_tanimoto_similarity"] = avg_sim
        results["internal_diversity"] = internal_diversity
        results["min_similarity"] = np.min(similarities) if len(similarities) > 0 else 0
        results["max_similarity"] = np.max(similarities) if len(similarities) > 0 else 0
        results["std_similarity"] = np.std(similarities) if len(similarities) > 0 else 0
    
    # 2. Scaffold diversity
    print("Computing scaffolds...")
    scaffolds = [get_scaffold(smi) for smi in smiles_list]
    scaffolds = [s for s in scaffolds if s is not None]
    unique_scaffolds = set(scaffolds)
    scaffold_diversity = len(unique_scaffolds) / n_mols if n_mols > 0 else 0
    
    print(f"Unique scaffolds: {len(unique_scaffolds)}")
    print(f"Scaffold diversity (unique/total): {scaffold_diversity:.4f}")
    
    results["unique_scaffolds"] = len(unique_scaffolds)
    results["scaffold_diversity"] = scaffold_diversity
    
    # 3. Molecular properties
    print("Computing molecular properties...")
    properties = compute_molecular_properties(smiles_list)
    
    prop_stats = {}
    for prop_name, values in properties.items():
        if values:
            prop_stats[prop_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            print(f"{prop_name}: mean={prop_stats[prop_name]['mean']:.2f}, "
                  f"std={prop_stats[prop_name]['std']:.2f}, "
                  f"range=[{prop_stats[prop_name]['min']:.2f}, {prop_stats[prop_name]['max']:.2f}]")
    
    results["properties"] = prop_stats
    
    # Save plots if output_dir provided
    if output_dir:
        plot_diversity_analysis(smiles_list, similarities if len(fps) > 1 else np.array([]), 
                                properties, name, output_dir)
    
    return results


def plot_diversity_analysis(smiles_list, similarities, properties, name, output_dir):
    """Generate diversity analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Diversity Analysis: {name}", fontsize=14)
    
    # 1. Tanimoto similarity distribution
    ax = axes[0, 0]
    if len(similarities) > 0:
        ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
        ax.legend()
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Pairwise Similarity Distribution")
    
    # 2. MW distribution
    ax = axes[0, 1]
    if properties["MW"]:
        ax.hist(properties["MW"], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("Frequency")
    ax.set_title("MW Distribution")
    
    # 3. LogP distribution
    ax = axes[0, 2]
    if properties["LogP"]:
        ax.hist(properties["LogP"], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel("LogP")
    ax.set_ylabel("Frequency")
    ax.set_title("LogP Distribution")
    
    # 4. TPSA distribution
    ax = axes[1, 0]
    if properties["TPSA"]:
        ax.hist(properties["TPSA"], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel("TPSA")
    ax.set_ylabel("Frequency")
    ax.set_title("TPSA Distribution")
    
    # 5. MW vs LogP scatter
    ax = axes[1, 1]
    if properties["MW"] and properties["LogP"]:
        ax.scatter(properties["MW"], properties["LogP"], alpha=0.5, s=10)
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("LogP")
    ax.set_title("MW vs LogP")
    
    # 6. Property ranges summary
    ax = axes[1, 2]
    prop_names = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]
    means = [np.mean(properties[p]) if properties[p] else 0 for p in prop_names]
    stds = [np.std(properties[p]) if properties[p] else 0 for p in prop_names]
    
    # Normalize for visualization
    normalized_means = []
    for i, p in enumerate(prop_names):
        if p == "MW":
            normalized_means.append(means[i] / 500)
        elif p == "TPSA":
            normalized_means.append(means[i] / 150)
        else:
            normalized_means.append(means[i] / 10 if means[i] > 0 else 0)
    
    x_pos = np.arange(len(prop_names))
    ax.bar(x_pos, normalized_means, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prop_names, rotation=45)
    ax.set_ylabel("Normalized Value")
    ax.set_title("Property Summary (Normalized)")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"diversity_{name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved diversity plot to: {plot_path}")


def compare_molecules(df1, df2, name1="file1", name2="file2"):
    """Compare two dataframes and find common/unique molecules."""
    set1 = set(df1["canonical_smiles"].unique())
    set2 = set(df2["canonical_smiles"].unique())
    
    common = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    
    print(f"\n{'='*50}")
    print(f"Comparison Results:")
    print(f"{'='*50}")
    print(f"Unique molecules in {name1}: {len(set1)}")
    print(f"Unique molecules in {name2}: {len(set2)}")
    print(f"Common molecules: {len(common)}")
    print(f"Only in {name1}: {len(only_in_1)}")
    print(f"Only in {name2}: {len(only_in_2)}")
    print(f"{'='*50}")
    
    # Calculate overlap metrics
    if len(set1) > 0:
        overlap_pct_1 = (len(common) / len(set1)) * 100
        print(f"Overlap as % of {name1}: {overlap_pct_1:.2f}%")
    if len(set2) > 0:
        overlap_pct_2 = (len(common) / len(set2)) * 100
        print(f"Overlap as % of {name2}: {overlap_pct_2:.2f}%")
    if len(set1 | set2) > 0:
        jaccard = len(common) / len(set1 | set2)
        print(f"Jaccard similarity: {jaccard:.4f}")
    
    return {
        "common": common,
        "only_in_1": only_in_1,
        "only_in_2": only_in_2,
    }


def save_results(results, df1, df2, output_dir, name1="file1", name2="file2"):
    """Save comparison results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Common molecules (with data from both files)
    if results["common"]:
        common_df1 = df1[df1["canonical_smiles"].isin(results["common"])].copy()
        common_df1["source"] = name1
        common_df2 = df2[df2["canonical_smiles"].isin(results["common"])].copy()
        common_df2["source"] = name2
        
        # Save just the common SMILES
        common_smiles = pd.DataFrame({"SMILES": list(results["common"])})
        common_path = os.path.join(output_dir, "common_molecules.csv")
        common_smiles.to_csv(common_path, index=False)
        print(f"Saved common molecules to: {common_path}")
    
    # Only in file 1
    if results["only_in_1"]:
        only1_df = df1[df1["canonical_smiles"].isin(results["only_in_1"])].copy()
        only1_path = os.path.join(output_dir, f"only_in_{name1}.csv")
        only1_df.to_csv(only1_path, index=False)
        print(f"Saved molecules only in {name1} to: {only1_path}")
    
    # Only in file 2
    if results["only_in_2"]:
        only2_df = df2[df2["canonical_smiles"].isin(results["only_in_2"])].copy()
        only2_path = os.path.join(output_dir, f"only_in_{name2}.csv")
        only2_df.to_csv(only2_path, index=False)
        print(f"Saved molecules only in {name2} to: {only2_path}")
    
    # Summary
    summary = {
        "metric": ["unique_in_" + name1, "unique_in_" + name2, "common", 
                   "only_in_" + name1, "only_in_" + name2],
        "count": [len(df1["canonical_smiles"].unique()), len(df2["canonical_smiles"].unique()),
                  len(results["common"]), len(results["only_in_1"]), len(results["only_in_2"])]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")


def save_diversity_summary(diversity_results, output_dir):
    """Save diversity analysis summary to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    for result in diversity_results:
        row = {
            "dataset": result.get("name", ""),
            "n_molecules": result.get("n_molecules", 0),
            "avg_tanimoto_similarity": result.get("avg_tanimoto_similarity", np.nan),
            "internal_diversity": result.get("internal_diversity", np.nan),
            "unique_scaffolds": result.get("unique_scaffolds", 0),
            "scaffold_diversity": result.get("scaffold_diversity", np.nan),
        }
        
        # Add property stats
        props = result.get("properties", {})
        for prop_name in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]:
            if prop_name in props:
                row[f"{prop_name}_mean"] = props[prop_name]["mean"]
                row[f"{prop_name}_std"] = props[prop_name]["std"]
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "diversity_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved diversity summary to: {summary_path}")


def find_closest_to_reference(query_smiles, reference_fps, reference_smiles, top_k=5, reference_names=None):
    """
    Find the closest molecules in a query set to a reference set.
    
    Args:
        query_smiles: List of query SMILES
        reference_fps: List of fingerprints for reference molecules
        reference_smiles: List of reference SMILES
        top_k: Number of closest matches to return per query molecule
        reference_names: Optional list of names for reference molecules
    
    Returns:
        List of dicts with query SMILES and their closest reference matches
    """
    if not query_smiles or not reference_fps:
        return []
    
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    results = []
    
    for query_smi in query_smiles:
        mol = Chem.MolFromSmiles(query_smi)
        if mol is None:
            continue
        
        query_fp = generator.GetFingerprint(mol)
        
        # Compute similarity to all reference molecules
        similarities = []
        for i, ref_fp in enumerate(reference_fps):
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            ref_name = reference_names[i] if reference_names and i < len(reference_names) else ""
            similarities.append((sim, reference_smiles[i], ref_name))
        
        # Sort by similarity (descending) and get top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_matches = similarities[:top_k]
        
        results.append({
            "query_smiles": query_smi,
            "top_matches": top_matches,  # Now tuples of (sim, smiles, name)
            "max_similarity": top_matches[0][0] if top_matches else 0.0,
            "avg_top_k_similarity": np.mean([s[0] for s in top_matches]) if top_matches else 0.0
        })
    
    return results


def compute_reference_similarity_stats(query_smiles, reference_fps, reference_smiles, sample_size=1000):
    """
    Compute statistics of similarity between query set and reference set.
    
    Args:
        query_smiles: List of query SMILES
        reference_fps: List of fingerprints for reference molecules
        reference_smiles: List of reference SMILES
        sample_size: Max number of query molecules to sample for efficiency
    
    Returns:
        Dictionary with similarity statistics
    """
    if not query_smiles or not reference_fps:
        return {}
    
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    
    # Sample if too large
    if len(query_smiles) > sample_size:
        query_sample = np.random.choice(query_smiles, sample_size, replace=False).tolist()
    else:
        query_sample = query_smiles
    
    max_similarities = []
    avg_similarities = []
    
    for query_smi in query_sample:
        mol = Chem.MolFromSmiles(query_smi)
        if mol is None:
            continue
        
        query_fp = generator.GetFingerprint(mol)
        
        sims = [DataStructs.TanimotoSimilarity(query_fp, ref_fp) for ref_fp in reference_fps]
        if sims:
            max_similarities.append(max(sims))
            avg_similarities.append(np.mean(sims))
    
    if not max_similarities:
        return {}
    
    return {
        "max_sim_mean": np.mean(max_similarities),
        "max_sim_std": np.std(max_similarities),
        "max_sim_min": np.min(max_similarities),
        "max_sim_max": np.max(max_similarities),
        "avg_sim_mean": np.mean(avg_similarities),
        "avg_sim_std": np.std(avg_similarities),
        "n_high_similarity_0.7": sum(1 for s in max_similarities if s >= 0.7),
        "n_high_similarity_0.8": sum(1 for s in max_similarities if s >= 0.8),
        "n_high_similarity_0.9": sum(1 for s in max_similarities if s >= 0.9),
        "pct_high_similarity_0.7": sum(1 for s in max_similarities if s >= 0.7) / len(max_similarities) * 100,
    }


def analyze_approved_set(approved_smiles, name="approved"):
    """Analyze properties of the approved/reference molecule set."""
    print(f"\n{'='*50}")
    print(f"Approved Set Analysis: {name}")
    print(f"{'='*50}")
    
    n_mols = len(approved_smiles)
    print(f"Number of molecules: {n_mols}")
    
    if n_mols == 0:
        return {}
    
    properties = compute_molecular_properties(approved_smiles)
    
    results = {"name": name, "n_molecules": n_mols}
    
    for prop_name in ["MW", "LogP", "TPSA"]:
        values = properties.get(prop_name, [])
        if values:
            results[f"{prop_name}_mean"] = np.mean(values)
            results[f"{prop_name}_std"] = np.std(values)
            results[f"{prop_name}_min"] = np.min(values)
            results[f"{prop_name}_max"] = np.max(values)
            results[f"{prop_name}_median"] = np.median(values)
            print(f"{prop_name}: mean={results[f'{prop_name}_mean']:.2f}, "
                  f"std={results[f'{prop_name}_std']:.2f}, "
                  f"median={results[f'{prop_name}_median']:.2f}, "
                  f"range=[{results[f'{prop_name}_min']:.2f}, {results[f'{prop_name}_max']:.2f}]")
    
    return results


def save_closest_matches(closest_results, output_path, dataset_name, top_k=5):
    """Save closest matches to CSV."""
    rows = []
    for result in closest_results:
        row = {
            "query_smiles": result["query_smiles"],
            "max_similarity": result["max_similarity"],
            "avg_top_k_similarity": result["avg_top_k_similarity"],
        }
        for i, match in enumerate(result["top_matches"][:top_k]):
            # Handle both old format (sim, smiles) and new format (sim, smiles, name)
            if len(match) >= 3:
                sim, ref_smi, ref_name = match[0], match[1], match[2]
            else:
                sim, ref_smi = match[0], match[1]
                ref_name = ""
            row[f"match_{i+1}_smiles"] = ref_smi
            row[f"match_{i+1}_similarity"] = sim
            row[f"match_{i+1}_name"] = ref_name
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("max_similarity", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Saved closest matches for {dataset_name} to: {output_path}")
    return df


def save_top_similar_molecules(closest_results, output_dir, dataset_name, n_top=10):
    """
    Save the top N most similar molecules as images paired with their closest approved match.
    
    Args:
        closest_results: List of dicts with query SMILES and their closest reference matches
        output_dir: Directory to save images
        dataset_name: Name of the dataset for labeling
        n_top: Number of top molecules to save
    """
    if not closest_results:
        print(f"No results to save for {dataset_name}")
        return
    
    # Sort by max similarity and get top N
    sorted_results = sorted(closest_results, key=lambda x: x["max_similarity"], reverse=True)
    top_results = sorted_results[:n_top]
    
    # Create output directory for images
    img_dir = os.path.join(output_dir, f"top_similar_{dataset_name}")
    os.makedirs(img_dir, exist_ok=True)
    
    # Save individual pair images
    pair_data = []
    for i, result in enumerate(top_results):
        query_smi = result["query_smiles"]
        if not result["top_matches"]:
            continue
        
        # Handle both old format (sim, smiles) and new format (sim, smiles, name)
        match = result["top_matches"][0]
        if len(match) >= 3:
            best_sim, best_ref_smi, best_ref_name = match[0], match[1], match[2]
        else:
            best_sim, best_ref_smi = match[0], match[1]
            best_ref_name = ""
        
        query_mol = Chem.MolFromSmiles(query_smi)
        ref_mol = Chem.MolFromSmiles(best_ref_smi)
        
        if query_mol is None or ref_mol is None:
            continue
        
        # Create label for approved molecule
        approved_label = f"{best_ref_name}\n(Approved)" if best_ref_name else "Approved Match"
        
        # Create side-by-side image
        img = Draw.MolsToGridImage(
            [query_mol, ref_mol],
            molsPerRow=2,
            subImgSize=(400, 400),
            legends=[f"Query ({dataset_name})\nSim: {best_sim:.3f}", approved_label]
        )
        
        img_path = os.path.join(img_dir, f"pair_{i+1}_sim{best_sim:.3f}.png")
        img.save(img_path)
        
        pair_data.append({
            "rank": i + 1,
            "query_smiles": query_smi,
            "approved_smiles": best_ref_smi,
            "approved_name": best_ref_name,
            "similarity": best_sim
        })
    
    # Save summary CSV
    if pair_data:
        summary_df = pd.DataFrame(pair_data)
        summary_path = os.path.join(img_dir, "top_similar_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved {len(pair_data)} top similar molecule pairs for {dataset_name} to: {img_dir}")
    
    # Create a combined grid image of all top pairs
    if len(top_results) > 0:
        create_combined_grid(top_results, img_dir, dataset_name, n_top)


def create_combined_grid(top_results, img_dir, dataset_name, n_top=10):
    """Create a single grid image showing all top similar pairs."""
    mols = []
    legends = []
    
    for i, result in enumerate(top_results[:n_top]):
        query_smi = result["query_smiles"]
        if not result["top_matches"]:
            continue
        
        # Handle both old format (sim, smiles) and new format (sim, smiles, name)
        match = result["top_matches"][0]
        if len(match) >= 3:
            best_sim, best_ref_smi, best_ref_name = match[0], match[1], match[2]
        else:
            best_sim, best_ref_smi = match[0], match[1]
            best_ref_name = ""
        
        query_mol = Chem.MolFromSmiles(query_smi)
        ref_mol = Chem.MolFromSmiles(best_ref_smi)
        
        if query_mol is not None and ref_mol is not None:
            mols.append(query_mol)
            legends.append(f"#{i+1} Query\nSim: {best_sim:.3f}")
            mols.append(ref_mol)
            # Use name if available
            if best_ref_name:
                legends.append(f"#{i+1} {best_ref_name}")
            else:
                legends.append(f"#{i+1} Approved")
    
    if mols:
        # Create grid with pairs side by side
        n_pairs = len(mols) // 2
        cols = 4  # 2 pairs per row (query + approved for each)
        
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=cols,
            subImgSize=(300, 300),
            legends=legends
        )
        
        grid_path = os.path.join(img_dir, f"combined_grid_{dataset_name}.png")
        img.save(grid_path)
        print(f"Saved combined grid to: {grid_path}")


def find_input_molecules_for_approved(output_closest, input_smiles, approved_fps, approved_valid_smiles, approved_valid_names=None):
    """
    For each approved molecule that an output molecule is most similar to,
    find the input molecule that is most similar to that same approved molecule.
    
    Args:
        output_closest: List of closest match results for output molecules
        input_smiles: List of input SMILES
        approved_fps: Fingerprints for approved molecules
        approved_valid_smiles: Valid approved SMILES
        approved_valid_names: Optional names for approved molecules
    
    Returns:
        List of dicts containing output mol, approved mol, and matching input mol
    """
    if not output_closest or not input_smiles:
        return []
    
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    
    # Compute fingerprints for input molecules
    input_fps = []
    valid_input_smiles = []
    for smi in input_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            input_fps.append(fp)
            valid_input_smiles.append(smi)
    
    if not input_fps:
        return []
    
    # Create mapping from approved SMILES to index
    approved_smi_to_idx = {smi: i for i, smi in enumerate(approved_valid_smiles)}
    
    results = []
    seen_approved = set()  # Track which approved molecules we've already processed
    
    for output_result in output_closest:
        if not output_result["top_matches"]:
            continue
        
        # Get the approved molecule this output is most similar to
        match = output_result["top_matches"][0]
        if len(match) >= 3:
            output_sim, approved_smi, approved_name = match[0], match[1], match[2]
        else:
            output_sim, approved_smi = match[0], match[1]
            approved_name = ""
        
        # Skip if we've already found an input for this approved molecule
        if approved_smi in seen_approved:
            continue
        seen_approved.add(approved_smi)
        
        # Get the approved molecule's fingerprint
        approved_idx = approved_smi_to_idx.get(approved_smi)
        if approved_idx is None:
            continue
        approved_fp = approved_fps[approved_idx]
        
        # Find input molecule most similar to this approved molecule
        best_input_sim = -1
        best_input_smi = None
        
        for i, input_fp in enumerate(input_fps):
            sim = DataStructs.TanimotoSimilarity(input_fp, approved_fp)
            if sim > best_input_sim:
                best_input_sim = sim
                best_input_smi = valid_input_smiles[i]
        
        if best_input_smi:
            results.append({
                "output_smiles": output_result["query_smiles"],
                "output_to_approved_sim": output_sim,
                "approved_smiles": approved_smi,
                "approved_name": approved_name,
                "input_smiles": best_input_smi,
                "input_to_approved_sim": best_input_sim,
            })
    
    return results


def save_input_output_approved_comparison(comparison_results, output_dir, dataset_name, n_top=10):
    """
    Save comparison of input -> approved <- output molecules as images and CSV.
    """
    if not comparison_results:
        print(f"No input-output-approved comparison results for {dataset_name}")
        return
    
    # Sort by output similarity and get top N
    sorted_results = sorted(comparison_results, key=lambda x: x["output_to_approved_sim"], reverse=True)
    top_results = sorted_results[:n_top]
    
    # Create output directory
    img_dir = os.path.join(output_dir, f"input_output_approved_{dataset_name}")
    os.makedirs(img_dir, exist_ok=True)
    
    # Save individual triplet images
    for i, result in enumerate(top_results):
        input_mol = Chem.MolFromSmiles(result["input_smiles"])
        approved_mol = Chem.MolFromSmiles(result["approved_smiles"])
        output_mol = Chem.MolFromSmiles(result["output_smiles"])
        
        if input_mol is None or approved_mol is None or output_mol is None:
            continue
        
        approved_label = result["approved_name"] if result["approved_name"] else "Approved"
        
        img = Draw.MolsToGridImage(
            [input_mol, approved_mol, output_mol],
            molsPerRow=3,
            subImgSize=(350, 350),
            legends=[
                f"Input\nSim to approved: {result['input_to_approved_sim']:.3f}",
                f"{approved_label}\n(Reference)",
                f"Output ({dataset_name})\nSim to approved: {result['output_to_approved_sim']:.3f}"
            ]
        )
        
        img_path = os.path.join(img_dir, f"triplet_{i+1}.png")
        img.save(img_path)
    
    # Save summary CSV
    summary_df = pd.DataFrame(top_results)
    summary_path = os.path.join(img_dir, "input_output_approved_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {len(top_results)} input-output-approved triplets for {dataset_name} to: {img_dir}")
    
    # Create combined grid
    create_triplet_grid(top_results, img_dir, dataset_name, n_top)


def create_triplet_grid(results, img_dir, dataset_name, n_top=10):
    """Create a single grid image showing input-approved-output triplets."""
    mols = []
    legends = []
    
    for i, result in enumerate(results[:n_top]):
        input_mol = Chem.MolFromSmiles(result["input_smiles"])
        approved_mol = Chem.MolFromSmiles(result["approved_smiles"])
        output_mol = Chem.MolFromSmiles(result["output_smiles"])
        
        if input_mol is None or approved_mol is None or output_mol is None:
            continue
        
        approved_name = result["approved_name"] if result["approved_name"] else "Approved"
        
        mols.extend([input_mol, approved_mol, output_mol])
        legends.extend([
            f"#{i+1} Input\n{result['input_to_approved_sim']:.3f}",
            f"#{i+1} {approved_name}",
            f"#{i+1} Output\n{result['output_to_approved_sim']:.3f}"
        ])
    
    if mols:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=6,  # 2 triplets per row
            subImgSize=(250, 250),
            legends=legends
        )
        
        grid_path = os.path.join(img_dir, f"triplet_grid_{dataset_name}.png")
        img.save(grid_path)
        print(f"Saved triplet grid to: {grid_path}")


def plot_similarity_to_approved(sim_stats_dict, output_dir):
    """Plot similarity statistics to approved set."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = list(sim_stats_dict.keys())
    if not datasets:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Max similarity distribution comparison
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    
    means = [sim_stats_dict[d].get("max_sim_mean", 0) for d in datasets]
    stds = [sim_stats_dict[d].get("max_sim_std", 0) for d in datasets]
    
    ax.bar(x, means, width, yerr=stds, label='Max Similarity (mean ± std)', capsize=5)
    ax.set_ylabel('Tanimoto Similarity')
    ax.set_title('Maximum Similarity to Approved Set')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Plot 2: High similarity counts
    ax = axes[1]
    thresholds = ["0.7", "0.8", "0.9"]
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, thresh in enumerate(thresholds):
        counts = [sim_stats_dict[d].get(f"n_high_similarity_{thresh}", 0) for d in datasets]
        ax.bar(x + i * width, counts, width, label=f'Sim ≥ {thresh}')
    
    ax.set_ylabel('Number of Molecules')
    ax.set_title('Molecules with High Similarity to Approved Set')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "similarity_to_approved.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved similarity plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare molecules in two CSV files.")
    parser.add_argument("--csv1", type=str, required=True, help="Path to first CSV file")
    parser.add_argument("--csv2", type=str, required=True, help="Path to second CSV file")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="Name of SMILES column")
    parser.add_argument("--smiles_col1", type=str, default=None, help="SMILES column name for csv1 (if different)")
    parser.add_argument("--smiles_col2", type=str, default=None, help="SMILES column name for csv2 (if different)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--name1", type=str, default=None, help="Label for first file")
    parser.add_argument("--name2", type=str, default=None, help="Label for second file")
    parser.add_argument("--analyze_diversity", action="store_true", default=True, 
                        help="Perform diversity analysis (default: True)")
    parser.add_argument("--no_diversity", action="store_true", help="Skip diversity analysis")
    parser.add_argument("--approved", type=str, default=None, 
                        help="Path to approved/reference molecules CSV file")
    parser.add_argument("--approved_smiles_col", type=str, default="SMILES", 
                        help="SMILES column name in approved file")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of closest matches to find per molecule")
    parser.add_argument("--n_top_images", type=int, default=10,
                        help="Number of top similar molecules to save as images")
    parser.add_argument("--input_mols", type=str, default=None,
                        help="Path to input molecules CSV file (to compare input->approved<-output)")
    parser.add_argument("--input_smiles_col", type=str, default="SMILES",
                        help="SMILES column name in input molecules file")
    args = parser.parse_args()

    # Determine column names
    col1 = args.smiles_col1 if args.smiles_col1 else args.smiles_col
    col2 = args.smiles_col2 if args.smiles_col2 else args.smiles_col

    # Determine labels
    name1 = args.name1 if args.name1 else os.path.splitext(os.path.basename(args.csv1))[0]
    name2 = args.name2 if args.name2 else os.path.splitext(os.path.basename(args.csv2))[0]
    
    # Determine output directory
    if args.output_dir is None:
        folder_name = f"{name1}_vs_{name2}"
        output_dir = os.path.join("./mol_comparison", folder_name)
    else:
        output_dir = args.output_dir
    
    # Load data
    print(f"Loading {args.csv1}...")
    df1 = load_molecules(args.csv1, col1)
    
    print(f"Loading {args.csv2}...")
    df2 = load_molecules(args.csv2, col2)
    
    # Compare
    results = compare_molecules(df1, df2, name1, name2)
    
    # Save results
    save_results(results, df1, df2, output_dir, name1, name2)
    
    # Diversity analysis
    if args.analyze_diversity and not args.no_diversity:
        diversity_results = []
        
        smiles1 = df1["canonical_smiles"].unique().tolist()
        div1 = analyze_diversity(smiles1, name1, output_dir)
        diversity_results.append(div1)
        
        smiles2 = df2["canonical_smiles"].unique().tolist()
        div2 = analyze_diversity(smiles2, name2, output_dir)
        diversity_results.append(div2)
        
        if results["common"]:
            common_smiles = list(results["common"])
            div_common = analyze_diversity(common_smiles, "common", output_dir)
            diversity_results.append(div_common)
        
        save_diversity_summary(diversity_results, output_dir)
    
    # Approved set comparison
    if args.approved:
        print(f"\n{'='*60}")
        print("APPROVED SET COMPARISON")
        print(f"{'='*60}")
        
        print(f"Loading approved molecules from {args.approved}...")
        approved_df = load_molecules(args.approved, args.approved_smiles_col)
        approved_smiles = approved_df["canonical_smiles"].unique().tolist()
        
        # Get names if available
        approved_names = None
        if "Name" in approved_df.columns:
            smiles_to_name = dict(zip(approved_df["canonical_smiles"], approved_df["Name"]))
            approved_names = [smiles_to_name.get(smi, "") for smi in approved_smiles]
            print(f"Found {sum(1 for n in approved_names if n)} molecule names in approved set")
        elif "name" in approved_df.columns:
            smiles_to_name = dict(zip(approved_df["canonical_smiles"], approved_df["name"]))
            approved_names = [smiles_to_name.get(smi, "") for smi in approved_smiles]
            print(f"Found {sum(1 for n in approved_names if n)} molecule names in approved set")
        
        # Analyze approved set properties
        approved_stats = analyze_approved_set(approved_smiles, "approved")
        
        # Save approved stats
        approved_stats_df = pd.DataFrame([approved_stats])
        approved_stats_path = os.path.join(output_dir, "approved_set_stats.csv")
        approved_stats_df.to_csv(approved_stats_path, index=False)
        print(f"Saved approved set stats to: {approved_stats_path}")
        
        # Compute fingerprints for approved set (with names)
        print("Computing fingerprints for approved set...")
        if approved_names:
            approved_fps, approved_valid_smiles, approved_valid_names = compute_fingerprints(
                approved_smiles, names_list=approved_names
            )
        else:
            approved_fps, approved_valid_smiles = compute_fingerprints(approved_smiles)
            approved_valid_names = None
        print(f"Valid approved fingerprints: {len(approved_fps)}")
        
        # Compute similarity stats for each dataset
        sim_stats_dict = {}
        
        smiles1 = df1["canonical_smiles"].unique().tolist()
        print(f"\nComputing similarity to approved set for {name1}...")
        sim_stats1 = compute_reference_similarity_stats(smiles1, approved_fps, approved_valid_smiles)
        sim_stats_dict[name1] = sim_stats1
        print(f"  Max similarity mean: {sim_stats1.get('max_sim_mean', 0):.4f}")
        print(f"  Molecules with similarity ≥ 0.7: {sim_stats1.get('n_high_similarity_0.7', 0)}")
        print(f"  Molecules with similarity ≥ 0.8: {sim_stats1.get('n_high_similarity_0.8', 0)}")
        
        smiles2 = df2["canonical_smiles"].unique().tolist()
        print(f"\nComputing similarity to approved set for {name2}...")
        sim_stats2 = compute_reference_similarity_stats(smiles2, approved_fps, approved_valid_smiles)
        sim_stats_dict[name2] = sim_stats2
        print(f"  Max similarity mean: {sim_stats2.get('max_sim_mean', 0):.4f}")
        print(f"  Molecules with similarity ≥ 0.7: {sim_stats2.get('n_high_similarity_0.7', 0)}")
        print(f"  Molecules with similarity ≥ 0.8: {sim_stats2.get('n_high_similarity_0.8', 0)}")
        
        # Save similarity stats
        sim_stats_rows = []
        for dataset_name, stats in sim_stats_dict.items():
            row = {"dataset": dataset_name}
            row.update(stats)
            sim_stats_rows.append(row)
        sim_stats_df = pd.DataFrame(sim_stats_rows)
        sim_stats_path = os.path.join(output_dir, "similarity_to_approved_stats.csv")
        sim_stats_df.to_csv(sim_stats_path, index=False)
        print(f"\nSaved similarity stats to: {sim_stats_path}")
        
        # Find closest matches (with names)
        print(f"\nFinding top {args.top_k} closest matches to approved set...")
        
        closest1 = find_closest_to_reference(
            smiles1, approved_fps, approved_valid_smiles, args.top_k, 
            reference_names=approved_valid_names
        )
        closest1_path = os.path.join(output_dir, f"closest_to_approved_{name1}.csv")
        save_closest_matches(closest1, closest1_path, name1, args.top_k)
        
        closest2 = find_closest_to_reference(
            smiles2, approved_fps, approved_valid_smiles, args.top_k,
            reference_names=approved_valid_names
        )
        closest2_path = os.path.join(output_dir, f"closest_to_approved_{name2}.csv")
        save_closest_matches(closest2, closest2_path, name2, args.top_k)
        
        # Load input molecules if provided
        input_smiles = None
        if args.input_mols:
            print(f"\nLoading input molecules from {args.input_mols}...")
            input_df = load_molecules(args.input_mols, args.input_smiles_col)
            input_smiles = input_df["canonical_smiles"].unique().tolist()
            print(f"Loaded {len(input_smiles)} unique input molecules")

        # Save top similar molecules as images
        print(f"\nSaving top {args.n_top_images} most similar molecules as image pairs...")
        save_top_similar_molecules(closest1, output_dir, name1, args.n_top_images)
        save_top_similar_molecules(closest2, output_dir, name2, args.n_top_images)

        # Compare input -> approved <- output if input molecules provided
        if input_smiles:
            print(f"\nFinding input molecules most similar to approved molecules matched by outputs...")
            print(f"Using {len(input_smiles)} input molecules")

            comparison1 = find_input_molecules_for_approved(
                closest1, input_smiles, approved_fps, approved_valid_smiles, approved_valid_names
            )
            save_input_output_approved_comparison(comparison1, output_dir, name1, args.n_top_images)

            comparison2 = find_input_molecules_for_approved(
                closest2, input_smiles, approved_fps, approved_valid_smiles, approved_valid_names
            )
            save_input_output_approved_comparison(comparison2, output_dir, name2, args.n_top_images)
        
        # Plot similarity comparison
        plot_similarity_to_approved(sim_stats_dict, output_dir)
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
