"""
Compound Picker for Iterative Design Phases

Selects k diverse, high-quality compounds from a target molecules CSV.
Mimics a medicinal chemist's prioritization: rank by quality, cluster for
diversity, then pick the best from each cluster.

Prioritizes the primary property (configurable via --primary_prop) and filters out molecules
with structural alerts (PAINS, toxicity flags).

Usage:
    python plot_utils/pick_compounds.py \
        --csv ia2c_results/<run_name>/unique_target_molecules.csv \
        --props_csv ia2c_results/<run_name>/unique_target_molecules_with_props.csv \
        --k 20 \
        --sa_cutoff 5.0 \
        --output picked_compounds.csv

Random Diverse Mode (no filters, just diversity):
    python plot_utils/pick_compounds.py \
        --csv molecules.csv \
        --k 10 \
        --random

Clean Mode for REINVENT (neutralize, strip salts, filter):
    python plot_utils/pick_compounds.py \
        --csv molecules.csv \
        --k 1000 \
        --random \
        --clean \
        --filter-charges \
        --filter-rare \
        --smi \
        --output reinvent_ready.csv
"""

import argparse
import sys
import os
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib'))
from rdkit.Contrib.SA_Score import sascorer
from plots.filter import alerts

# Create Morgan fingerprint generator (replaces deprecated GetMorganFingerprintAsBitVect)
_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

# Common organic elements (filter out rare elements)
COMMON_ELEMENTS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'B', 'Si'}
DRUGLIKE_ATOMS = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'}  # Strict drug-like atom set

# Unusual bracket tokens to filter
UNUSUAL_BRACKETS = re.compile(r'\[([0-9]+[A-Z]|[A-Z][a-z]?[@\+\-]+|\d+\*)')


def clean_molecule(smiles, strip_salts=True, uncharge=True):
    """
    Clean a SMILES string for REINVENT compatibility.
    
    Steps:
    1. Parse SMILES
    2. Strip salts (largest fragment)
    3. Uncharge (neutralize)
    4. Canonicalize
    
    Args:
        smiles: Input SMILES string
        strip_salts: If True, keep only the largest fragment
        uncharge: If True, neutralize charges
        
    Returns:
        Cleaned canonical SMILES or None if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Strip salts - keep largest fragment
        if strip_salts:
            mol = rdMolStandardize.FragmentParent(mol)
        
        # Uncharge - neutralize where possible  
        if uncharge:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        
        # Canonicalize
        clean_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return clean_smi
        
    except Exception as e:
        return None


def has_formal_charges(smiles):
    """Check if molecule has any remaining formal charges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # Fail safe
    
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return True
    return False


def has_rare_elements(smiles):
    """Check if molecule contains rare/uncommon elements or isotopes."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    
    for atom in mol.GetAtoms():
        # Check for isotopes
        if atom.GetIsotope() != 0:
            return True
        # Check for rare elements
        symbol = atom.GetSymbol()
        if symbol not in COMMON_ELEMENTS:
            return True
    return False


def has_unusual_brackets(smiles):
    """Check for unusual bracket tokens that might cause issues."""
    # Check for isotopes like [13C], charged atoms like [N+], etc.
    # Also catches numbered attachment points [1*] etc.
    return bool(UNUSUAL_BRACKETS.search(smiles))


def has_disallowed_atoms(smiles):
    """Check if molecule contains atoms outside the drug-like set {C,N,O,S,F,Cl,Br,I}."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in DRUGLIKE_ATOMS:
            return True
    return False


def clean_and_filter_molecules(df, smiles_col='SMILES', strip_salts=True, uncharge=True,
                                filter_charges=True, filter_rare=True, filter_brackets=False,
                                filter_atoms=False, verbose=True):
    """
    Clean and filter a DataFrame of molecules.
    
    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column
        strip_salts: Strip counter-ions and salts
        uncharge: Neutralize charges where possible
        filter_charges: Remove molecules with remaining formal charges
        filter_rare: Remove molecules with rare elements/isotopes
        filter_brackets: Remove molecules with unusual bracket tokens
        filter_atoms: Remove molecules with atoms outside {C,N,O,S,F,Cl,Br,I}
        verbose: Print progress
        
    Returns:
        Cleaned DataFrame with new 'SMILES_clean' column
    """
    cleaned = []
    original = []
    
    for smi in df[smiles_col].tolist():
        clean_smi = clean_molecule(smi, strip_salts=strip_salts, uncharge=uncharge)
        if clean_smi is not None:
            cleaned.append(clean_smi)
            original.append(smi)
        else:
            cleaned.append(None)
            original.append(smi)
    
    df = df.copy()
    df['SMILES_clean'] = cleaned
    df['SMILES_original'] = original
    
    # Remove failed cleaning
    n_before = len(df)
    df = df[df['SMILES_clean'].notna()].copy()
    n_clean_fail = n_before - len(df)
    
    # Apply filters
    n_charged = 0
    n_rare = 0  
    n_brackets = 0
    n_atoms = 0
    
    if filter_charges:
        mask = ~df['SMILES_clean'].apply(has_formal_charges)
        n_charged = (~mask).sum()
        df = df[mask].copy()
    
    if filter_rare:
        mask = ~df['SMILES_clean'].apply(has_rare_elements)
        n_rare = (~mask).sum()
        df = df[mask].copy()
    
    if filter_brackets:
        mask = ~df['SMILES_clean'].apply(has_unusual_brackets)
        n_brackets = (~mask).sum()
        df = df[mask].copy()
    
    if filter_atoms:
        mask = ~df['SMILES_clean'].apply(has_disallowed_atoms)
        n_atoms = (~mask).sum()
        df = df[mask].copy()
    
    # Replace SMILES column with cleaned version
    df[smiles_col] = df['SMILES_clean']
    df = df.drop(columns=['SMILES_clean'])
    
    if verbose:
        print(f"\n=== Molecule Cleaning Summary ===")
        print(f"  Cleaning failed:        {n_clean_fail}")
        print(f"  Remaining charges:      {n_charged}")
        print(f"  Rare elements/isotopes: {n_rare}")
        if filter_brackets:
            print(f"  Unusual brackets:       {n_brackets}")
        if filter_atoms:
            print(f"  Disallowed atoms:       {n_atoms}")
        print(f"  Final molecules:        {len(df)}")
        print(f"================================\n")
    
    return df


def compute_fingerprints(smiles_list):
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = _morgan_gen.GetFingerprint(mol)
            fps.append(fp)
            valid_idx.append(i)
    return fps, valid_idx


def compute_sa_scores(smiles_list):
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scores.append(sascorer.calculateScore(mol))
        else:
            scores.append(10.0)  # worst possible
    return scores


def tanimoto_distance(fps_i, fps_j):
    return 1.0 - DataStructs.TanimotoSimilarity(fps_i, fps_j)


def filter_alerts(smiles_list, alert_path="./data/alert_collection.csv"):
    """Filter molecules that raise structural alerts (PAINS, toxicity flags).
    
    Returns:
        valid_mask: Boolean list, True if molecule passes (no alerts)
    """
    valid_mask = []
    alert_args = {"alert_collection_path": alert_path}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_alert = alerts(mol, alert_args)
            valid_mask.append(has_alert == 0)  # 0 means no alerts
        else:
            valid_mask.append(False)
    return valid_mask


def pick_diverse(df, fps, k):
    """Use MaxMinPicker to select k diverse compounds from fingerprints."""
    n = len(fps)
    if n <= k:
        return list(range(n))

    def dist_func(i, j):
        return tanimoto_distance(fps[i], fps[j])

    picker = MaxMinPicker()
    pick_indices = list(picker.LazyPick(dist_func, n, k, seed=42))
    return pick_indices


def main():
    parser = argparse.ArgumentParser(description="Pick k diverse, high-quality compounds from target molecules.")
    parser.add_argument("--csv", required=True, help="Path to unique_target_molecules.csv")
    parser.add_argument("--props_csv", default=None, help="Path to CSV with properties (e.g., fxa_pIC50)")
    parser.add_argument("--k", type=int, default=20, help="Number of compounds to pick")
    parser.add_argument("--sa_cutoff", type=float, default=5.0,
                        help="SA score cutoff (lower = more synthesizable, default 5.0)")
    parser.add_argument("--alert_path", default="./data/alert_collection.csv",
                        help="Path to alert collection TSV file")
    parser.add_argument("--primary_prop", default="fxa_pIC50",
                        help="Primary property to prioritize (default: fxa_pIC50)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: same dir as input)")
    parser.add_argument("--random", action="store_true",
                        help="Pick k diverse compounds randomly without any filters (diversity only)")
    
    # Cleaning options for REINVENT compatibility
    parser.add_argument("--clean", action="store_true",
                        help="Clean molecules: uncharge, strip salts, canonicalize")
    parser.add_argument("--no-strip-salts", action="store_true",
                        help="Don't strip salts/counter-ions (only with --clean)")
    parser.add_argument("--no-uncharge", action="store_true",
                        help="Don't neutralize charges (only with --clean)")
    parser.add_argument("--filter-charges", action="store_true",
                        help="Remove molecules with remaining formal charges (only with --clean)")
    parser.add_argument("--filter-rare", action="store_true",
                        help="Remove molecules with rare elements/isotopes (only with --clean)")
    parser.add_argument("--filter-brackets", action="store_true",
                        help="Remove molecules with unusual bracket tokens (only with --clean)")
    parser.add_argument("--filter-atoms", action="store_true",
                        help="Remove molecules with atoms outside {C,N,O,S,F,Cl,Br,I} (only with --clean)")
    parser.add_argument("--smi", action="store_true",
                        help="Also output a .smi file (SMILES only, one per line)")
    
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} target molecules")

    # Random diverse mode: pick diverse FIRST, then clean (more efficient)
    if args.random:
        print("\n=== Random Diverse Mode ===")
        
        # Step 1: Compute fingerprints on raw molecules (fast)
        print("Computing fingerprints...")
        fps, valid_idx = compute_fingerprints(df['SMILES'].tolist())
        df_valid = df.iloc[valid_idx].reset_index(drop=True)
        
        if len(df_valid) == 0:
            print("Error: no valid molecules for fingerprinting")
            sys.exit(1)
        
        # Step 2: Pick diverse candidates (2x k to have buffer for cleaning losses)
        pool_size = min(len(df_valid), args.k * 2 if args.clean else args.k)
        print(f"Selecting {pool_size} diverse candidates...")
        pick_idx = pick_diverse(df_valid, fps, pool_size)
        df_pool = df_valid.iloc[pick_idx].reset_index(drop=True)
        
        # Step 3: Clean only the diverse pool (if requested)
        if args.clean:
            print("\n=== Cleaning Selected Molecules ===")
            df_pool = clean_and_filter_molecules(
                df_pool, 
                smiles_col='SMILES',
                strip_salts=not args.no_strip_salts,
                uncharge=not args.no_uncharge,
                filter_charges=args.filter_charges,
                filter_rare=args.filter_rare,
                filter_brackets=args.filter_brackets,
                filter_atoms=args.filter_atoms,
                verbose=True
            )
            
            if len(df_pool) == 0:
                print("Error: No molecules left after cleaning!")
                sys.exit(1)
        
        # Step 4: Take final k molecules
        k = min(args.k, len(df_pool))
        df_picked = df_pool.head(k).reset_index(drop=True)
        
        # Add rank column
        df_picked.insert(0, 'Rank', range(1, len(df_picked) + 1))
        cols = ['SMILES', 'Rank'] + [c for c in df_picked.columns if c not in ('SMILES', 'Rank')]
        df_picked = df_picked[cols]
        
        # Summary stats
        print(f"\nPicked {len(df_picked)} diverse compounds")
        
        # Compute pairwise Tanimoto stats on final picked set
        picked_fps = []
        for smi in df_picked['SMILES'].tolist():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                picked_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        tani_vals = []
        for i in range(len(picked_fps)):
            for j in range(i + 1, len(picked_fps)):
                tani_vals.append(DataStructs.TanimotoSimilarity(picked_fps[i], picked_fps[j]))
        if tani_vals:
            print(f"  Avg pairwise Tanimoto: {np.mean(tani_vals):.3f} (lower = more diverse)")
        
        # Save
        if args.output is None:
            out_dir = os.path.dirname(args.csv)
            out_path = os.path.join(out_dir, f"random_diverse_{k}_compounds.csv")
        else:
            out_path = args.output
        df_picked.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        
        # Save .smi file if requested
        if args.smi:
            smi_path = out_path.replace('.csv', '.smi')
            with open(smi_path, 'w') as f:
                for smi in df_picked['SMILES'].tolist():
                    f.write(f"{smi}\n")
            print(f"Saved SMILES to {smi_path}")
        
        sys.exit(0)

    # Merge with properties CSV if provided
    if args.props_csv:
        props_df = pd.read_csv(args.props_csv)
        # Merge on SMILES column
        if 'SMILES' in props_df.columns and 'SMILES' in df.columns:
            df = df.merge(props_df[['SMILES', args.primary_prop]], on='SMILES', how='left')
            print(f"Merged properties from {args.props_csv}")
        elif 'smiles' in props_df.columns:
            props_df = props_df.rename(columns={'smiles': 'SMILES'})
            df = df.merge(props_df[['SMILES', args.primary_prop]], on='SMILES', how='left')
            print(f"Merged properties from {args.props_csv}")

    # Step 1: Filter molecules with structural alerts
    print("Filtering molecules with structural alerts...")
    alert_mask = filter_alerts(df['SMILES'].tolist(), args.alert_path)
    df_no_alerts = df[alert_mask].copy()
    print(f"After alert filter: {len(df_no_alerts)} molecules (removed {len(df) - len(df_no_alerts)} with alerts)")

    if len(df_no_alerts) == 0:
        print("Warning: All molecules have alerts. Proceeding with original set.")
        df_no_alerts = df.copy()

    # Step 2: Compute SA scores and filter
    df_no_alerts['SA_Score'] = compute_sa_scores(df_no_alerts['SMILES'].tolist())
    df_filtered = df_no_alerts[df_no_alerts['SA_Score'] <= args.sa_cutoff].copy()
    print(f"After SA filter (≤{args.sa_cutoff}): {len(df_filtered)} molecules")

    if len(df_filtered) == 0:
        print("No molecules passed SA filter. Relaxing to all alert-filtered molecules.")
        df_filtered = df_no_alerts.copy()

    # Step 3: Sort by primary property (see --primary_prop) first, then Quality_Score
    # Higher pIC50 is better (more potent)
    sort_cols = []
    sort_ascending = []
    
    if args.primary_prop in df_filtered.columns:
        sort_cols.append(args.primary_prop)
        sort_ascending.append(False)  # Higher pIC50 is better
        print(f"Prioritizing by {args.primary_prop} (higher is better)")
    
    if 'Quality_Score' in df_filtered.columns:
        sort_cols.append('Quality_Score')
        sort_ascending.append(False)
    
    if sort_cols:
        df_filtered = df_filtered.sort_values(sort_cols, ascending=sort_ascending)
    
    # Use a pool 5x larger than k to give MaxMin enough candidates
    pool_size = min(len(df_filtered), args.k * 5)
    df_pool = df_filtered.head(pool_size).reset_index(drop=True)
    print(f"Quality pool: top {len(df_pool)} by {sort_cols}")

    # Compute fingerprints
    fps, valid_idx = compute_fingerprints(df_pool['SMILES'].tolist())
    df_valid = df_pool.iloc[valid_idx].reset_index(drop=True)

    if len(df_valid) == 0:
        print("Error: no valid molecules for fingerprinting")
        sys.exit(1)

    # Pick k diverse compounds
    k = min(args.k, len(df_valid))
    pick_idx = pick_diverse(df_valid, fps, k)
    df_picked = df_valid.iloc[pick_idx].reset_index(drop=True)

    # Add rank column and ensure SMILES is first
    df_picked.insert(0, 'Rank', range(1, len(df_picked) + 1))
    cols = ['SMILES', 'Rank'] + [c for c in df_picked.columns if c not in ('SMILES', 'Rank')]
    df_picked = df_picked[cols]

    # Summary stats
    print(f"\nPicked {len(df_picked)} compounds:")
    if args.primary_prop in df_picked.columns:
        print(f"  {args.primary_prop}: {df_picked[args.primary_prop].mean():.3f} ± {df_picked[args.primary_prop].std():.3f}")
    if 'Quality_Score' in df_picked.columns:
        print(f"  Quality_Score: {df_picked['Quality_Score'].mean():.3f} ± {df_picked['Quality_Score'].std():.3f}")
    print(f"  SA_Score:      {df_picked['SA_Score'].mean():.2f} ± {df_picked['SA_Score'].std():.2f}")

    # Compute pairwise Tanimoto stats for picked set
    picked_fps = [fps[i] for i in pick_idx]
    tani_vals = []
    for i in range(len(picked_fps)):
        for j in range(i + 1, len(picked_fps)):
            tani_vals.append(DataStructs.TanimotoSimilarity(picked_fps[i], picked_fps[j]))
    if tani_vals:
        print(f"  Avg pairwise Tanimoto: {np.mean(tani_vals):.3f} (lower = more diverse)")

    # Save
    if args.output is None:
        out_dir = os.path.dirname(args.csv)
        out_path = os.path.join(out_dir, f"picked_{k}_compounds.csv")
    else:
        out_path = args.output
    df_picked.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    # Save .smi file if requested
    if args.smi:
        smi_path = out_path.replace('.csv', '.smi')
        with open(smi_path, 'w') as f:
            for smi in df_picked['SMILES'].tolist():
                f.write(f"{smi}\n")
        print(f"Saved SMILES to {smi_path}")


if __name__ == "__main__":
    main()
