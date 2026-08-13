import sys
import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
import subprocess
import logging
import warnings
import requests
import matplotlib.pyplot as plt
import json
import argparse
import encoders.fmpo_utils.mol_utils as mol_utils

# Disable werkzeug logging
log = logging.getLogger('werkzeug')
log.disabled = True
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
import time

sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer

# Import from reward_functions (no circular import now)
from utils.reward_functions import (
    get_score, start_rest_subprocess, wait_for_rest_api, 
    rest_scorer, rdkit_scorer, prop_dict, check_bounds_all
)


def decode(x, translation):
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
    fs = [Chem.Mol(translation[e]) for e in enc]
    
    try:
        return mol_utils.join_fragments(fs)
    except:
        raise RuntimeError("Something went wrong when joining fragments.")


def filter_good(X, cfg, decodings, return_all=False):
    """
    Discard molecules which fulfill all targets (used to remove too good lead molecules).
    Uses scorers from reward_functions.py.
    """
    print("Filtering good molecules based on property bounds defined in config")
    prop_list = cfg['properties']['names']
    bounds = cfg['properties']['bounds']
    use_rest = cfg['properties']['use_scorer']
    prop_types = cfg['properties'].get('types', [None] * len(prop_list))
    
    # Determine scoring mode: all-REST, all-RDKit, or mixed
    rest_indices = [i for i, t in enumerate(prop_types) if t is not None]
    rdkit_indices = [i for i, t in enumerate(prop_types) if t is None]
    is_mixed = len(rest_indices) > 0 and len(rdkit_indices) > 0
    needs_rest = use_rest or len(rest_indices) > 0
    
    # Decode molecules to SMILES
    X_mols = [decode(X[i], decodings) for i in range(X.shape[0])]
    X_smiles = [Chem.MolToSmiles(mol) for mol in X_mols]
    
    if is_mixed or (needs_rest and not use_rest):
        # Mixed mode: score REST and RDKit properties separately, then merge
        rest_process = start_rest_subprocess()
        rest_url = "http://localhost:2000"
        rest_session = requests.Session()
        wait_for_rest_api(rest_url)
        
        rest_names = [prop_list[i] for i in rest_indices]
        rest_scores = rest_scorer(rest_names, X_smiles, rest_url, rest_session)
        # rest_scores: list of lists, one per REST property, each with len(X_smiles) values
        
        rdkit_funcs = [prop_dict[prop_list[i]] for i in rdkit_indices]
        
        X_scores = []
        for mol_idx in range(len(X_mols)):
            scores = [None] * len(prop_list)
            for j, idx in enumerate(rest_indices):
                scores[idx] = rest_scores[j][mol_idx]
            for j, idx in enumerate(rdkit_indices):
                scores[idx] = rdkit_funcs[j](X_mols[mol_idx])
            X_scores.append(tuple(scores))
        
        rest_process.terminate()
        rest_process.wait()
    elif use_rest:
        rest_process = start_rest_subprocess()
        rest_url = "http://localhost:2000"
        rest_session = requests.Session()
        wait_for_rest_api(rest_url)
        
        scores = rest_scorer(prop_list, X_smiles, rest_url, rest_session)
        X_scores = list(zip(*scores))
        
        rest_process.terminate()
        rest_process.wait()
    else:
        model_list = [prop_dict[x] for x in prop_list]
        X_scores = [rdkit_scorer(X_mols[i], model_list) for i in range(len(X_mols))]
    
    print(X_scores[0])

    # Filter out molecules that meet all property bounds
    if return_all:
        X_bounds = [check_bounds([X_scores[i][j] for j in range(len(prop_list))], bounds) for i in range(len(X_mols))]
        return X_scores, X_mols, X_bounds
    else:
        X_bounds = [
            check_bounds([X_scores[i][j] for j in range(len(prop_list))], bounds)
            for i in range(X.shape[0])
        ]
        X = [X[i] for i in range(X.shape[0]) if not X_bounds[i]]
    
    return np.asarray(X)


def smiles_analysis(smiles, cfg):
    """
    Analyze SMILES strings and return their property scores.
    Uses scorers from reward_functions.py.
    
    Args:
        smiles: List of SMILES strings
        cfg: Configuration dictionary with properties settings
        
    Returns:
        DataFrame with SMILES, scores, and bound checks
    """
    prop_list = cfg['properties']['names']
    use_rest = cfg['properties']['use_scorer']
    bounds = cfg['properties']['bounds']
    prop_types = cfg['properties'].get('types', [None] * len(prop_list))
    
    # Determine scoring mode: all-REST, all-RDKit, or mixed
    rest_indices = [i for i, t in enumerate(prop_types) if t is not None]
    rdkit_indices = [i for i, t in enumerate(prop_types) if t is None]
    is_mixed = len(rest_indices) > 0 and len(rdkit_indices) > 0
    needs_rest = use_rest or len(rest_indices) > 0
    
    X_mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    
    if is_mixed or (needs_rest and not use_rest):
        # Mixed mode: score REST and RDKit properties separately, then merge
        rest_process = start_rest_subprocess()
        rest_url = "http://localhost:2000"
        rest_session = requests.Session()
        wait_for_rest_api(rest_url)
        
        rest_names = [prop_list[i] for i in rest_indices]
        rest_scores = rest_scorer(rest_names, smiles, rest_url, rest_session)
        
        rdkit_funcs = [prop_dict[prop_list[i]] for i in rdkit_indices]
        
        X_scores = []
        for mol_idx in range(len(X_mols)):
            scores = [None] * len(prop_list)
            for j, idx in enumerate(rest_indices):
                scores[idx] = rest_scores[j][mol_idx]
            for j, idx in enumerate(rdkit_indices):
                scores[idx] = rdkit_funcs[j](X_mols[mol_idx])
            X_scores.append(tuple(scores))
        
        rest_process.terminate()
        rest_process.wait()
    elif use_rest:
        rest_process = start_rest_subprocess()
        rest_url = "http://localhost:2000"
        rest_session = requests.Session()
        wait_for_rest_api(rest_url)
        
        scores = rest_scorer(prop_list, smiles, rest_url, rest_session)
        X_scores = list(zip(*scores))
        
        rest_process.terminate()
        rest_process.wait()
    else:
        model_list = [prop_dict[x] for x in prop_list]
        X_scores = [rdkit_scorer(X_mols[i], model_list) for i in range(len(X_mols))]
    
    score_df = pd.DataFrame(X_scores, columns=prop_list)
    score_df['smiles'] = smiles
    
    for score in prop_list:
        score_df[score+'_in_bounds'] = score_df.apply(
            lambda row: bounds[prop_list.index(score)][0] <= row[score] <= bounds[prop_list.index(score)][1],
            axis=1
        )
    score_df['all_scores_in_bounds'] = score_df.apply(
        lambda row: all(row[score+'_in_bounds'] for score in prop_list),
        axis=1
    )
    return score_df


def check_bounds(scores, bounds):
    """Check if all scores are within their respective bounds."""
    for i, score in enumerate(scores):
        if score < bounds[i][0] or score > bounds[i][1]:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate molecular properties using RDKit and ML models.")
