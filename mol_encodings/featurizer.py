# MolFeaturizer - Simplified MARL Package

"""
mol_featurizer.py

Molecular featurization layer for the MARL package.
Sits between the dataset layer and the environment/algorithm layers.

Pipeline:
    BaseDataset (SMILES list)
        ↓
    MolFeaturizer.fit_transform(smiles_list)
        ↓
    Build Vocabulary  ← one pass over full dataset
        ↓
    Featurize All     ← convert each molecule using vocab
        ↓
    {vocab, features, metadata}  ← ready for environment

Class Hierarchy:
    BaseMolFeaturizer (abstract)
    ├── BaseFragmentFeaturizer (abstract, shared fragment utilities)
    │   ├── BRICSFeaturizer
    │   └── DeepFMPOFeaturizer
    └── GraphFeaturizer

Usage:
    featurizer = create_featurizer(method='graph')
    result = featurizer.fit_transform(smiles_list)
    # result = {
    #     'vocab':    { 'atom_to_idx': {...}, 'vocab_size': int, ... },
    #     'features': np.ndarray of shape (n_molecules, feature_dim),
    #     'metadata': { 'method': str, 'n_molecules': int, ... }
    # }
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────

# featurizer.py
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import warnings
import time


# Put imports in the factory function instead:

try:
    from .brics import BRICSFeaturizer
    from .deepfmpo import DeepFMPOFeaturizer
    from .graph import GraphFeaturizer
    from .base import BaseMolFeaturizer
except ImportError:
    # Allow running the file directly: python mol_encodings/featurizer.py
    from brics import BRICSFeaturizer
    from deepfmpo import DeepFMPOFeaturizer
    from graph import GraphFeaturizer
    from base import BaseMolFeaturizer  

# Using pickle for efficient storage of large vocabularies.
# For production systems with untrusted data, consider MessagePack or HDF5 instead.

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Abstract Base: BaseFragmentFeaturizer
# ─────────────────────────────────────────────


_SUPPORTED_METHODS = {
    'brics':    BRICSFeaturizer,
    'deepfmpo': DeepFMPOFeaturizer,
    'graph':    GraphFeaturizer,
}


def create_featurizer(method: str = 'graph', **kwargs: Any) -> BaseMolFeaturizer:
    """
    Factory function to instantiate the correct featurizer by method name.

    Parameters
    ----------
    method : str
        Featurization method. One of: 'brics', 'deepfmpo', 'graph'.
    **kwargs
        Additional parameters passed to the featurizer constructor.
        Examples:
            min_frequency=2, max_vocab_size=1000  (all methods)
            vector_type='count'                   (fragment methods)
            add_self_loops=True                   (graph method)

    Returns
    -------
    featurizer : BaseMolFeaturizer
        An instance of the appropriate featurizer class.

    Raises
    ------
    ValueError
        If method is not one of the supported methods.

    Examples
    --------
    >>> featurizer = create_featurizer('graph')
    >>> featurizer = create_featurizer('brics', min_frequency=2, vector_type='count')
    >>> featurizer = create_featurizer('deepfmpo', max_vocab_size=500)
    """
    method = method.lower().strip()

    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown featurization method: '{method}'. "
            f"Supported methods: {list(_SUPPORTED_METHODS.keys())}"
        )

    featurizer_class = _SUPPORTED_METHODS[method]
    logger.info(f"Creating featurizer: {featurizer_class.__name__}")
    return featurizer_class(**kwargs)




# ─────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # ── Sample dataset (SMILES list from BaseDataset) ──────────────────────
    # In practice: smiles_list = [sample['smiles'] for sample in dataset]
    sample_smiles = [
        'CCO',                          # ethanol
        'CC(=O)Oc1ccccc1C(=O)O',        # aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # testosterone
        'INVALID_SMILES',               # intentionally bad — will be skipped
        'c1ccccc1',                     # benzene
    ]

    # ── 1. Create featurizer using factory ─────────────────────────────────
    print("\n--- Graph Featurizer ---")
    graph_featurizer = create_featurizer('graph', add_self_loops=False)

    # ── 2. Fit on dataset (build vocabulary) ───────────────────────────────
    graph_featurizer.fit(sample_smiles)

    # ── 3. Transform molecules to features ─────────────────────────────────
    result = graph_featurizer.transform(sample_smiles)

    print(f"Method:       {result['metadata']['method']}")
    print(f"N molecules:  {result['metadata']['n_molecules']}")
    print(f"N failed:     {result['metadata']['n_failed']}")
    print(f"Feature dim:  {result['metadata']['feature_dim']}")
    print(f"Feature shape:{result['features'].shape}")

    # ── 4. Access vocabulary ────────────────────────────────────────────────
    vocab = graph_featurizer.get_vocab()
    print(f"\nAtom vocab size: {vocab['vocab_size']}")
    print(f"Atom types:      {list(vocab['atom_to_idx'].keys())}")
    print(f"Bond types:      {list(vocab['bond_to_idx'].keys())}")

    # ── 5. Save and load vocabulary ─────────────────────────────────────────
    graph_featurizer.save_vocab('/tmp/graph_vocab.pkl')

    new_featurizer = create_featurizer('graph')
    new_featurizer.load_vocab('/tmp/graph_vocab.pkl')
    print(f"\nLoaded vocab size: {new_featurizer.get_vocab()['vocab_size']}")

    # ── 6. fit_transform convenience method ────────────────────────────────
    print("\n--- BRICS Featurizer ---")
    brics_featurizer = create_featurizer('brics', min_frequency=1, vector_type='binary')
    brics_result = brics_featurizer.fit_transform(sample_smiles)

    print(f"Method:       {brics_result['metadata']['method']}")
    print(f"Vocab size:   {brics_result['vocab']['vocab_size']}")
    print(f"Feature shape:{brics_result['features'].shape}")

    # ── Save and load BRICS vocabulary ─────────────────────────────────────
    brics_featurizer.save_vocab('brics_vocab.pkl')

    loaded_brics = create_featurizer('brics')
    loaded_brics.load_vocab('brics_vocab.pkl')
    print(f"Loaded BRICS vocab size: {loaded_brics.get_vocab()['vocab_size']}")

    # ── 7. Switch between methods ───────────────────────────────────────────
    print("\n--- DeepFMPO Featurizer ---")
    deepfmpo_result = create_featurizer('deepfmpo', vector_type='count') \
                          .fit_transform(sample_smiles)

    print(f"Method:       {deepfmpo_result['metadata']['method']}")
    print(f"Vocab size:   {deepfmpo_result['vocab']['vocab_size']}")
    print(f"Feature shape:{deepfmpo_result['features'].shape}")

    # ── Output dict structure reference ────────────────────────────────────
    # result = {
    #     'vocab': {
    #         # Fragment-based (BRICS, DeepFMPO):
    #         'fragment_to_idx': {'CCO': 0, 'c1ccccc1': 1, ...},
    #         'idx_to_fragment': {0: 'CCO', 1: 'c1ccccc1', ...},
    #         'vocab_size':      int,
    #         'fragment_counts': {'CCO': 42, ...}
    #
    #         # Graph-based:
    #         # 'atom_to_idx':  {'<UNK>': 0, 'C': 1, 'N': 2, ...},
    #         # 'bond_to_idx':  {'SINGLE': 0, 'DOUBLE': 1, ...},
    #         # 'vocab_size':   int
    #     },
    #     'features': np.ndarray,   # shape: (n_molecules, feature_dim)
    #                               # for graph: (n_molecules,) object array of dicts
    #     'metadata': {
    #         'method':          'brics' | 'deepfmpo' | 'graph',
    #         'feature_dim':     int,
    #         'n_molecules':     int,
    #         'n_failed':        int,
    #         'processing_time': float
    #     }
    # }
