"""
DataPipeline — Generic data loading and featurization orchestrator for MARL training.

Replaces the inline data loading logic in train.py. Integrates:
    - dataset/dataset.py (BaseDataset) for CSV loading and filtering
    - mol_encodings/ (create_featurizer) for molecular featurization
    - encoders/fmpo_utils/generate_encodings.py (encoding_maker) as legacy backend

Pipeline stages:
    1. Load    → BaseDataset reads CSV(s)
    2. Filter  → Optional filtering (remove molecules already satisfying criteria)
    3. Encode  → Featurize molecules into arrays + build vocabulary/decodings
    4. Output  → Returns encoded arrays, decodings dict, and config values

Usage:
    from pipeline import DataPipeline

    dp = DataPipeline(cfg)
    data = dp.run()
    # data.train_codes, data.eval_codes, data.decodings, etc.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dataset.dataset import BaseDataset
from mol_encodings import create_featurizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────

@dataclass
class PipelineOutput:
    """Container for all outputs of the data pipeline."""
    lead_codes: np.ndarray
    train_codes: np.ndarray
    eval_codes: np.ndarray
    decodings: Dict[str, Any]
    lead_smiles_dict: Dict[str, np.ndarray]
    max_frag: int
    num_bits: int
    freeze_encodings: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# DataPipeline
# ─────────────────────────────────────────────

class DataPipeline:
    """
    Orchestrates data loading, filtering, and featurization for MARL training.

    Supports two backends:
        - 'legacy': uses encoding_maker() from encoders/fmpo_utils (current working path)
        - 'featurizer': uses mol_encodings/ featurizer classes (for future use)

    Parameters
    ----------
    cfg : dict
        Full training configuration dictionary.
    backend : str
        Which encoding backend to use. 'legacy' (default) or 'featurizer'.
    """

    def __init__(self, cfg: Dict[str, Any], backend: str = 'legacy'):
        self.cfg = cfg
        self.backend = backend
        self._validate_config()

    def _validate_config(self) -> None:
        """Check that required config keys exist."""
        required_sections = ['deepfmpo', 'data', 'properties']
        missing = [s for s in required_sections if s not in self.cfg]
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")

    # ─── Public API ───────────────────────────────────────────

    def run(self, filter_func=None) -> PipelineOutput:
        """
        Execute the full pipeline: load → filter → encode.

        Parameters
        ----------
        filter_func : callable, optional
            A filtering function compatible with BaseDataset.apply_filter().
            If None and cfg['data']['filtering'] is True, uses default property filter.

        Returns
        -------
        PipelineOutput
            Dataclass containing all encoded data and metadata.
        """
        if self.backend == 'legacy':
            return self._run_legacy(filter_func)
        elif self.backend == 'featurizer':
            return self._run_featurizer(filter_func)
        else:
            raise ValueError(f"Unknown backend: '{self.backend}'. Use 'legacy' or 'featurizer'.")

    # ─── Legacy backend (encoding_maker) ──────────────────────

    def _run_legacy(self, filter_func=None) -> PipelineOutput:
        """
        Uses the existing encoding_maker() function for backward compatibility.
        Wraps it with BaseDataset for consistent data loading interface.
        """
        from encoders.fmpo_utils.generate_encodings import encoding_maker

        # Stage 1: Run encoding_maker (handles its own loading + fragmentation)
        lead_codes, train_codes, eval_codes, decodings, lead_smiles_dict, max_frag, num_bits, freeze_encodings = \
            encoding_maker(self.cfg)

        logger.info(f"Legacy pipeline: {len(train_codes)} train, {len(eval_codes)} eval molecules encoded")
        logger.info(f"Vocab size: {len(decodings)}, max_frag: {max_frag}, num_bits: {num_bits}")

        # Stage 2: Apply filtering if requested
        if filter_func is not None:
            train_codes = filter_func(train_codes, self.cfg, decodings)
            logger.info(f"After filtering: {len(train_codes)} train molecules remain")
        elif self.cfg['data'].get('filtering', False):
            from utils.analyse_properties import filter_good
            train_codes = filter_good(train_codes, self.cfg, decodings)
            logger.info(f"After default filtering: {len(train_codes)} train molecules remain")

        # Stage 3: Deduplicate
        train_codes = np.unique(train_codes, axis=0)
        logger.info(f"After deduplication: {len(train_codes)} unique train molecules")

        return PipelineOutput(
            lead_codes=lead_codes,
            train_codes=train_codes,
            eval_codes=eval_codes,
            decodings=decodings,
            lead_smiles_dict=lead_smiles_dict,
            max_frag=max_frag,
            num_bits=num_bits,
            freeze_encodings=freeze_encodings,
            metadata={
                'backend': 'legacy',
                'use_brics': self.cfg['deepfmpo'].get('use_brics', False),
                'use_reinvent': self.cfg['deepfmpo'].get('use_reinvent', False),
            }
        )

    # ─── Featurizer backend (mol_encodings/) ──────────────────

    def _run_featurizer(self, filter_func=None) -> PipelineOutput:
        """
        Uses BaseDataset + mol_encodings featurizers.
        This is the target architecture once featurizers are fully implemented.
        """
        deepfmpo_cfg = self.cfg['deepfmpo']

        # Stage 1: Load data using BaseDataset
        train_dataset = BaseDataset(
            file_path=deepfmpo_cfg['train_file'],
            smiles_column='smiles',
        )
        eval_dataset = BaseDataset(
            file_path=deepfmpo_cfg['eval_file'],
            smiles_column='smiles',
        )
        lead_dataset = BaseDataset(
            file_path=deepfmpo_cfg['lead_file'],
            smiles_column='smiles',
        )

        train_smiles = [train_dataset[i]['smiles'] for i in range(len(train_dataset))]
        eval_smiles = [eval_dataset[i]['smiles'] for i in range(len(eval_dataset))]
        lead_smiles = [lead_dataset[i]['smiles'] for i in range(len(lead_dataset))]

        logger.info(f"Loaded {len(train_smiles)} train, {len(eval_smiles)} eval, {len(lead_smiles)} lead molecules")

        # Stage 2: Determine featurization method from config
        if deepfmpo_cfg.get('use_brics', False):
            method = 'brics'
        elif deepfmpo_cfg.get('use_reinvent', False):
            method = 'deepfmpo'
        else:
            method = 'deepfmpo'  # default fragmentation

        featurizer = create_featurizer(method=method)

        # Fit vocabulary on all molecules combined (same as encoding_maker)
        all_smiles = lead_smiles + train_smiles + eval_smiles
        featurizer.fit(all_smiles)

        # Stage 3: Transform each set
        lead_result = featurizer.transform(lead_smiles)
        train_result = featurizer.transform(train_smiles)
        eval_result = featurizer.transform(eval_smiles)

        vocab = featurizer.get_vocab()
        max_frag = vocab.get('max_fragments', lead_result['metadata'].get('feature_dim', 0))
        num_bits = vocab.get('num_bits', vocab.get('vocab_size', 0))

        # Build decodings dict (fragment_code → mol/smiles mapping)
        decodings = vocab.get('decodings', {})

        # Build lead smiles dict
        lead_smiles_dict = {
            smi: code for smi, code in zip(lead_smiles, lead_result['features'])
            if code is not None
        }

        # Stage 4: Filter if requested
        train_codes = train_result['features']
        if filter_func is not None:
            train_codes = filter_func(train_codes, self.cfg, decodings)

        # Deduplicate
        train_codes = np.unique(train_codes, axis=0)

        return PipelineOutput(
            lead_codes=lead_result['features'],
            train_codes=train_codes,
            eval_codes=eval_result['features'],
            decodings=decodings,
            lead_smiles_dict=lead_smiles_dict,
            max_frag=max_frag,
            num_bits=num_bits,
            freeze_encodings=[],
            metadata={
                'backend': 'featurizer',
                'method': method,
                'vocab_size': vocab.get('vocab_size', 0),
                'n_failed_train': train_result['metadata'].get('n_failed', 0),
                'n_failed_eval': eval_result['metadata'].get('n_failed', 0),
            }
        )
