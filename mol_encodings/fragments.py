
import logging
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from .base import BaseMolFeaturizer
except ImportError:
    from base import BaseMolFeaturizer
import numpy as np

# Using pickle for efficient storage of large vocabularies.
# For production systems with untrusted data, consider MessagePack or HDF5 instead.

logger = logging.getLogger(__name__)


class BaseFragmentFeaturizer(BaseMolFeaturizer):
    """
    Abstract base class for fragment-based featurizers (BRICS, DeepFMPO).

    Provides shared utilities for:
        - Extracting fragments from a SMILES string
        - Building a fragment → index vocabulary
        - Filtering rare fragments by frequency
        - Converting a fragment list to a feature vector

    Subclasses (BRICSFeaturizer, DeepFMPOFeaturizer) only need to implement
    their specific fragment extraction logic in _extract_fragments().

    Parameters
    ----------
    method : str
        Passed through to BaseMolFeaturizer.
    min_frequency : int
        Minimum number of times a fragment must appear across the dataset
        to be included in the vocabulary.
    max_vocab_size : Optional[int]
        Maximum vocabulary size (keeps most frequent fragments).
    vector_type : str
        'binary'  → 1 if fragment present, 0 if absent  (shape: vocab_size)
        'count'   → raw fragment count per molecule      (shape: vocab_size)
    **kwargs
        Additional parameters passed to BaseMolFeaturizer.
    """

    def __init__(
        self,
        method: str,
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
        vector_type: str = 'binary',
        **kwargs: Any
    ) -> None:
        super().__init__(method=method, min_frequency=min_frequency,
                         max_vocab_size=max_vocab_size, **kwargs)
        self.vector_type = vector_type  # 'binary' or 'count'

    # ── Shared Fragment Utilities ─────────────

    def _extract_fragments(self, smiles: str) -> List[str]:
        """
        Extract fragments from a single SMILES string.

        This is the key method that differs between BRICS and DeepFMPO.
        Subclasses override this with their specific fragmentation rules.

        Parameters
        ----------
        smiles : str
            A single SMILES string.

        Returns
        -------
        fragments : List[str]
            List of fragment SMILES strings extracted from the molecule.
            Returns empty list if extraction fails.
        """
        # TODO: Override in subclasses with method-specific fragmentation.
        # This base version returns an empty list as a safe default.
        return []

    def _build_fragment_vocab(
        self,
        all_fragments: List[List[str]]
    ) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
        """
        Build a fragment → index mapping from all extracted fragments.

        Parameters
        ----------
        all_fragments : List[List[str]]
            One list of fragments per molecule (output of _extract_fragments
            called on every molecule in the dataset).

        Returns
        -------
        fragment_to_idx : Dict[str, int]
            Maps fragment SMILES → integer index.
        idx_to_fragment : Dict[int, str]
            Maps integer index → fragment SMILES.
        fragment_counts : Counter
            Raw frequency of each fragment across the dataset.
        """
        # Count fragment frequencies across the full dataset
        fragment_counts: Counter = Counter()
        for mol_fragments in all_fragments:
            fragment_counts.update(mol_fragments)

        # Apply frequency filter
        filtered = self._filter_rare_fragments(fragment_counts, self.min_frequency)

        # Apply vocab size limit (keep most frequent)
        if self.max_vocab_size is not None:
            most_common = filtered.most_common(self.max_vocab_size)
            filtered = Counter(dict(most_common))

        # Build bidirectional mappings
        sorted_fragments = sorted(filtered.keys())  # deterministic ordering
        fragment_to_idx = {frag: idx for idx, frag in enumerate(sorted_fragments)}
        idx_to_fragment = {idx: frag for frag, idx in fragment_to_idx.items()}

        return fragment_to_idx, idx_to_fragment, fragment_counts

    def _filter_rare_fragments(
        self,
        fragment_counts: Counter,
        min_freq: int
    ) -> Counter:
        """
        Remove fragments that appear fewer than min_freq times in the dataset.

        Parameters
        ----------
        fragment_counts : Counter
            Raw fragment frequency counts.
        min_freq : int
            Minimum frequency threshold.

        Returns
        -------
        filtered : Counter
            Fragment counts with rare fragments removed.
        """
        return Counter({
            frag: count
            for frag, count in fragment_counts.items()
            if count >= min_freq
        })

    def _fragment_to_feature(
        self,
        fragments: List[str],
        vocab: Dict[str, int]
    ) -> np.ndarray:
        """
        Convert a list of fragments to a fixed-length feature vector.

        Uses self.vector_type to determine encoding:
            'binary' → 1.0 if fragment in vocab, 0.0 otherwise
            'count'  → raw count of each vocab fragment in this molecule

        Parameters
        ----------
        fragments : List[str]
            Fragments extracted from one molecule.
        vocab : Dict[str, int]
            fragment_to_idx mapping from the fitted vocabulary.

        Returns
        -------
        feature_vector : np.ndarray
            Shape: (vocab_size,), dtype float32.
        """
        vocab_size = len(vocab)
        feature_vector = np.zeros(vocab_size, dtype=np.float32)

        for fragment in fragments:
            if fragment in vocab:
                idx = vocab[fragment]
                if self.vector_type == 'binary':
                    feature_vector[idx] = 1.0
                elif self.vector_type == 'count':
                    feature_vector[idx] += 1.0

        return feature_vector

    # ── Shared _build_vocab (uses shared utilities) ───

    def _build_vocab(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Build fragment vocabulary from the full dataset.

        One pass: extract fragments from every molecule, count frequencies,
        filter by min_frequency, build index mappings.

        Parameters
        ----------
        smiles_list : List[str]
            Full list of SMILES strings.

        Returns
        -------
        vocab : Dict[str, Any]
            {
                'fragment_to_idx': Dict[str, int],
                'idx_to_fragment': Dict[int, str],
                'vocab_size':      int,
                'fragment_counts': Dict[str, int]  ← raw frequencies
            }
        """
        all_fragments: List[List[str]] = []

        for smiles in smiles_list:
            try:
                fragments = self._extract_fragments(smiles)
                all_fragments.append(fragments)
            except Exception as e:
                
            # Log full details internally
                logger.exception(f"[{self.method}] Fragment extraction failed for '{smiles}'")
    
                all_fragments.append([])  # empty fragment list for failed molecules

        fragment_to_idx, idx_to_fragment, fragment_counts = self._build_fragment_vocab(all_fragments)

        logger.info(
            f"[{self.method}] Fragment vocab: {len(fragment_to_idx)} fragments "
            f"(min_frequency={self.min_frequency})."
        )

        return {
            'fragment_to_idx': fragment_to_idx,
            'idx_to_fragment': idx_to_fragment,
            'vocab_size':      len(fragment_to_idx),
            'fragment_counts': dict(fragment_counts)
        }

    def _featurize_molecule(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a single SMILES to a fragment feature vector.

        Uses the fitted fragment_to_idx vocabulary.

        Parameters
        ----------
        smiles : str
            A single SMILES string.

        Returns
        -------
        feature_vector : Optional[np.ndarray]
            Shape: (vocab_size,), or None if extraction fails.
        """
        try:
            fragments = self._extract_fragments(smiles)
            vocab = self._vocab.get('fragment_to_idx', {})
            return self._fragment_to_feature(fragments, vocab)
        except Exception:
            return None

