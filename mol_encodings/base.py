
# featurizer.py
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from venv import logger
import numpy as np
from collections import Counter
import warnings
import time
logger = logging.getLogger(__name__)


class FeaturizationError(Exception):
    """
    Raised when a critical featurization error occurs that cannot be recovered from.
    For non-critical per-molecule failures, a warning is issued and the molecule is skipped.
    """
    pass


# ─────────────────────────────────────────────
# Abstract Base: BaseMolFeaturizer
# ─────────────────────────────────────────────

class BaseMolFeaturizer(ABC):
    """
    Abstract base class for all molecular featurizers.

    Defines the shared interface and workflow:
        1. fit(smiles_list)         → build vocabulary from dataset
        2. transform(smiles_list)   → featurize molecules using vocabulary
        3. fit_transform(smiles_list) → convenience: fit + transform in one call

    All subclasses must implement:
        - _build_vocab(smiles_list): build the vocabulary dict
        - _featurize_molecule(smiles): convert one SMILES to a feature vector

    Parameters
    ----------
    method : str
        Name of the featurization method ('brics', 'deepfmpo', 'graph').
    min_frequency : int
        Minimum frequency for a vocabulary token to be kept.
        Tokens appearing fewer times are discarded.
    max_vocab_size : Optional[int]
        Maximum number of tokens in the vocabulary.
        If None, no limit is applied.
    **kwargs
        Additional method-specific parameters passed to subclasses.
    """

    def __init__(
        self,
        method: str = 'graph',
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        self.method = method
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.kwargs = kwargs

        # Set after fit()
        self._vocab: Dict[str, Any] = {}
        self._is_fitted: bool = False
        self._feature_dim: Optional[int] = None

    # ── Public API ────────────────────────────

    def fit(self, smiles_list: List[str]) -> "BaseMolFeaturizer":
        """
        Build the vocabulary from the full dataset in one pass.

        Parameters
        ----------
        smiles_list : List[str]
            Full list of SMILES strings from the dataset.

        Returns
        -------
        self : BaseMolFeaturizer
            Returns self to allow method chaining.

        Raises
        ------
        FeaturizationError
            If vocabulary building fails critically.
        """
        logger.info(f"[{self.method}] Building vocabulary from {len(smiles_list)} molecules...")

        if not smiles_list:
            raise FeaturizationError("Cannot fit on an empty SMILES list.")

        self._vocab = self._build_vocab(smiles_list)
        self._is_fitted = True

        logger.info(f"[{self.method}] Vocabulary built. Size: {self._vocab.get('vocab_size', 'N/A')}")
        return self

    def transform(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Featurize all molecules using the fitted vocabulary.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to featurize.

        Returns
        -------
        result : Dict[str, Any]
            {
                'vocab':    vocabulary dict (see _build_vocab for structure),
                'features': np.ndarray of shape (n_valid_molecules, feature_dim),
                'metadata': processing statistics dict
            }

        Raises
        ------
        FeaturizationError
            If transform is called before fit.
        """
        if not self._is_fitted:
            raise FeaturizationError(
                "Featurizer is not fitted yet. Call fit() or fit_transform() first."
            )

        logger.info(f"[{self.method}] Featurizing {len(smiles_list)} molecules...")
        start_time = time.time()

        features = []
        n_failed = 0

        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0 and i > 0:
                logger.debug(f"[{self.method}] Progress: {i}/{len(smiles_list)}")

            try:
                feature_vector = self._featurize_molecule(smiles)

                if feature_vector is None:
                    warnings.warn(
                        f"[{self.method}] Featurization returned None for SMILES: '{smiles}'. Skipping.",
                        UserWarning,
                        stacklevel=2
                    )
                    n_failed += 1
                    continue

                features.append(feature_vector)

                # Infer feature dim from first successful featurization
                if self._feature_dim is None:
                    self._feature_dim = len(feature_vector)
            except Exception as e:
                # Log full details internally for debugging
                logger.exception(f"[{self.method}] Failed to featurize SMILES: '{smiles}'")
               
            # except Exception as e:
            #     warnings.warn(
            #         f"[{self.method}] Failed to featurize SMILES: '{smiles}'. Reason: {e}. Skipping.",
            #         UserWarning,
            #         stacklevel=2
            #     )
                n_failed += 1

        processing_time = time.time() - start_time
        feature_matrix = np.array(features) if features else np.empty((0, self._feature_dim or 0))

        if n_failed > 0:
            logger.warning(f"[{self.method}] {n_failed}/{len(smiles_list)} molecules failed featurization.")

        logger.info(
            f"[{self.method}] Done. "
            f"{len(features)} molecules featurized in {processing_time:.2f}s."
        )

        return {
            'vocab': self._vocab,
            'features': feature_matrix,          # shape: (n_molecules, feature_dim)
            'metadata': {
                'method':          self.method,
                'feature_dim':     self._feature_dim,
                'n_molecules':     len(features),
                'n_failed':        n_failed,
                'processing_time': round(processing_time, 4)
            }
        }

    def fit_transform(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Convenience method: fit vocabulary then featurize in one call.

        Parameters
        ----------
        smiles_list : List[str]
            Full list of SMILES strings.

        Returns
        -------
        result : Dict[str, Any]
            Same output as transform().
        """
        return self.fit(smiles_list).transform(smiles_list)

    def get_vocab(self) -> Dict[str, Any]:
        """
        Return the fitted vocabulary dictionary.

        Returns
        -------
        vocab : Dict[str, Any]
            The vocabulary built during fit().

        Raises
        ------
        FeaturizationError
            If called before fit().
        """
        if not self._is_fitted:
            raise FeaturizationError("Featurizer is not fitted yet. Call fit() first.")
        return self._vocab

    def get_feature_dim(self) -> Optional[int]:
        """
        Return the dimensionality of the feature vectors.

        Returns None if transform() has not been called yet,
        since feature_dim is inferred from the first successful featurization.

        Returns
        -------
        feature_dim : Optional[int]
        """
        return self._feature_dim

    def save_vocab(self, path: str) -> None:
        """
        Save the vocabulary to a JSON file.

        JSON is used instead of pickle to avoid CWE-502 (deserialization of
        untrusted data). All vocabulary values are simple types (str, int,
        dict, list) and are safe to JSON-encode.

        Parameters
        ----------
        path : str
            File path to save the vocabulary (e.g., 'vocab.json').

        Raises
        ------
        FeaturizationError
            If called before fit(), or if the file cannot be written.
        """
        if not self._is_fitted:
            raise FeaturizationError("No vocabulary to save. Call fit() first.")

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'method': self.method, 'vocab': self._vocab}, f)
        except (OSError, TypeError, ValueError) as e:
            raise FeaturizationError(f"Failed to save vocabulary to '{path}': {e}") from e

        logger.info(f"[{self.method}] Vocabulary saved to '{path}'.")

    @staticmethod
    def _restore_int_keys(obj: Any) -> Any:
        """Recursively convert dict keys that look like ints back to ints.

        JSON only supports string keys, so integer-keyed dicts (e.g.
        ``idx_to_atom``) get flattened on save. This restores them after load.
        """
        if isinstance(obj, dict):
            restored = {}
            for k, v in obj.items():
                if isinstance(k, str) and (k.lstrip('-').isdigit()):
                    restored[int(k)] = BaseMolFeaturizer._restore_int_keys(v)
                else:
                    restored[k] = BaseMolFeaturizer._restore_int_keys(v)
            return restored
        if isinstance(obj, list):
            return [BaseMolFeaturizer._restore_int_keys(v) for v in obj]
        return obj

    def load_vocab(self, path: str) -> "BaseMolFeaturizer":
        """
        Load a vocabulary from a JSON file.

        Parameters
        ----------
        path : str
            File path to load the vocabulary from (e.g., 'vocab.json').

        Returns
        -------
        self : BaseMolFeaturizer
            Returns self to allow method chaining.

        Raises
        ------
        FeaturizationError
            If the file cannot be read or the JSON data is corrupted.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FeaturizationError(f"Vocabulary file not found: '{path}'") from e
        except (json.JSONDecodeError, OSError) as e:
            raise FeaturizationError(f"Failed to load vocabulary from '{path}': file may be corrupted. {e}") from e

        if not isinstance(data, dict) or 'vocab' not in data:
            raise FeaturizationError(f"Vocabulary file '{path}' has an unrecognized format.")

        self._vocab = self._restore_int_keys(data['vocab'])
        self._is_fitted = True

        logger.info(f"[{self.method}] Vocabulary loaded from '{path}'.")
        return self

    # ── Abstract Methods (must implement in subclasses) ───

    @abstractmethod
    def _build_vocab(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Build and return the vocabulary from the full dataset.

        Called once during fit(). Should do a single pass over all SMILES,
        collect tokens (fragments or graph elements), apply frequency filtering,
        and return a structured vocabulary dict.

        Parameters
        ----------
        smiles_list : List[str]
            Full list of SMILES strings.

        Returns
        -------
        vocab : Dict[str, Any]
            Vocabulary dict. Structure depends on method family:
            - Fragment-based: {'fragment_to_idx': {...}, 'idx_to_fragment': {...}, 'vocab_size': int}
            - Graph-based:    {'atom_to_idx': {...}, 'bond_to_idx': {...}, 'vocab_size': int}
        """
        pass

    @abstractmethod
    def _featurize_molecule(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a single SMILES string to a feature vector using the fitted vocabulary.

        Called once per molecule during transform(). Should return None
        (not raise) for invalid or unfeaturizable molecules — the base class
        handles skipping and warning.

        Parameters
        ----------
        smiles : str
            A single SMILES string.

        Returns
        -------
        feature_vector : Optional[np.ndarray]
            1D array of shape (feature_dim,), or None if featurization fails.
        """
        pass

