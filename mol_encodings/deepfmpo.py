
from typing import Any, List, Optional

from .fragments import BaseFragmentFeaturizer


class DeepFMPOFeaturizer(BaseFragmentFeaturizer):
    """
    Fragment-based featurizer using DeepFMPO fragmentation rules.

    DeepFMPO uses a different fragmentation strategy than BRICS, designed
    for fragment-based molecular optimization. It typically produces larger,
    more pharmacophore-like fragments.

    Parameters
    ----------
    min_frequency : int
        Minimum fragment frequency to include in vocabulary.
    max_vocab_size : Optional[int]
        Maximum vocabulary size.
    vector_type : str
        'binary' or 'count' (default: 'count' — DeepFMPO fragments carry weight).
    max_cuts : int
        Maximum number of bond cuts per molecule (controls fragment granularity).

    Output
    ------
    feature_vector : np.ndarray
        Shape: (vocab_size,), dtype float32.
        Count vector by default (fragment frequency within molecule).
    """

    def __init__(
        self,
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
        vector_type: str = 'count',
        max_cuts: int = 3,
        **kwargs: Any
    ) -> None:
        super().__init__(
            method='deepfmpo',
            min_frequency=min_frequency,
            max_vocab_size=max_vocab_size,
            vector_type=vector_type,
            **kwargs
        )
        self.max_cuts = max_cuts

    def _extract_fragments(self, smiles: str) -> List[str]:
        """
        Extract DeepFMPO fragments from a SMILES string.

        Parameters
        ----------
        smiles : str
            A single SMILES string.

        Returns
        -------
        fragments : List[str]
            List of fragment SMILES strings.
            Returns empty list if molecule is invalid or fragmentation fails.

        Expected shape after featurization:
            feature_vector: (vocab_size,) — one entry per vocab fragment
        """
        # TODO: Implement DeepFMPO fragmentation rules
        #
        # DeepFMPO fragmentation differs from BRICS in that it:
        #   - Cuts at rotatable bonds (not retrosynthetic bonds)
        #   - Preserves ring systems intact
        #   - Allows overlapping fragments (a bond can appear in multiple fragments)
        #   - Limits cuts to self.max_cuts per molecule
        #
        # Steps:
        #   1. Parse SMILES:
        #       mol = Chem.MolFromSmiles(smiles)
        #       if mol is None: return []
        #
        #   2. Identify cuttable bonds:
        #       # Rotatable, acyclic, single bonds between heavy atoms
        #       # Exclude bonds adjacent to rings
        #       cuttable = [bond for bond in mol.GetBonds() if is_cuttable(bond)]
        #
        #   3. Apply cuts (up to self.max_cuts):
        #       # Use rdkit.Chem.FragmentOnBonds or manual bond breaking
        #       from rdkit.Chem import FragmentOnBonds
        #       # Try combinations of up to max_cuts bonds
        #
        #   4. Collect and canonicalize fragment SMILES
        #
        #   5. Return list of canonical fragment SMILES
        #
        # Alternative: Use the original DeepFMPO codebase fragmentation logic
        # Reference implementation: https://github.com/jeremydouglass/deepfmpo
        #
        # Dependencies: rdkit
        # Expected output: ~3-10 fragments per drug-like molecule

        

        return []  # placeholder



