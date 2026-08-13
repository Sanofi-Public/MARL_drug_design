"""
Molecule buffer for managing the exploration pool in MARL environments.

Supports two eviction strategies:
  - "fifo": Simple first-in first-out (oldest molecule removed)
  - "diversity": Evict the most redundant molecule (smallest Hamming distance to nearest neighbor)
"""

import random
import numpy as np


class MoleculeBuffer:
    """Buffer that stores molecule arrays and manages eviction when full.
    
    Parameters
    ----------
    molecule_array : np.ndarray
        Initial molecules, shape (n_mols, n_frags, encoding_dim).
    max_size : int
        Maximum number of molecules in the buffer.
    sampling_type : str
        "fifo" or "diversity".
    """

    def __init__(self, molecule_array, max_size, sampling_type="diversity"):
        self.molecule_array = molecule_array.copy()
        self.max_size = max(len(molecule_array), max_size)
        self.name_list = [f"mol_{i}" for i in range(len(self.molecule_array))]

        if sampling_type not in ("fifo", "diversity"):
            raise ValueError(f"Unknown sampling_type '{sampling_type}'. Use 'fifo' or 'diversity'.")
        self.sampling_type = sampling_type

    def __len__(self):
        return len(self.molecule_array)

    def sample_index(self, rng=None):
        """Return a random index into the buffer."""
        if rng is not None:
            return rng.choice(range(len(self.molecule_array)))
        return random.choice(range(len(self.molecule_array)))

    def get(self, index):
        """Return molecule array at *index*."""
        return self.molecule_array[index]

    def try_add(self, new_mol, episode_number, current_step):
        """Add *new_mol* to the buffer if it is not a duplicate.

        If the buffer is full the configured eviction strategy is used.
        Returns True if the molecule was added, False otherwise.
        """
        if self._is_duplicate(new_mol):
            return False

        if len(self.molecule_array) >= self.max_size:
            self._evict()

        self.molecule_array = np.concatenate(
            (self.molecule_array, [new_mol]), axis=0
        )
        self.name_list.append(f"mol_{episode_number}_{current_step}")
        return True
    
    def _is_duplicate(self, mol):
        return any(np.array_equal(mol, m) for m in self.molecule_array)

    def _evict(self):
        if self.sampling_type == "fifo":
            self._evict_fifo()
        else:
            self._evict_most_redundant()

    def _evict_fifo(self):
        """Remove the oldest molecule (index 0)."""
        self.molecule_array = self.molecule_array[1:]
        self.name_list = self.name_list[1:]

    def _evict_most_redundant(self):
        """Remove the molecule whose nearest neighbour is closest (Hamming distance).

        For buffers > 200 molecules a random subsample is used for efficiency.
        """
        n = len(self.molecule_array)
        if n <= 1:
            return

        flat = self.molecule_array.reshape(n, -1)

        if n > 200:
            indices = np.random.choice(n, 200, replace=False)
            flat_sub = flat[indices]
            dists = np.abs(flat_sub[:, None, :] - flat_sub[None, :, :]).sum(axis=2)
            np.fill_diagonal(dists, dists.max() + 1)
            min_dists = dists.min(axis=1)
            evict_idx = indices[np.argmin(min_dists)]
        else:
            dists = np.abs(flat[:, None, :] - flat[None, :, :]).sum(axis=2)
            np.fill_diagonal(dists, dists.max() + 1)
            min_dists = dists.min(axis=1)
            evict_idx = np.argmin(min_dists)

        self.molecule_array = np.delete(self.molecule_array, evict_idx, axis=0)
        del self.name_list[evict_idx]


def should_add_to_buffer(prop_indicator, strategy="stratified", min_props_satisfied=1):
    """Decide whether a molecule should be added to the exploration buffer.

    Parameters
    ----------
    prop_indicator : list[float]
        Binary indicator list (0.0 or 1.0) for each property.
    strategy : str
        ``"stratified"`` – probability increases with the number of satisfied
        properties (original implementation).
        ``"filtered"`` – add if at least *min_props_satisfied* properties are
        satisfied, with a baseline 10% probability regardless.
    min_props_satisfied : int
        Minimum number of properties that must be satisfied for the
        ``"filtered"`` strategy.  Ignored when *strategy* is ``"stratified"``.

    Returns
    -------
    bool
    """
    n_satisfied = int(sum(prop_indicator))

    if strategy == "stratified":
        add_probability = min(0.5, 0.05 + 0.1 * n_satisfied)
        return random.random() < add_probability

    if strategy == "filtered":
        return n_satisfied >= min_props_satisfied and random.random() < 0.1

    raise ValueError(
        f"Unknown addition strategy '{strategy}'. Use 'stratified' or 'filtered'."
    )
