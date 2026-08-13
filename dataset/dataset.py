
# -----------------------------------------------------------------------------
import logging
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BaseDataset")

# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class DatasetError(Exception):
    """Raised for errors during dataset loading or access."""
    pass


# -----------------------------------------------------------------------------
# BaseDataset Class
# -----------------------------------------------------------------------------

class BaseDataset:
    """
    Abstract base class for molecular datasets in the MARL package.

    Handles loading from CSV, user-defined filtering, train/val/test splitting,
    and sample access. All filtering is delegated to the user via apply_filter().

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing molecular data.
    smiles_column : str
        Name of the column containing SMILES strings. Default: 'smiles'.
    id_column : str, optional
        Name of the column to use as sample IDs. If None, uses row index.
    property_columns : list of str, optional
        Names of columns to include as properties per sample.
        If None, no properties are loaded.
    lazy_load : bool
        If True, data is not loaded until explicitly needed. Default: False.
    random_seed : int
        Seed for reproducible train/val/test splits. Default: 42.

    Attributes
    ----------
    _data : pd.DataFrame or None
        The currently active (possibly filtered) dataset.
    _original_data : pd.DataFrame or None
        The original loaded data before any filters are applied.
    _splits : dict
        Dictionary holding 'train', 'val', 'test' DataFrames after splitting.
    _filter_history : list of str
        Log of all filters applied and their effects.

    Example
    -------
    >>> dataset = BaseDataset(
    ...     file_path="molecules.csv",
    ...     smiles_column="smiles",
    ...     id_column="mol_id",
    ...     property_columns=["logP", "MW"],
    ...     random_seed=42
    ... )
    >>> dataset.load_data()
    >>> dataset.apply_filter(lambda df: df[df["smiles"].str.len() > 5])
    >>> dataset.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    >>> sample = dataset[0]
    """

    def __init__(
        self,
        file_path: str,
        smiles_column: str = "smiles",
        id_column: Optional[str] = None,
        property_columns: Optional[List[str]] = None,
        lazy_load: bool = False,
        random_seed: int = 42,
    ):
        self.file_path = file_path
        self.smiles_column = smiles_column
        self.id_column = id_column
        self.property_columns = property_columns or []
        self.lazy_load = lazy_load
        self.random_seed = random_seed

        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self._original_data: Optional[pd.DataFrame] = None
        self._splits: Dict[str, pd.DataFrame] = {}
        self._filter_history: List[str] = []

        # Load immediately unless lazy loading is requested
        if not self.lazy_load:
            self.load_data()

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_data(self) -> None:
        """
        Load molecular data from the CSV file specified in file_path.

        Validates that the SMILES column exists. Optionally validates
        id_column and property_columns if provided.

        Raises
        ------
        DatasetError
            If the file cannot be read or required columns are missing.
        """
        logger.info(f"Loading data from: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise DatasetError(f"File not found: {self.file_path}")
        except Exception as e:
            raise DatasetError(f"Failed to read CSV file: {e}")

        # Validate required SMILES column
        if self.smiles_column not in df.columns:
            raise DatasetError(
                f"SMILES column '{self.smiles_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Validate optional id column
        if self.id_column and self.id_column not in df.columns:
            raise DatasetError(
                f"ID column '{self.id_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Validate optional property columns
        missing_props = [c for c in self.property_columns if c not in df.columns]
        if missing_props:
            raise DatasetError(
                f"Property columns not found in CSV: {missing_props}. "
                f"Available columns: {list(df.columns)}"
            )

        self._original_data = df.copy()
        self._data = df.copy()
        self._filter_history = []  # Reset history on fresh load
        self._splits = {}

        logger.info(f"Loaded {len(self._data)} samples.")

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def apply_filter(self, filter_func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Apply a user-defined filter function to the current dataset.

        The filter function receives the full DataFrame and must return
        a filtered DataFrame. Filters are applied in sequence (chaining).
        Each application is logged in _filter_history.

        Parameters
        ----------
        filter_func : Callable[[pd.DataFrame], pd.DataFrame]
            A function that takes a DataFrame and returns a filtered DataFrame.
            The user has full control over the filtering logic.

        Raises
        ------
        DatasetError
            If data has not been loaded yet.

        Example
        -------
        # Filter out short SMILES (custom validity proxy)
        >>> dataset.apply_filter(lambda df: df[df["smiles"].str.len() > 5])

        # Filter out duplicates
        >>> dataset.apply_filter(lambda df: df.drop_duplicates(subset=["smiles"]))

        # Filter by property range
        >>> dataset.apply_filter(lambda df: df[df["MW"].between(100, 500)])

        # Use a named function for clarity
        >>> def remove_invalid(df):
        ...     from rdkit import Chem
        ...     mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        ...     return df[mask]
        >>> dataset.apply_filter(remove_invalid)

        Notes
        -----
        - Extension point: Add any filter logic here without modifying the class.
        - Filters are cumulative; call reset_filters() to start fresh.
        """
        self._ensure_loaded()

        size_before = len(self._data)

        try:
            filtered_df = filter_func(self._data)
        except Exception as e:
            raise DatasetError(f"Filter function raised an error: {e}")

        if not isinstance(filtered_df, pd.DataFrame):
            raise DatasetError(
                "Filter function must return a pandas DataFrame, "
                f"got {type(filtered_df)} instead."
            )

        self._data = filtered_df.reset_index(drop=True)
        size_after = len(self._data)
        removed = size_before - size_after

        # Log filter effect
        filter_name = getattr(filter_func, "__name__", "<lambda>")
        log_entry = (
            f"Filter '{filter_name}': {size_before} → {size_after} "
            f"({removed} removed)"
        )
        self._filter_history.append(log_entry)
        logger.info(log_entry)

    def reset_filters(self) -> None:
        """
        Reset the dataset to the original loaded data, removing all applied filters.

        Also clears any existing splits. Call split_data() again after resetting.

        Raises
        ------
        DatasetError
            If data has not been loaded yet.
        """
        self._ensure_loaded()

        self._data = self._original_data.copy()
        self._splits = {}
        self._filter_history = []
        logger.info(f"Filters reset. Dataset restored to {len(self._data)} samples.")

    # -------------------------------------------------------------------------
    # Splitting
    # -------------------------------------------------------------------------

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> None:
        """
        Split the current (filtered) dataset into train, val, and test sets.

        Splits are reproducible via the random_seed set at initialization.
        Ratios must sum to 1.0.

        Parameters
        ----------
        train_ratio : float
            Proportion of data for training. Default: 0.8.
        val_ratio : float
            Proportion of data for validation. Default: 0.1.
        test_ratio : float
            Proportion of data for testing. Default: 0.1.
        shuffle : bool
            Whether to shuffle before splitting. Default: True.

        Raises
        ------
        DatasetError
            If data has not been loaded, or if ratios do not sum to 1.0.

        Example
        -------
        >>> dataset.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        >>> train_data = dataset.get_split("train")
        """
        self._ensure_loaded()

        # Validate ratios
        total = round(train_ratio + val_ratio + test_ratio, 10)
        if abs(total - 1.0) > 1e-6:
            raise DatasetError(
                f"Split ratios must sum to 1.0, got {total:.4f}. "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        df = self._data.copy()

        if shuffle:
            df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self._splits["train"] = df.iloc[:n_train].reset_index(drop=True)
        self._splits["val"] = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        self._splits["test"] = df.iloc[n_train + n_val :].reset_index(drop=True)

        logger.info(
            f"Split complete → train: {len(self._splits['train'])}, "
            f"val: {len(self._splits['val'])}, "
            f"test: {len(self._splits['test'])}"
        )

    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of samples in the current (filtered) dataset.

        Returns
        -------
        int
            Number of samples.

        Raises
        ------
        DatasetError
            If data has not been loaded yet.
        """
        self._ensure_loaded()
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single sample as a dictionary.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict with keys:
            - 'id'         : Sample ID (from id_column or row index)
            - 'smiles'     : SMILES string
            - 'properties' : Dict of property_column values (empty if none specified)

        Raises
        ------
        DatasetError
            If data has not been loaded yet.
        IndexError
            If idx is out of range.

        Example
        -------
        >>> sample = dataset[0]
        >>> print(sample["smiles"])
        >>> print(sample["properties"]["logP"])
        """
        self._ensure_loaded()

        if idx < 0 or idx >= len(self._data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._data)}."
            )

        row = self._data.iloc[idx]

        sample_id = row[self.id_column] if self.id_column else idx
        smiles = row[self.smiles_column]
        properties = {col: row[col] for col in self.property_columns}

        return {
            "id": sample_id,
            "smiles": smiles,
            "properties": properties,
        }

    def get_split(self, split_name: str) -> pd.DataFrame:
        """
        Return a specific data split as a DataFrame.

        Parameters
        ----------
        split_name : str
            One of 'train', 'val', or 'test'.

        Returns
        -------
        pd.DataFrame
            The requested split.

        Raises
        ------
        DatasetError
            If split_data() has not been called yet, or split_name is invalid.

        Example
        -------
        >>> train_df = dataset.get_split("train")
        >>> val_df = dataset.get_split("val")
        """
        valid_splits = ["train", "val", "test"]

        if split_name not in valid_splits:
            raise DatasetError(
                f"Invalid split name '{split_name}'. Choose from: {valid_splits}"
            )

        if not self._splits:
            raise DatasetError(
                "No splits found. Call split_data() before accessing splits."
            )

        return self._splits[split_name]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return basic statistics about the current dataset state.

        Returns
        -------
        dict with keys:
            - 'total_size'       : Current number of samples (after filtering)
            - 'original_size'    : Number of samples before any filtering
            - 'removed_total'    : Total samples removed by filters
            - 'split_sizes'      : Dict of split sizes (empty if not split yet)
            - 'property_ranges'  : Min/max per property column (if any)
            - 'filter_history'   : List of filter log entries

        Raises
        ------
        DatasetError
            If data has not been loaded yet.

        Example
        -------
        >>> stats = dataset.get_statistics()
        >>> print(stats["total_size"])
        >>> print(stats["property_ranges"])
        """
        self._ensure_loaded()

        original_size = len(self._original_data)
        current_size = len(self._data)

        split_sizes = {
            name: len(split) for name, split in self._splits.items()
        }

        # Property ranges (only if property columns are defined)
        property_ranges = {}
        for col in self.property_columns:
            if col in self._data.columns:
                property_ranges[col] = {
                    "min": float(self._data[col].min()),
                    "max": float(self._data[col].max()),
                    "mean": float(self._data[col].mean()),
                }

        return {
            "total_size": current_size,
            "original_size": original_size,
            "removed_total": original_size - current_size,
            "split_sizes": split_sizes,
            "property_ranges": property_ranges,
            "filter_history": self._filter_history.copy(),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """
        Ensure data has been loaded. Triggers load_data() if lazy loading is on.

        Raises
        ------
        DatasetError
            If data is not loaded and lazy loading is disabled.
        """
        if self._data is None:
            if self.lazy_load:
                logger.info("Lazy load triggered.")
                self.load_data()
            else:
                raise DatasetError(
                    "Data not loaded. Call load_data() first, "
                    "or set lazy_load=True at initialization."
                )

    def __repr__(self) -> str:
        size = len(self._data) if self._data is not None else "not loaded"
        return (
            f"BaseDataset("
            f"file='{self.file_path}', "
            f"samples={size}, "
            f"filters_applied={len(self._filter_history)})"
        )




if __name__ == "__main__":


    dataset = BaseDataset(
        file_path="",
        smiles_column="SMILES",
        random_seed=42,
    )

    print(dataset)

    dataset.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    train_df = dataset.get_split("train")
    val_df   = dataset.get_split("val")
    test_df  = dataset.get_split("test")

    sample = dataset[0]
    print(sample)

    stats = dataset.get_statistics()
    print(f"Original size : {stats['original_size']}")
    print(f"After filters : {stats['total_size']}")
    print(f"Removed total : {stats['removed_total']}")
    print(f"Split sizes   : {stats['split_sizes']}")
