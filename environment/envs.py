"""
Environment creation utilities.

The main factory logic has moved to pipeline.env_factory.EnvironmentFactory.
This module provides make_parallel_envs() for backward compatibility.
"""

import warnings
from pipeline.env_factory import EnvironmentFactory


def make_parallel_envs(cfg, input_mols, decoding_smiles, scorer, shared_rest_session, train):
    """
    Backward-compatible wrapper around EnvironmentFactory.

    New code should use EnvironmentFactory directly:
        from pipeline import EnvironmentFactory
        factory = EnvironmentFactory(cfg, scorer, rest_session)
        envs = factory.create_envs(molecules, decodings)
    """
    warnings.warn(
        "make_parallel_envs() is deprecated. Use pipeline.EnvironmentFactory instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    factory = EnvironmentFactory(cfg, scorer=scorer, rest_session=shared_rest_session)
    return factory.create_envs(input_mols, decoding_smiles)

