"""
EnvironmentFactory — Generic environment creation for MARL training.

Encapsulates the config extraction and environment instantiation logic
previously spread across envs.py's make_parallel_envs().

Usage:
    from pipeline.env_factory import EnvironmentFactory

    env_factory = EnvironmentFactory(cfg, scorer, rest_session)
    train_env = env_factory.create_envs(train_mols, decodings)
    eval_env  = env_factory.create_envs(eval_mols, decodings)
"""

import logging
import random
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

logger = logging.getLogger(__name__)


class EnvironmentFactory:
    """
    Creates vectorized parallel molecule environments for MARL training.

    Extracts environment parameters from the config once at construction,
    then provides a clean interface for creating train and eval environments.

    Parameters
    ----------
    cfg : dict
        Full training configuration dictionary.
    scorer : str or None
        Scorer type ('rest' or None).
    rest_session : requests.Session or None
        Shared REST session for external scoring.
    """

    def __init__(self, cfg: Dict[str, Any], scorer: Any = None, rest_session: Any = None):
        self.cfg = cfg
        self.scorer = scorer
        self.rest_session = rest_session
        self._parse_config()

    def _parse_config(self) -> None:
        """Extract all environment-related parameters from config."""
        env_cfg = self.cfg['env']
        deepfmpo_cfg = self.cfg['deepfmpo']
        prop_cfg = self.cfg['properties']
        gnn_cfg = self.cfg.get('gnn', {})
        repr_cfg = self.cfg.get('representation', {})
        curriculum_cfg = self.cfg.get('curriculum', {})

        # Core environment parameters
        self.env_id = env_cfg['env_id']
        self.n_agents = env_cfg['n_agents']
        self.parallel_envs = env_cfg['parallel_envs']
        self.max_ep_length = env_cfg['max_ep_length']
        self.wrappers = env_cfg['wrappers']
        self.seed = self.cfg.get('seed', 1)

        # Fragment swap parameters
        self.max_swap = deepfmpo_cfg['MAX_SWAP']
        self.max_fragments = deepfmpo_cfg['MAX_FRAGMENTS']
        self.n_actions = self.max_fragments * self.max_swap + 1
        self.freeze_encodings = deepfmpo_cfg.get('freeze_encodings', [])
        self.use_brics = deepfmpo_cfg.get('use_brics', False)

        # Reward parameters
        self.big_reward = env_cfg['big_reward']
        self.small_reward = env_cfg['small_reward']
        self.repeat_penalty = env_cfg['repeat_penalty']
        self.novelty_reward = env_cfg['novelty_reward']
        self.multiplier = env_cfg['multiplier']
        self.credit_window = env_cfg['credit_window']
        self.shared_credit = env_cfg['shared_credit']

        # Property parameters
        self.prop_names = prop_cfg['names']
        self.prop_bounds = prop_cfg['bounds']
        self.use_scorer = prop_cfg['use_scorer']
        self.reward_types = prop_cfg.get('reward_types', [None] * len(self.prop_names))
        self.reward_args = prop_cfg.get('reward_args', [None] * len(self.prop_names))
        self.prop_types = prop_cfg.get('types', [None] * len(self.prop_names))

        # Novelty and buffer parameters
        self.novelty_mode = env_cfg.get('novelty_mode', 'local')
        self.buffer_size = env_cfg.get('buffer_size', 500)
        self.sampling_type = env_cfg.get('sampling_type', 'diversity')
        self.addition_strategy = env_cfg.get('addition_strategy', 'stratified')
        self.min_props_satisfied = env_cfg.get('min_props_satisfied', 1)

        # Invalid action handling
        self.invalid_penalty_mult = env_cfg.get('invalid_penalty_mult', 5.0)
        self.terminate_on_invalid = env_cfg.get('terminate_on_invalid', False)

        # GNN parameters
        self.use_gnn = gnn_cfg.get('use_gnn', False)
        self.gnn_model = gnn_cfg.get('gnn_model', '')

        # Representation type
        self.representation_type = repr_cfg.get('type', 'fragment')
        self.use_legacy = repr_cfg.get('use_legacy', True)

        # Goal conditioning
        self.goal_conditioned = curriculum_cfg.get('goal_conditioned', False)

        # Entry point for legacy environments
        self.entry_point = env_cfg.get(
            'entry_point', 'environment.fmpo_collect_v2:madfmpo_drugenv'
        )

    def create_envs(
        self,
        molecules: np.ndarray,
        decodings: Dict[str, Any],
    ) -> SubprocVecEnv:
        """
        Create vectorized parallel environments.

        Parameters
        ----------
        molecules : np.ndarray
            Encoded molecule arrays for the environment to sample from.
        decodings : dict
            Fragment code → RDKit Mol mapping for decoding.

        Returns
        -------
        SubprocVecEnv
            Vectorized parallel environments ready for training or evaluation.
        """
        # Pre-build BRICS cache once if using modular envs with BRICS
        brics_cache = self._build_brics_cache(decodings)

        # Register legacy environment
        if self.use_legacy:
            self._register_legacy_env()

        # Build environment thunks
        env_thunks = [
            partial(self._make_single_env, i, molecules, decodings, brics_cache)
            for i in range(self.parallel_envs)
        ]

        envs = SubprocVecEnv(env_thunks, start_method='fork')
        logger.info(
            f"Created {self.parallel_envs} parallel environments "
            f"(representation={self.representation_type}, legacy={self.use_legacy})"
        )
        return envs

    def _make_single_env(
        self,
        index: int,
        molecules: np.ndarray,
        decodings: Dict[str, Any],
        brics_cache: Optional[Dict] = None,
    ):
        """Factory function for a single environment instance."""
        from utils.wrappers import RecordEpisodeStatistics, TimeLimit

        env_kwargs = self._build_env_kwargs(index, molecules, decodings)

        if self.use_legacy:
            from gymnasium import make
            env = make(self.env_id, **env_kwargs)
            env = env.unwrapped
        else:
            env = self._create_modular_env(env_kwargs, brics_cache)

        env.seed(self.seed)

        # Apply wrappers
        for wrapper in self.wrappers:
            if wrapper == 'TimeLimit':
                env = TimeLimit(env, max_episode_steps=self.max_ep_length)
            elif wrapper == 'RecordStats':
                env = RecordEpisodeStatistics(
                    env,
                    use_gnn=self.use_gnn,
                    gnn_model=self.gnn_model,
                    goal_conditioned=self.goal_conditioned,
                )

        return env

    def _build_env_kwargs(
        self,
        index: int,
        molecules: np.ndarray,
        decodings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the kwargs dict for environment construction."""
        return dict(
            env_index=index,
            molecule_array=molecules,
            n_actions=self.n_actions,
            max_swap=self.max_swap,
            decodings=decodings,
            n_agents=self.n_agents,
            prop_list=self.prop_names,
            prop_bounds=self.prop_bounds,
            small_reward=self.small_reward,
            big_reward=self.big_reward,
            repeat_penalty=self.repeat_penalty,
            novelty_reward=self.novelty_reward,
            multiplier=self.multiplier,
            ep_length=self.max_ep_length,
            credit_window=self.credit_window,
            shared_credit_bonus=self.shared_credit,
            use_scorer=self.use_scorer,
            scorer=self.scorer,
            session=self.rest_session,
            render_mode='human',
            disable_env_checker=None,
            reward_types=self.reward_types,
            reward_args=self.reward_args,
            novelty_mode=self.novelty_mode,
            buffer_size=self.buffer_size,
            sampling_type=self.sampling_type,
            addition_strategy=self.addition_strategy,
            min_props_satisfied=self.min_props_satisfied,
            prop_types=self.prop_types,
            invalid_penalty_mult=self.invalid_penalty_mult,
            terminate_on_invalid=self.terminate_on_invalid,
        )

    def _create_modular_env(self, env_kwargs: Dict, brics_cache: Optional[Dict]):
        """Create environment using the modular (non-legacy) path."""
        # Remove keys not accepted by modular constructors
        modular_kwargs = {k: v for k, v in env_kwargs.items()
                         if k not in ['disable_env_checker']}

        if self.representation_type == 'fragment':
            from environment.fragment_env import FragmentMoleculeEnv
            modular_kwargs['use_brics'] = self.use_brics
            modular_kwargs['brics_cache'] = brics_cache
            modular_kwargs['freeze_encodings'] = self.freeze_encodings
            return FragmentMoleculeEnv(**modular_kwargs)
        elif self.representation_type == 'smiles_token':
            from environment.smiles_token_env import SmilesTokenMoleculeEnv
            smiles_config = self.cfg.get('representation', {}).get('smiles_token', {})
            # Replace fragment-specific keys with SMILES-specific ones
            modular_kwargs.pop('molecule_array', None)
            modular_kwargs.pop('n_actions', None)
            modular_kwargs.pop('max_swap', None)
            modular_kwargs.pop('decodings', None)
            modular_kwargs['smiles_list'] = smiles_config.get('smiles_list')
            modular_kwargs['vocab_path'] = smiles_config.get('vocab_path')
            modular_kwargs['max_seq_len'] = smiles_config.get('max_seq_len', 100)
            return SmilesTokenMoleculeEnv(**modular_kwargs)
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")

    def _build_brics_cache(self, decodings: Dict) -> Optional[Dict]:
        """Pre-build BRICS type-safe action cache if needed."""
        if not self.use_brics or self.use_legacy:
            return None

        from encoders.fmpo_utils.brics_utils import (
            build_brics_valid_action_cache,
            get_brics_compatibility_stats,
        )
        logger.info("[BRICS] Pre-building type-safe action cache...")
        cache = build_brics_valid_action_cache(decodings, self.max_swap)
        stats = get_brics_compatibility_stats(decodings)
        logger.info(
            f"[BRICS] Fragment library: {stats['total_fragments']} fragments, "
            f"{stats['n_unique_signatures']} unique attachment signatures"
        )
        return cache

    def _register_legacy_env(self) -> None:
        """Register the gymnasium environment for legacy path."""
        from gymnasium.envs.registration import register
        try:
            register(
                id=self.env_id,
                entry_point=self.entry_point,
                kwargs={'ep_length': self.max_ep_length},
            )
        except Exception:
            pass  # Already registered

    @property
    def action_count(self) -> int:
        """Total number of actions available to agents."""
        return self.n_actions
