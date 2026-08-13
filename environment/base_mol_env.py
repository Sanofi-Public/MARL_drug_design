#!/usr/bin/env python
# coding: utf-8

"""
Abstract base class for molecule optimization environments.

This provides the shared infrastructure for multi-agent RL molecule optimization,
independent of the molecular representation (fragments, SMILES tokens, etc.).

Subclasses must implement:
- encode_molecule: Convert RDKit Mol to internal representation
- decode_molecule: Convert internal representation to RDKit Mol  
- apply_action: Apply an action to modify the current molecule
- get_valid_actions: Return mask of valid actions for current state
- _get_observation_space: Return gymnasium observation space
- _get_action_space: Return gymnasium action space
"""

from abc import ABC, abstractmethod
import gymnasium as gym
import random
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from utils.reward_functions import (
    target_check, get_score, differentiable_prop_improvement_reward,
    discrete_action_validity, ATOM_COUNTERS
)
from utils.sampling import MoleculeBuffer, should_add_to_buffer
from itertools import permutations


class Agent:
    """
    Agent assigned to optimize a specific molecular property.
    
    Each agent tracks its own rewards, property targets, and can have
    custom reward functions (gaussian, double_sigmoid, etc.)
    """
    def __init__(self):
        self.number = None
        self.eval_prop = None
        self.rewards = 0
        self.score_bounds = None
        self.prop_targets = 0
        self.deferred_bonus = 0  # Track deferred bonuses for shared credit
        self.agent_type = None   # 'counter' or 'scorer'
        self.reward_type = "gaussian"
        self.reward_args = None

    @property
    def name(self):
        return self.eval_prop
    
    def reset(self):
        """Reset agent state for new episode."""
        self.rewards = 0
        self.prop_targets = 0
        self.deferred_bonus = 0


class BaseMoleculeEnv(gym.Env, ABC):
    """
    Abstract base class for molecule optimization environments.
    
    Provides shared infrastructure for:
    - Multi-agent turn-taking
    - Reward calculation (property improvement, novelty, repeat penalty)
    - Episode management
    - Credit assignment (sliding window + shared credit)
    - Molecule tracking (canonical SMILES for uniqueness)
    - Buffer management for exploration
    
    Subclasses implement the representation-specific logic:
    - How molecules are encoded/decoded
    - How actions modify molecules
    - What actions are valid
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_index=None,
        n_agents=1,
        prop_list=['logp'],
        prop_bounds=[[-2, 5]],
        small_reward=-0.1,
        big_reward=0.5,
        repeat_penalty=0.25,
        novelty_reward=0.25,
        multiplier=2.0,
        ep_length=30,
        use_scorer=False,
        scorer=None,
        session=None,
        render_mode='human',
        credit_window=3,
        shared_credit_bonus=0.5,
        reward_types=None,
        reward_args=None,
        novelty_mode="local",
        buffer_size=500,
        sampling_type="diversity",
        addition_strategy="stratified",
        min_props_satisfied=1,
        prop_types=None,
        invalid_penalty_mult=5.0,
        terminate_on_invalid=False,
    ):
        """
        Initialize the base molecule environment.
        
        Args:
            env_index: Index for parallel environment identification
            n_agents: Number of agents (each assigned a property)
            prop_list: List of property names to optimize
            prop_bounds: List of [min, max] bounds for each property
            small_reward: Small reward/penalty magnitude
            big_reward: Large reward magnitude (target achievement)
            repeat_penalty: Penalty for revisiting molecules
            novelty_reward: Reward for novel molecules
            multiplier: Reward multiplier for target achievement
            ep_length: Maximum episode length
            use_scorer: Whether to use external scorer
            scorer: Scorer object for property evaluation
            session: REST session for external scoring
            render_mode: Rendering mode
            credit_window: Sliding window size for credit assignment
            shared_credit_bonus: Bonus distributed to qualified agents
            reward_types: Per-agent reward function types
            reward_args: Per-agent reward function arguments
            novelty_mode: "local" (per-episode) or "global" (cross-episode)
            buffer_size: Maximum molecules in exploration buffer
            sampling_type: "fifo" or "diversity" for buffer sampling
            addition_strategy: "stratified" or "filtered" for buffer addition
            min_props_satisfied: Minimum properties satisfied for filtered strategy
            prop_types: Per-property scoring types (None=RDKit)
            invalid_penalty_mult: Multiplier for invalid action penalty
            terminate_on_invalid: If True, end episode on invalid action (single-agent mode)
        """
        super().__init__()
        
        self.env_index = env_index
        self.episode_number = 0
        self.n_agents = n_agents
        self.prop_list = prop_list
        self.prop_score_bounds = prop_bounds
        self.ep_length = ep_length
        self.small_rew = small_reward
        self.big_rew = big_reward
        self.repeat_penalty = repeat_penalty
        self.novelty_reward = novelty_reward
        self.invalid_penalty_mult = invalid_penalty_mult
        self.terminate_on_invalid = terminate_on_invalid
        self.multiplier = multiplier
        
        # Credit assignment
        self.credit_window = credit_window
        self.shared_credit_bonus = shared_credit_bonus
        self.improvement_history = []
        
        # Novelty tracking
        self.novelty_mode = novelty_mode
        self.global_target_counts = {}      # canonical SMILES -> int
        self.global_non_target_counts = {}  # canonical SMILES -> int
        
        # Buffer settings
        self.buffer_size = buffer_size
        self.sampling_type = sampling_type
        self.addition_strategy = addition_strategy
        self.min_props_satisfied = min_props_satisfied
        
        # Scoring
        self.scorer = scorer
        self.rest_session = session
        self.prop_types = prop_types
        
        # Initialize agents
        self.agents = [Agent() for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.agents[i].number = i
            self.agents[i].eval_prop = self.prop_list[i] if i < len(self.prop_list) else self.prop_list[0]
            self.agents[i].score_bounds = self.prop_score_bounds[i] if i < len(self.prop_score_bounds) else self.prop_score_bounds[0]
            if self.agents[i].eval_prop in ATOM_COUNTERS:
                self.agents[i].agent_type = 'counter'
            else:
                self.agents[i].agent_type = 'scorer'
            if reward_types is not None and i < len(reward_types) and reward_types[i] is not None:
                self.agents[i].reward_type = reward_types[i]
            if reward_args is not None and i < len(reward_args) and reward_args[i] is not None:
                self.agents[i].reward_args = reward_args[i]
        
        # Agent ordering for multi-agent
        self.all_possible_agent_orders = list(permutations(range(self.n_agents)))
        self.agent_ordering = None
        self.starting_agent = 0
        self.next_agent = None
        
        # Episode state
        self.current_step = 0
        self.curr_mol = None  # Internal representation (subclass-specific)
        self.prev_mol = None
        self.mol_made = False
        self.mol_id = None
        self.prop_indicator = [0.0] * self.n_agents
        
        # Molecule tracking (per-episode)
        self.target_dict = {}
        self.non_target_dict = {}
        self.target_count = {}
        self.non_target_count = {}
        self.local_target_smiles = {}      # canonical SMILES -> visit count
        self.local_non_target_smiles = {}  # canonical SMILES -> visit count
        
        # Buffer will be initialized by subclass
        self.buffer = None
        
        # Original MPO tracking for logging (updated each step)
        self._original_mpo_score = 0.0
        self._orig_made_stat = [False] * self.n_agents
        
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # GOAL-CONDITIONING / CURRICULUM METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def set_active_properties(self, active_mask, randomized_bounds=None):
        """Override bounds for inactive properties and optionally set new bounds for active ones.
        
        Args:
            active_mask: list of bool, length n_agents. True = real bounds, False = trivialize.
            randomized_bounds: optional list of [lower, upper] per agent.
        """
        for i in range(self.n_agents):
            if active_mask[i]:
                if randomized_bounds is not None:
                    self.agents[i].score_bounds = randomized_bounds[i]
                else:
                    self.agents[i].score_bounds = self.prop_score_bounds[i]
            else:
                self.agents[i].score_bounds = [-1e6, 1e6]

    def get_active_bounds(self):
        """Return current per-agent bounds (may be overridden by set_active_properties)."""
        return [self.agents[i].score_bounds for i in range(self.n_agents)]

    def set_terminate_on_invalid(self, value):
        """Toggle terminate_on_invalid at runtime (used by validity warmup)."""
        self.terminate_on_invalid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ═══════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def encode_molecule(self, mol) -> np.ndarray:
        """
        Convert RDKit Mol to internal representation.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Internal representation (e.g., fragment array, token sequence)
        """
        pass
    
    @abstractmethod
    def decode_molecule(self, representation) -> Chem.Mol:
        """
        Convert internal representation to RDKit Mol.
        
        Args:
            representation: Internal representation
            
        Returns:
            RDKit Mol object, or None if invalid
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: int, agent_id: int) -> tuple:
        """
        Apply an action to modify the current molecule.
        
        Args:
            action: Action index
            agent_id: ID of the agent taking the action
            
        Returns:
            Tuple of (success: bool, info: str, is_noop: bool)
            - success: Whether the action was valid
            - info: Description of what happened
            - is_noop: Whether this was a no-op action
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> np.ndarray:
        """
        Return boolean mask of valid actions for current state.
        
        Returns:
            np.ndarray: Boolean mask of shape (n_actions,)
        """
        pass
    
    @abstractmethod
    def _get_observation_space(self) -> gym.Space:
        """Return the observation space for this representation."""
        pass
    
    @abstractmethod
    def _get_action_space(self) -> gym.Space:
        """Return the action space for this representation."""
        pass
    
    @abstractmethod
    def _initialize_buffer(self, molecule_data):
        """
        Initialize the molecule buffer with starting molecules.
        
        Args:
            molecule_data: Data to initialize buffer (format depends on subclass)
        """
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SHARED METHODS - Used by all subclasses
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _make_obs(self, curr_mol):
        """Returns the current molecule observation."""
        return curr_mol
    
    def get_agent_ordering(self):
        """Return current agent ordering."""
        return self.agent_ordering
    
    def seed(self, seed=None):
        """Set random seed."""
        pass
    
    def select_next_agent(self, agent_id):
        """Get the next agent in the cyclical order."""
        return (agent_id + 1) % self.n_agents
    
    def get_qualified_agents(self):
        """Return set of agents who improved in the sliding window."""
        qualified_agents = set()
        for agent_id, improved in self.improvement_history:
            if improved:
                qualified_agents.add(agent_id)
        return qualified_agents
    
    def distribute_shared_credit(self, exclude_agent):
        """Distribute shared credit to qualified agents in the sliding window."""
        qualified_agents = self.get_qualified_agents()
        qualified_agents.discard(exclude_agent)
        
        n_qualified = len(qualified_agents)
        shared_credit_info = {}
        
        if n_qualified > 0:
            bonus_per_agent = self.shared_credit_bonus / n_qualified
            for agent_id in qualified_agents:
                self.agents[agent_id].deferred_bonus += bonus_per_agent
                shared_credit_info[agent_id] = bonus_per_agent
        
        return shared_credit_info
    
    def add_to_mol_dict(self):
        """
        Track molecule in target or non-target dictionary.
        
        Uses canonical SMILES as the primary key to handle representation aliasing
        (same molecule can have different internal representations).
        """
        if self.curr_mol is None:
            return
            
        # Get canonical SMILES for proper uniqueness tracking
        try:
            mol = self.decode_molecule(self.curr_mol)
            if mol is None:
                return
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            smiles = None
        
        if self.mol_made:
            # Canonical SMILES tracking (local and global)
            if smiles is not None:
                self.local_target_smiles[smiles] = self.local_target_smiles.get(smiles, 0) + 1
                self.global_target_counts[smiles] = self.global_target_counts.get(smiles, 0) + 1
                
                # Legacy array tracking
                if smiles not in [self._mol_to_smiles(v) for v in self.target_dict.values()]:
                    idx = len(self.target_dict)
                    self.target_dict[idx] = self.curr_mol.copy() if hasattr(self.curr_mol, 'copy') else self.curr_mol
                    self.target_count[idx] = 1
                else:
                    # Find and increment existing
                    for idx, v in self.target_dict.items():
                        if self._mol_to_smiles(v) == smiles:
                            self.target_count[idx] += 1
                            break
        else:
            # Canonical SMILES tracking (local and global)
            if smiles is not None:
                self.local_non_target_smiles[smiles] = self.local_non_target_smiles.get(smiles, 0) + 1
                self.global_non_target_counts[smiles] = self.global_non_target_counts.get(smiles, 0) + 1
                
                # Legacy array tracking
                if smiles not in [self._mol_to_smiles(v) for v in self.non_target_dict.values()]:
                    idx = len(self.non_target_dict)
                    self.non_target_dict[idx] = self.curr_mol.copy() if hasattr(self.curr_mol, 'copy') else self.curr_mol
                    self.non_target_count[idx] = 1
                else:
                    for idx, v in self.non_target_dict.items():
                        if self._mol_to_smiles(v) == smiles:
                            self.non_target_count[idx] += 1
                            break
    
    def _mol_to_smiles(self, representation):
        """Helper to convert internal representation to canonical SMILES."""
        try:
            mol = self.decode_molecule(representation)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
        return None
    
    def mol_count(self):
        """
        Return visit count for current molecule using canonical SMILES.
        
        Uses canonical SMILES as key to properly handle representation aliasing.
        """
        try:
            mol = self.decode_molecule(self.curr_mol)
            if mol is None:
                return 1
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return 1
        
        if self.mol_made:
            return self.local_target_smiles.get(smiles, 1)
        else:
            return self.local_non_target_smiles.get(smiles, 1)
    
    def mol_evaluate(self, prop_zone, prop_targets):
        """
        Evaluate repeat penalties and novelty rewards (local mode).
        
        Args:
            prop_zone: Whether agent improved its property
            prop_targets: Number of times agent reached property target
            
        Returns:
            Tuple of (repeat_penalty, novelty_reward)
        """
        repeat_penalty = 0.0
        novelty_reward = 0.0
        count = self.mol_count()

        if self.mol_made:
            if count == 1:
                n_targets = len(self.target_dict)
                base_factor = 1 + 1.0 / (1.0 + n_targets)
                factor = base_factor if prop_zone else 0
                novelty_reward = factor * self.novelty_reward
            else:
                penalty_mult = self.multiplier
                repeat_penalty = -self.repeat_penalty * penalty_mult * np.sqrt(count - 1)
        else:
            if count == 1:
                n_non_targets = len(self.non_target_dict)
                base_factor = 1.0 + 1.0 / (1.0 + n_non_targets)
                factor = base_factor if prop_zone else 0.25 * base_factor
                novelty_reward = factor * self.novelty_reward
            else:
                penalty_mult = 1.0
                repeat_penalty = -self.repeat_penalty * penalty_mult * np.sqrt(count - 1)

        return repeat_penalty, novelty_reward

    def mol_evaluate_global(self, prop_zone, prop_targets):
        """
        Evaluate novelty reward using global (cross-episode) counts.
        
        Uses canonical SMILES as keys so counts persist across episodes.
        Unified count-based decay: diminishing returns replace punishment.
        """
        repeat_penalty = 0.0
        novelty_reward = 0.0

        try:
            mol = self.decode_molecule(self.curr_mol)
            if mol is None:
                return repeat_penalty, novelty_reward
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return repeat_penalty, novelty_reward

        if self.mol_made:
            global_count = self.global_target_counts.get(smiles, 0)
            n_unique_targets = len(self.global_target_counts)
            base_factor = 1.0 + 1.0 / (1.0 + n_unique_targets)
            factor = base_factor if prop_zone else 0
            novelty_reward = factor * self.novelty_reward / (1.0 + global_count)
        else:
            global_count = self.global_non_target_counts.get(smiles, 0)
            n_unique_non_targets = len(self.global_non_target_counts)
            base_factor = 1.0 + 1.0 / (1.0 + n_unique_non_targets)
            factor = base_factor if prop_zone else 0.25 * base_factor
            novelty_reward = factor * self.novelty_reward / (1.0 + global_count)

        return repeat_penalty, novelty_reward
    
    def eval_prop_score(self, mol):
        """Evaluate property scores for a molecule."""
        return [get_score(prop, mol) for prop in self.prop_list]
    
    def _reset_episode_state(self):
        """Reset episode-level state variables."""
        self.current_step = 0
        self.prev_mol = None
        self.improvement_history = []
        
        # Reset tracking dicts
        self.target_dict = {}
        self.non_target_dict = {}
        self.target_count = {}
        self.non_target_count = {}
        self.local_target_smiles = {}
        self.local_non_target_smiles = {}
        
        # Reset agents
        for agent in self.agents:
            agent.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Subclasses should call super().reset() first, then set:
        - self.curr_mol (starting molecule representation)
        - Any representation-specific state
        """
        super().reset(seed=seed)
        self.episode_number += 1
        random.seed(seed)
        
        self._reset_episode_state()
        
        # Set agent ordering
        ordering_index = self.episode_number % len(self.all_possible_agent_orders)
        self.agent_ordering = self.all_possible_agent_orders[ordering_index]
        self.starting_agent = self.agent_ordering[0]
        
        # Subclass must set self.curr_mol and return (obs, info)
        return None, {}
    
    def step(self, action):
        """
        Execute a single environment step.
        
        This implements the shared reward logic, calling subclass methods
        for representation-specific operations.
        """
        if hasattr(action, 'item'):
            action = action.item()
        else:
            action = int(action)
        
        # Initialize reward components
        no_op_reward = 0.0
        base_reward = 0.0
        invalid_penalty = 0.0
        repeat_penalty = 0.0
        novelty_reward_val = 0.0
        target_bonus = 0.0
        deferred_bonus_collected = 0.0
        
        is_prop_improved = False
        is_noop = False
        is_invalid = False
        
        # Get current agent
        if self.current_step == 0:
            agent_id = self.starting_agent
            self.next_agent = None
        else:
            agent_id = self.next_agent
        
        current_agent = self.agents[agent_id]
        
        # Get starting SMILES
        try:
            start_mol = self.decode_molecule(self.curr_mol)
            start_smiles = Chem.MolToSmiles(start_mol, canonical=True) if start_mol else ""
        except Exception:
            start_smiles = ""
        
        # Apply action (subclass-specific)
        success, infostr, is_noop = self.apply_action(action, agent_id)
        
        # Check if THIS agent's property is trivialized (bounds ±1e6)
        agent_bounds = current_agent.score_bounds
        agent_trivialized = (agent_bounds[0] <= -1e5 and agent_bounds[1] >= 1e5)
        
        if agent_trivialized:
            # Trivialized agent: flat reward, no other components.
            # No-op is the only correct action; everything else is penalized.
            if is_noop:
                no_op_reward = self.small_rew * 2.0
            else:
                invalid_penalty = -self.small_rew * self.invalid_penalty_mult
                is_invalid = True
            self.improvement_history.append((agent_id, is_noop))
        elif is_noop:
            # No-op action handling
            deferred_bonus_collected = current_agent.deferred_bonus
            current_agent.deferred_bonus = 0
            
            if self.mol_made:
                no_op_reward = -self.small_rew
                self.improvement_history.append((agent_id, False))
            elif self.prop_indicator[current_agent.number] == 1.0:
                no_op_reward = self.small_rew
                self.improvement_history.append((agent_id, True))
            else:
                no_op_reward = 0.0
                self.improvement_history.append((agent_id, False))
        elif not success:
            # Invalid action
            is_invalid = True
            invalid_penalty = -self.small_rew * self.invalid_penalty_mult
            self.improvement_history.append((agent_id, False))
        else:
            # Valid action - evaluate property improvement
            base_reward, is_prop_improved = self._evaluate_property_rewards(agent_id)
            self.improvement_history.append((agent_id, is_prop_improved))
            
            if base_reward == -10:
                # Counter agent invalid or function error
                base_reward = 0.0
                is_invalid = True
            else:
                # Valid action - collect deferred bonus
                deferred_bonus_collected = current_agent.deferred_bonus
                current_agent.deferred_bonus = 0
        
        # Maintain sliding window size
        while len(self.improvement_history) > self.credit_window:
            self.improvement_history.pop(0)
        
        # Check if target molecule achieved
        mol = self.decode_molecule(self.curr_mol)
        made_stat, scores, mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.get_active_bounds(), mol, prop_types=self.prop_types
        )
        # Also check against original (unmodified) bounds for logging
        self._orig_made_stat, _, self._original_mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.prop_score_bounds, mol, prop_types=self.prop_types
        )
        
        shared_credit_info = {}
        skip_mol_dict = is_noop or is_invalid
        
        # Check if ALL properties are trivialized (validity warmup mode)
        all_trivialized = all(b[0] <= -1e5 and b[1] >= 1e5 for b in self.get_active_bounds())

        if all(made_stat):
            self.mol_made = True
            if not skip_mol_dict:
                self.add_to_mol_dict()
            
            if is_prop_improved and not all_trivialized and not agent_trivialized:
                target_bonus = self.multiplier * self.big_rew
            
            # Distribute shared credit (skip during warmup and for trivialized agents)
            if not all_trivialized and not agent_trivialized:
                shared_credit_info = self.distribute_shared_credit(exclude_agent=agent_id)
        else:
            self.mol_made = False
            if not skip_mol_dict:
                self.add_to_mol_dict()
        
        self.current_step += 1
        truncated = self.current_step >= self.ep_length
        
        self.prop_indicator = [float(x) for x in made_stat]
        if self.prop_indicator[current_agent.number] == 1.0:
            current_agent.prop_targets += 1
        
        # Calculate novelty/repeat rewards (skip for trivialized agents)
        if not skip_mol_dict and not agent_trivialized:
            if self.novelty_mode == "global":
                repeat_penalty, novelty_reward_val = self.mol_evaluate_global(
                    is_prop_improved, current_agent.prop_targets
                )
            else:
                repeat_penalty, novelty_reward_val = self.mol_evaluate(
                    is_prop_improved, current_agent.prop_targets
                )
        
        # Total reward
        total_reward = (
            no_op_reward + base_reward + invalid_penalty + target_bonus +
            repeat_penalty + novelty_reward_val + deferred_bonus_collected
        )
        
        current_agent.rewards += total_reward
        
        # Get ending SMILES
        try:
            end_mol = self.decode_molecule(self.curr_mol)
            end_smiles = Chem.MolToSmiles(end_mol, canonical=True) if end_mol else ""
        except Exception:
            end_smiles = ""
        
        reward = [x.rewards for x in self.agents]
        obs = self._make_obs(self.curr_mol)
        
        true_done = False
        # Terminate on: truncation, max targets reached, or invalid action with terminate_on_invalid
        terminated_on_invalid = is_invalid and self.terminate_on_invalid
        if truncated or len(self.target_dict) == 10000 or terminated_on_invalid:
            true_done = True
            if self.rest_session is not None:
                self.rest_session.close()
        
        count = self.mol_count() if not skip_mol_dict else 0
        
        # Buffer addition for exploration
        if not self.mol_made and not skip_mol_dict and self.buffer is not None:
            if should_add_to_buffer(
                self.prop_indicator, strategy=self.addition_strategy,
                min_props_satisfied=self.min_props_satisfied
            ):
                new_mol = self.curr_mol.copy() if hasattr(self.curr_mol, 'copy') else self.curr_mol
                self.buffer.try_add(new_mol, self.episode_number, self.current_step)
        
        self.next_agent = self.select_next_agent(current_agent.number)
        
        info = self._build_info_dict(
            current_agent, action, start_smiles, end_smiles, count,
            no_op_reward, base_reward, invalid_penalty, target_bonus,
            repeat_penalty, novelty_reward_val, deferred_bonus_collected,
            shared_credit_info, total_reward, is_noop, is_invalid,
            is_prop_improved, scores, mpo_score, infostr
        )
        
        return obs, total_reward, true_done, truncated, info
    
    def _build_info_dict(
        self, current_agent, action, start_smiles, end_smiles, count,
        no_op_reward, base_reward, invalid_penalty, target_bonus,
        repeat_penalty, novelty_reward, deferred_bonus_collected,
        shared_credit_info, total_reward, is_noop, is_invalid,
        is_prop_improved, scores, mpo_score, infostr
    ):
        """Build the info dictionary returned by step()."""
        return {
            'env_index': self.env_index,
            'agent_id': current_agent.number,
            'all rewards': [x.rewards for x in self.agents],
            'actions': action,
            'indicators': self.prop_indicator,
            'start smiles': start_smiles,
            'end smiles': end_smiles,
            'current_molecule_count': count,
            'unique_found': len(self.target_dict),
            # Reward components
            'no_op_reward': no_op_reward,
            'base_reward': base_reward,
            'invalid_penalty': invalid_penalty,
            'target_bonus': target_bonus,
            'repeat_penalty': repeat_penalty,
            'novelty_reward': novelty_reward,
            'deferred_bonus_collected': deferred_bonus_collected,
            'shared_credit_distributed': shared_credit_info,
            'total_reward': total_reward,
            # Action flags
            'is_noop': is_noop,
            'is_invalid': is_invalid,
            'is_prop_improved': is_prop_improved,
            # Other info
            'qualified_agents': list(self.get_qualified_agents()),
            'ep_reward': current_agent.rewards,
            'ep_step': self.current_step,
            'target': self.mol_made,
            'episode_number': self.episode_number,
            'scores': scores,
            'mpo_score': mpo_score,
            'original_mpo_score': self._original_mpo_score,
            'original_target': all(self._orig_made_stat),
            'infostr': infostr,
            'mol_id': self.mol_id
        }
    
    @abstractmethod
    def _evaluate_property_rewards(self, agent_id) -> tuple:
        """
        Evaluate reward based on property improvement.
        
        Args:
            agent_id: ID of the agent to evaluate
            
        Returns:
            Tuple of (reward: float, improved: bool)
        """
        pass
    
    def render(self):
        """Render current molecule with property scores."""
        mol = self.decode_molecule(self.curr_mol)
        if mol is None:
            print("No valid molecule to render")
            return
        
        scores = self.eval_prop_score(mol)
        props = ', '.join(f'{prop}: {scores[j]:.1f}' for j, prop in enumerate(self.prop_list))
        smiles = Chem.MolToSmiles(mol)
        print(f"Molecule: {smiles}")
        print(f"Properties: {props}")
