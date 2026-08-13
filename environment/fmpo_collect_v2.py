#!/usr/bin/env python
# coding: utf-8

""" 
FMPO Based multi-agent environment implemented in gymnasium.
Agents change molecules using fragment swaps to improve assigned chemical properties.
The implementation is cyclical - agents take turns optimizing their respective properties.
"""
import requests
import gymnasium as gym
import random
import numpy as np
from rdkit import Chem, RDLogger
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*') 
from utils.reward_functions import * 
import encoders.fmpo_utils.mol_utils as mol_utils
from itertools import permutations
from utils.sampling import MoleculeBuffer, should_add_to_buffer

class Agent:
    """
    Each agent is initialised with its assigned property, bounds, and zero rewards.
    """
    def __init__(self):
        self.eval_prop = None
        self.rewards = None
        self.score_bounds = None
        self.prop_targets = 0
        self.deferred_bonus = 0  # Track deferred bonuses for shared credit
        self.agent_type=None
        self.reward_type = "gaussian"
        self.reward_args = None

    @property
    def name(self):
        return self.eval_prop


class madfmpo_drugenv(gym.Env):
    """ 
    FMPO implementation in gymnasium format for multiple agents.
    
    Molecules are represented as binary arrays of fragments.
    Actions are fragment swaps that transform molecules.
    """   

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_index=None,
        molecule_array=[],
        n_actions=None,
        max_swap=None,
        decodings=None,
        n_agents=None,
        prop_list=['p1','p2'],
        prop_bounds=[],
        small_reward=-0.1,
        big_reward=0.5,
        repeat_penalty=0.25,
        novelty_reward=0.25,
        multiplier=2.0,
        ep_length=None,
        use_scorer=False,
        scorer=None,
        session=None,
        render_mode='human',
        credit_window=3,  
        shared_credit_bonus=0.5, 
        reward_types=None,
        reward_args=None,
        novelty_mode="local",  # "local" (per-episode) or "global" (cross-episode)
        buffer_size=500,  # Max molecules in exploration buffer
        sampling_type="diversity",  # "fifo" or "diversity"
        addition_strategy="stratified",  # "stratified" or "filtered"
        min_props_satisfied=1,  # Minimum properties satisfied for "filtered" strategy
        prop_types=None,  # Per-property scoring types (None=RDKit, non-None=REST)
        invalid_penalty_mult=5.0,  # Multiplier for invalid action penalty (default: 5x small_reward)
        terminate_on_invalid=False,  # If True, end episode on invalid action (for single-agent)
    ):
        self.env_index = env_index
        self.episode_number = 0
        
        # Each environment gets a different subset of molecules based on env_index
        # This ensures diversity across parallel environments while being reproducible
        array_length = min(len(molecule_array), 20)
        if env_index is not None:
            rng = random.Random(env_index)  # Local RNG seeded by env_index
            sampled_indices = rng.sample(range(len(molecule_array)), array_length)
        else:
            sampled_indices = random.sample(range(len(molecule_array)), array_length)
        initial_mols = np.array([molecule_array[i] for i in sampled_indices])
        
        # Molecule buffer with configurable eviction strategy
        self.buffer = MoleculeBuffer(initial_mols, max_size=buffer_size, sampling_type=sampling_type)
        self.addition_strategy = addition_strategy
        self.min_props_satisfied = min_props_satisfied
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.max_swap = max_swap
        self.decodings = decodings
        self.prop_list = prop_list
        self.prop_score_bounds = prop_bounds
        self.ep_length = ep_length
        self.small_rew = small_reward
        self.big_rew = big_reward
        self.repeat_penalty = repeat_penalty
        self.novelty_reward = novelty_reward
        self.invalid_penalty_mult = invalid_penalty_mult  # Stronger penalty for invalid actions
        self.terminate_on_invalid = terminate_on_invalid  # End episode on invalid (single-agent mode)

        self.scorer = scorer
        self.multiplier = multiplier
        
        # Credit assignment window settings
        self.credit_window = credit_window
        self.shared_credit_bonus = shared_credit_bonus
        self.improvement_history = []  # Sliding window: list of (agent_id, improved) tuples
        
        # Novelty tracking mode
        self.novelty_mode = novelty_mode  # "local" or "global"
        # Global SMILES visit counts (persist across reset; per env instance)
        self.global_target_counts = {}    # canonical SMILES -> int
        self.global_non_target_counts = {} # canonical SMILES -> int
        
        self.agents = [Agent() for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.agents[i].number = i
            self.agents[i].eval_prop = self.prop_list[i]
            self.agents[i].score_bounds = self.prop_score_bounds[i]
            self.agents[i].prop_targets = 0
            if self.agents[i].eval_prop in ATOM_COUNTERS:
                self.agents[i].agent_type = 'counter'
            else:
                self.agents[i].agent_type = 'scorer'
            if reward_types is not None and i < len(reward_types) and reward_types[i] is not None:
                self.agents[i].reward_type = reward_types[i]
            if reward_args is not None and i < len(reward_args) and reward_args[i] is not None:
                self.agents[i].reward_args = reward_args[i]
        
        self.rest_session = session
        self.prop_types = prop_types
        
        d1, d2 = self.buffer.molecule_array[0].shape
        self.observation_space = gym.spaces.MultiBinary([d2, d1])
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        self.curr_mol = None
        self.prev_mol = None
        self.mol_made = None
        self.target_dict = {}
        self.non_target_dict = {}
        self.target_count = {}
        self.non_target_count = {}
        # Canonical SMILES tracking (handles fragment aliasing - same molecule, different encodings)
        self.local_target_smiles = {}      # canonical_smiles -> visit_count (per episode)
        self.local_non_target_smiles = {}  # canonical_smiles -> visit_count (per episode)
        self.all_possible_agent_orders = list(permutations(range(self.n_agents)))
        
        # Pre-compute valid action mask to avoid invalid bit flips
        self.valid_action_cache = self._build_valid_action_cache()
      
    def _build_valid_action_cache(self):
        """Pre-compute which bit flips lead to valid fragment codes.
        
        Returns:
            dict: {binary_code_string: set of valid swap bit indices}
        """
        cache = {}
        for code in self.decodings.keys():
            valid_swaps = set()
            for bit in range(self.max_swap):
                flipped = self._flip_bit(code, bit)
                if flipped in self.decodings:
                    valid_swaps.add(bit)
            cache[code] = valid_swaps
        return cache
    
    def _flip_bit(self, code, bit_pos):
        """Flip bit at position (from right) in binary code string."""
        bits = list(code)
        idx = -(1 + bit_pos)
        bits[idx] = '1' if bits[idx] == '0' else '0'
        return ''.join(bits)
    
    def get_valid_actions(self):
        """Return boolean mask of valid actions for current molecule state.
        
        An action is valid if:
        1. It's a no-op (always valid), OR
        2. The target slot is active AND the bit flip produces an existing fragment code
        
        Returns:
            np.ndarray: Boolean mask of shape (n_actions,)
        """
        valid = np.zeros(self.n_actions, dtype=bool)
        valid[-1] = True  # No-op is always valid
        
        if self.curr_mol is None:
            return valid
        
        for frag_idx in range(self.curr_mol.shape[0]):
            if self.curr_mol[frag_idx, 0] == 1:  # Slot is active
                # Get current fragment code
                code = "".join([str(int(y)) for y in self.curr_mol[frag_idx, 1:]])
                # Get valid swaps for this code
                valid_swaps = self.valid_action_cache.get(code, set())
                for swap_bit in valid_swaps:
                    action = frag_idx * self.max_swap + swap_bit
                    valid[action] = True
        
        return valid

    def set_active_properties(self, active_mask, randomized_bounds=None):
        """Override bounds for inactive properties and optionally set new bounds for active ones.
        
        Called by the training script to implement property curriculum.
        Active properties use randomized_bounds (if provided) or their original bounds;
        inactive ones get [-1e6, 1e6] so they are always 'satisfied'.
        
        Args:
            active_mask: list of bool, length n_agents. True = real bounds, False = trivialize.
            randomized_bounds: optional list of [lower, upper] per agent. If provided,
                               active agents use these instead of original bounds.
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

    def get_agent_ordering(self):
        return self.agent_ordering

    def _get_observation_space(self):
        return self.mol
                    
    def seed(self, seed=None):
        pass
    
    def reset(self, seed=None, options=None):
        """Resets the environment with a randomly chosen lead molecule."""
        super().reset(seed=seed)
        self.episode_number += 1
        random.seed(seed) 
        index = self.buffer.sample_index()
        
        ordering_index = self.episode_number % len(self.all_possible_agent_orders)
        self.agent_ordering = self.all_possible_agent_orders[ordering_index]
        
        self.mol_id = self.buffer.name_list[index]
        self.starting_molecule = self.buffer.get(index).copy()
        self.target_dict = {}
        self.target_count = {}
        self.non_target_dict = {}
        self.non_target_count = {}
        # Reset per-episode canonical SMILES tracking
        self.local_target_smiles = {}
        self.local_non_target_smiles = {}
        self.curr_mol = self.starting_molecule.copy()
        self.prev_mol = None
        self.current_step = 0
        self.frag_removed = None
        self.frag_added = None
        self.improvement_history = []  # Reset sliding window
        self.starting_agent = self.agent_ordering[0]
        for i in range(self.n_agents):
            self.agents[i].rewards = 0
            self.agents[i].prop_targets = 0
            self.agents[i].deferred_bonus = 0  # Reset deferred bonus
        
        og_mol = self.decode(self.curr_mol, self.decodings)
        og_smiles = Chem.MolToSmiles(og_mol)

        targ_call, scores, mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list, 
            self.get_active_bounds(), self.decode(self.curr_mol, self.decodings),
            prop_types=self.prop_types
        )
        self.prop_indicator = [float(x) for x in targ_call]
        
        self.mol_made = True if all(targ_call) else False
        self.add_to_mol_dict()
        observations = self._make_obs(self.curr_mol)
        self.starting_agent = self.agent_ordering[0]
        infos = {
            'molindex': self.mol_id, 
            'og_smiles': og_smiles,
            'agent_id': 0, 
            'indicators': self.prop_indicator, 
            'unique_found': 0,
            'agent_ordering': self.agent_ordering,
            'target': self.mol_made,
            'current_molecule_count':1
        }
        return observations, infos

    def _make_obs(self, curr_mol):
        """Returns the current molecule observation."""
        return curr_mol
    
    def eval_prop_score(self, mol):
        """Evaluate property scores for current molecule."""
        return [get_score(prop, mol) for prop in self.prop_list]

    def make_fragmol(self, frag_index, swap_bit, current_agent):
        """Make a molecule from the set of fragments via swap."""
        self.prev_mol = self.curr_mol.copy()
        temp_mol = self.curr_mol.copy()
        
        # Get removed fragment encoding and convert to SMILES
        frag_removed_bits = temp_mol[frag_index].copy()[1:]
        frag_removed_key = "".join([str(int(y)) for y in frag_removed_bits])
        if frag_removed_key in self.decodings:
            self.frag_removed = Chem.MolToSmiles(Chem.Mol(self.decodings[frag_removed_key]))
        else:
            self.frag_removed = None
        
        # Apply swap
        temp_mol[frag_index] = self.modify_fragment(temp_mol[frag_index], swap_bit)
        self.curr_mol[frag_index] = temp_mol[frag_index]
        
        # Get added fragment encoding and convert to SMILES
        frag_added_bits = temp_mol[frag_index].copy()[1:]
        frag_added_key = "".join([str(int(y)) for y in frag_added_bits])
        if frag_added_key in self.decodings:
            self.frag_added = Chem.MolToSmiles(Chem.Mol(self.decodings[frag_added_key]))
        else:
            self.frag_added = None

    def evaluate_property_rewards(self, agent_id):
        """Evaluate reward based on property improvement."""
        # Skip property reward if bounds are trivialized (validity warmup)
        bounds = self.agents[agent_id].score_bounds
        if bounds[0] <= -1e5 and bounds[1] >= 1e5:
            return 0.0, True

        prev_molecule = self.decode(self.prev_mol, self.decodings)
        current_molecule = self.decode(self.curr_mol, self.decodings)
        mol_tuple = [prev_molecule, current_molecule]
        prop_index = self.prop_list.index(self.agents[agent_id].eval_prop)
        
        # Check if this is a counter agent with invalid action
        current_agent = self.agents[agent_id]
        if current_agent.agent_type == 'counter':
            if not discrete_action_validity(self.frag_removed, self.frag_added, current_agent.eval_prop):
                return -10, False
        
        # frag_removed and frag_added are already SMILES strings (or None)
        ds_kwargs = {}
        if current_agent.reward_type == "double_sigmoid" and current_agent.reward_args:
            ds_kwargs = {
                "ds_k": current_agent.reward_args.get("k", 0.0),
                "ds_k_left": current_agent.reward_args.get("k_left", 1.0),
                "ds_k_right": current_agent.reward_args.get("k_right", 1.0),
            }
        prop_reward, improved = differentiable_prop_improvement_reward(
            self.rest_session, self.scorer, self.prop_list, prop_index,
            self.agents[agent_id].score_bounds, mol_tuple, self.frag_removed, self.frag_added,
            reward_type=current_agent.reward_type, prop_types=self.prop_types, **ds_kwargs)
        
        return prop_reward, improved 

    def select_next_agent(self, agent_id):
        """Get the next agent in the cyclical order."""
        return (agent_id + 1) % self.n_agents

    def step(self, action):
        """Execute a single environment step."""
        if hasattr(action, 'item'):
            action = action.item()
        else:
            action = int(action)
        
        # Initialize reward components
        no_op_reward = 0.0
        base_reward = 0.0
        invalid_penalty = 0.0
        repeat_penalty = 0.0
        novelty_reward = 0.0
        target_bonus = 0.0
        deferred_bonus_collected = 0.0  # Initialize here, collect later if valid
        
        is_prop_improved = False
        is_noop = False
        is_invalid = False
        
        if self.current_step == 0:
            agent_id = self.starting_agent
            self.next_agent = None
        else:
            agent_id = self.next_agent
        
        current_agent = self.agents[agent_id]
        
        start_smiles = Chem.MolToSmiles(self.decode(self.curr_mol, self.decodings), canonical=True)
        
        # Check if THIS agent's property is trivialized (bounds ±1e6)
        agent_bounds = current_agent.score_bounds
        agent_trivialized = (agent_bounds[0] <= -1e5 and agent_bounds[1] >= 1e5)
        
        if agent_trivialized:
            # Trivialized agent: strongly reward no-op, penalize acting
            if action == self.n_actions - 1:
                is_noop = True
                infostr = 'No-op (trivialized agent - correct)'
                no_op_reward = self.small_rew * 2.0
                self.frag_removed = None
                self.frag_added = None
                deferred_bonus_collected = current_agent.deferred_bonus
                current_agent.deferred_bonus = 0
                self.improvement_history.append((agent_id, True))
            else:
                infostr = 'Trivialized agent acted - penalized'
                invalid_penalty = -self.small_rew * self.invalid_penalty_mult
                is_invalid = True
                self.frag_removed = None
                self.frag_added = None
                self.improvement_history.append((agent_id, False))
        elif action == self.n_actions - 1:
            # No-op action - collect deferred bonus for no-ops (they are valid actions)
            is_noop = True
            deferred_bonus_collected = current_agent.deferred_bonus
            current_agent.deferred_bonus = 0
            self.frag_removed = None
            self.frag_added = None
            
            if self.mol_made:
                infostr = 'No-op after mol is made'
                no_op_reward = -self.small_rew  # Penalize no-op after target achieved
                self.improvement_history.append((agent_id, False))
            elif self.prop_indicator[current_agent.number] == 1.0:
                infostr = 'No-op waiting'
                no_op_reward = self.small_rew  # Reward waiting in good state
                self.improvement_history.append((agent_id, True))
            else:
                infostr = 'No-op in a wrong state'
                no_op_reward = 0.0  # Neutral for no-op in wrong state
                self.improvement_history.append((agent_id, False))
        else:
            # Fragment swap action
            frag_index = int(action // self.max_swap)
            swap_bit = action % self.max_swap
            
            # Save current molecule state BEFORE any modification attempt
            saved_mol = self.curr_mol.copy()
                    
            if self.curr_mol[frag_index, 0] == 1:
                try:
                    self.make_fragmol(frag_index, swap_bit, current_agent)
                    base_reward, is_prop_improved = self.evaluate_property_rewards(current_agent.number)
                    
                    # Track improvement in sliding window
                    self.improvement_history.append((agent_id, is_prop_improved))
                    
                    if base_reward == -10:
                        infostr = 'counter invalid action' if current_agent.agent_type == 'counter' else 'function error'
                        base_reward = 0.0
                        if current_agent.agent_type == 'counter':
                            # Counter agent made swap not involving target atom
                            # Don't revert - let molecule evolve to maintain diversity
                            # But mark invalid to skip novelty rewards
                            invalid_penalty = 0.0
                            is_invalid = True  # Skip novelty/repeat rewards
                            # No reversion - molecule keeps the change
                        else:
                            # Other function errors - revert
                            invalid_penalty = -self.small_rew * self.invalid_penalty_mult
                            is_invalid = True
                            self.curr_mol = saved_mol
                    else:
                        infostr = f'Agent {current_agent.number} swapped bit {swap_bit} at fragment {frag_index}'
                        # Valid action - collect deferred bonus
                        deferred_bonus_collected = current_agent.deferred_bonus
                        current_agent.deferred_bonus = 0
                        # Add small bonus if mol was already made
                        # if self.mol_made:
                        #     base_reward += self.small_rew
                except:
                    infostr = 'invalid joining'
                    base_reward = 0.0
                    invalid_penalty = -self.small_rew * self.invalid_penalty_mult
                    # Revert to saved state (before make_fragmol was called)
                    self.curr_mol = saved_mol
                    self.improvement_history.append((agent_id, False))
                    is_invalid = True
                    # Do NOT collect deferred bonus for invalid actions
            else:
                infostr = 'Did not find fragment'
                base_reward = 0.0
                invalid_penalty = -self.small_rew * self.invalid_penalty_mult
                self.improvement_history.append((agent_id, False))
                is_invalid = True
                # Do NOT collect deferred bonus for invalid actions
        
        # Maintain sliding window size
        while len(self.improvement_history) > self.credit_window:
            self.improvement_history.pop(0)
             
        made_stat, scores, mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.get_active_bounds(), self.decode(self.curr_mol, self.decodings),
            prop_types=self.prop_types
        )
        # Also check against original (unmodified) bounds for logging
        orig_made_stat, _, original_mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.prop_score_bounds, self.decode(self.curr_mol, self.decodings),
            prop_types=self.prop_types
        )
        
        shared_credit_info = {}
        
        # Skip mol dict updates for no-ops and invalid actions
        skip_mol_dict = is_noop or is_invalid
        
        # Check if ALL properties are trivialized (validity warmup mode)
        all_trivialized = all(b[0] <= -1e5 and b[1] >= 1e5 for b in self.get_active_bounds())

        if all(made_stat):
            self.mol_made = True
            if not skip_mol_dict:
                self.add_to_mol_dict() 

            if is_prop_improved and not all_trivialized and not agent_trivialized:                       
                infostr = 'target achieved correctly'
                target_bonus = self.multiplier * self.big_rew
            else:
                target_bonus = 0.0
                infostr = 'target molecule found suboptimally' if not all_trivialized else 'validity warmup (no target bonus)'
            
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
        
        # Calculate repeat penalty and novelty reward (skip for no-ops, invalid actions, and trivialized agents)
        if not skip_mol_dict and not agent_trivialized:
            if self.novelty_mode == "global":
                repeat_penalty, novelty_reward = self.mol_evaluate_global(is_prop_improved, current_agent.prop_targets)
            else:
                repeat_penalty, novelty_reward = self.mol_evaluate(is_prop_improved, current_agent.prop_targets)
        
        # Calculate total reward as sum of all components
        total_reward = (
            no_op_reward + 
            base_reward + 
            invalid_penalty + 
            target_bonus + 
            repeat_penalty + 
            novelty_reward + 
            deferred_bonus_collected
        )
        
        current_agent.rewards += total_reward
       
        t_mol = self.decode(self.curr_mol, self.decodings)
        end_smiles = Chem.MolToSmiles(t_mol, canonical=True)
        reward = [x.rewards for x in self.agents]
        obs = self._make_obs(self.curr_mol)

        true_done = False       
        ep_num = self.episode_number
        
        # Terminate on invalid action if configured (for single-agent training)
        if is_invalid and self.terminate_on_invalid:
            true_done = True
            total_reward = 0.0  # Zero reward for invalid episode
        
        if truncated or len(self.target_dict) == 10000:
            true_done = True
            if self.rest_session is not None:
                self.rest_session.close()
        
        count = self.mol_count() if not skip_mol_dict else 0
        
        # Skip for no-ops and invalid actions since no valid molecule change occurred
        if not self.mol_made and not skip_mol_dict:
        
            if should_add_to_buffer(self.prop_indicator, strategy=self.addition_strategy,
                                    min_props_satisfied=self.min_props_satisfied):
                new_mol = self.curr_mol.copy()
                self.buffer.try_add(new_mol, self.episode_number, self.current_step)


        self.next_agent = self.select_next_agent(current_agent.number)

        info = {
            'env_index': self.env_index,
            'agent_id': current_agent.number,
            'all rewards': reward,
            'actions': action,
            'indicators': self.prop_indicator,
            'start smiles': start_smiles,
            'end smiles': end_smiles,
            'current_molecule_count': count,
            'unique_found': len(self.target_dict),
            # Reward components for tracking
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
            'frag_added': self.frag_added,
            'frag_removed': self.frag_removed,
            'ep_step': self.current_step,
            'target': self.mol_made,
            'episode_number': ep_num,
            'scores': scores,
            'mpo_score': mpo_score,
            'original_mpo_score': original_mpo_score,
            'original_target': all(orig_made_stat),
            'infostr': infostr,
            'mol_id': self.mol_id
        }

        return obs, total_reward, true_done, truncated, info

    def distribute_shared_credit(self, exclude_agent):
        """Distribute shared credit to qualified agents in the sliding window."""
        qualified_agents = self.get_qualified_agents()
        qualified_agents.discard(exclude_agent)  # Exclude current agent
        
        n_qualified = len(qualified_agents)
        shared_credit_info = {}
        
        if n_qualified > 0:
            bonus_per_agent = self.shared_credit_bonus / n_qualified
            
            for agent_id in qualified_agents:
                self.agents[agent_id].deferred_bonus += bonus_per_agent
                shared_credit_info[agent_id] = bonus_per_agent
        
        return shared_credit_info
    
    def modify_fragment(self, f, swap):
        """Flip a bit in the fragment encoding."""
        f[-(1 + swap)] = (f[-(1 + swap)] + 1) % 2
        return f

    def add_to_mol_dict(self):
        """Track molecule in target or non-target dictionary.
        
        Uses canonical SMILES as the primary key to handle fragment aliasing
        (same molecule can have different fragment encodings).
        """
        curr_mol_copy = self.curr_mol.copy()
        
        # Get canonical SMILES for proper uniqueness tracking
        try:
            smiles = Chem.MolToSmiles(self.decode(self.curr_mol, self.decodings), canonical=True)
        except Exception:
            smiles = None
        
        if self.mol_made:
            # Array-based tracking (legacy)
            found = any(np.array_equal(curr_mol_copy, v) for v in self.target_dict.values())
            if not found:
                idx = len(self.target_dict)
                self.target_dict[idx] = curr_mol_copy
                self.target_count[idx] = 1
            else:
                idx = next(k for k, v in self.target_dict.items() if np.array_equal(curr_mol_copy, v))
                self.target_count[idx] += 1
            # Canonical SMILES tracking (local and global)
            if smiles is not None:
                self.local_target_smiles[smiles] = self.local_target_smiles.get(smiles, 0) + 1
                self.global_target_counts[smiles] = self.global_target_counts.get(smiles, 0) + 1
        else:
            # Array-based tracking (legacy)
            found = any(np.array_equal(curr_mol_copy, v) for v in self.non_target_dict.values())
            if not found:
                idx = len(self.non_target_dict)
                self.non_target_dict[idx] = curr_mol_copy
                self.non_target_count[idx] = 1
            else:
                idx = next(k for k, v in self.non_target_dict.items() if np.array_equal(curr_mol_copy, v))
                self.non_target_count[idx] += 1
            # Canonical SMILES tracking (local and global)
            if smiles is not None:
                self.local_non_target_smiles[smiles] = self.local_non_target_smiles.get(smiles, 0) + 1
                self.global_non_target_counts[smiles] = self.global_non_target_counts.get(smiles, 0) + 1

    def get_qualified_agents(self):
        """Return set of agents who improved in the sliding window."""
        qualified_agents = set()
        for agent_id, improved in self.improvement_history:
            if improved:
                qualified_agents.add(agent_id)
        return qualified_agents

    def mol_count(self):
        """Return visit count for current molecule using canonical SMILES.
        
        Uses canonical SMILES as key to properly handle fragment aliasing
        (same molecule can have different fragment encodings).
        """
        try:
            smiles = Chem.MolToSmiles(self.decode(self.curr_mol, self.decodings), canonical=True)
        except Exception:
            return 1  # Default to 1 if decoding fails
        
        if self.mol_made:
            return self.local_target_smiles.get(smiles, 1)
        else:
            return self.local_non_target_smiles.get(smiles, 1)
    
    def mol_evaluate(self, prop_zone, prop_targets):
        """
        Evaluate repeat penalties and novelty rewards.
        
        - Targets: Full novelty if agent improved its property, partial otherwise
        - Non-targets: Small novelty for exploration, bonus if property improved
        - Repeat penalty: Always applies to discourage loops
        """
        repeat_penalty = 0.0
        novelty_reward = 0.0
        count = self.mol_count()

        if self.mol_made:
            if count == 1:
                n_targets = len(self.target_dict)
                #base_factor = 1.0 + 2.0 / (1.0 + 0.1 * n_targets)
                base_factor=1+ 1.0/(1.0+n_targets)
                if prop_zone:
                    #factor = self.multiplier * base_factor
                    factor=base_factor
                else:
                    #factor = 0.5 * base_factor
                    factor=0
                novelty_reward = factor * self.novelty_reward
            else:
                #penalty_mult = self.multiplier if prop_zone else self.multiplier * 1.5
                penalty_mult = self.multiplier 
                repeat_penalty = -self.repeat_penalty * penalty_mult * np.sqrt(count - 1)
        else:
            if count == 1:
                n_non_targets = len(self.non_target_dict)
                #base_factor = 1.0 + 1.0 / (1.0 + 0.05 * n_non_targets)
                base_factor=1.0 + 1.0/(1.0 + n_non_targets)
                if prop_zone:
                    #factor = 0.75 * base_factor
                    factor=base_factor
                else:
                    # Small exploration credit even without improvement
                    factor = 0.25 * base_factor
                
                novelty_reward = factor * self.novelty_reward
            else:
                penalty_mult = 1.0
                #penalty_mult = 0.3 if prop_zone else 0.5
                repeat_penalty = -self.repeat_penalty * penalty_mult * np.sqrt(count - 1)

        return repeat_penalty, novelty_reward

    def mol_evaluate_global(self, prop_zone, prop_targets):
        """
        Evaluate novelty reward using global (cross-episode) counts.
        
        Uses canonical SMILES as keys so counts persist across episodes.
        Unified count-based decay: always positive, decays with visit count.
            reward = base_factor * novelty_reward / (1 + global_count)
        No separate repeat penalty — diminishing returns replace punishment.
        """
        repeat_penalty = 0.0
        novelty_reward = 0.0

        try:
            smiles = Chem.MolToSmiles(self.decode(self.curr_mol, self.decodings), canonical=True)
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

    def decode(self, x, translation):
        """Convert array encoding to rdkit.Mol."""
        enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
        fs = [Chem.Mol(translation[e]) for e in enc]
        try:
            return mol_utils.join_fragments(fs)
        except:
            raise RuntimeError("Something went wrong when joining fragments.")
        
    def render(self):
        """Render current molecule with property scores."""
        mol = self.decode(self.curr_mol, self.decodings)
        scores = self.eval_prop_score(mol)
        props = ', '.join(f'{prop}: {scores[j]:.1f}' for j, prop in enumerate(self.prop_list))

        legend_string = f'{props}'

        return Chem.Draw.MolsToImage([mol], legends=[legend_string])