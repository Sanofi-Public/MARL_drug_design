#!/usr/bin/env python
# coding: utf-8

"""
Fragment-based molecule environment.

Molecules are represented as binary arrays where each row is a fragment slot
with a presence bit and binary-encoded fragment ID.

Actions are bit flips that swap one fragment for another.
"""

import random
import numpy as np
import gymnasium as gym
from rdkit import Chem

from environment.base_mol_env import BaseMoleculeEnv
from encoders.fmpo_utils.brics_utils import build_brics_valid_action_cache, get_brics_compatibility_stats
from utils.reward_functions import (
    target_check, differentiable_prop_improvement_reward,
    discrete_action_validity
)
from utils.sampling import MoleculeBuffer
import encoders.fmpo_utils.mol_utils as mol_utils


class FragmentMoleculeEnv(BaseMoleculeEnv):
    """
    Fragment-based molecule optimization environment.
    
    Molecules are represented as 2D binary arrays:
    - Each row is a fragment slot
    - First column indicates if slot is active (1) or empty (0)
    - Remaining columns are the binary-encoded fragment ID
    
    Actions are (frag_index, swap_bit) pairs that flip a bit in a fragment's
    encoding, effectively swapping it for a different fragment.
    """
    
    def __init__(
        self,
        env_index=None,
        molecule_array=[],
        n_actions=None,
        max_swap=None,
        decodings=None,
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
        use_brics=False,
        brics_cache=None,
        freeze_encodings=None,
    ):
        """
        Initialize the fragment-based molecule environment.
        
        Args:
            molecule_array: Array of starting molecule representations
            n_actions: Total number of actions (max_frag * max_swap + 1)
            max_swap: Number of bits that can be swapped per fragment
            decodings: Dict mapping binary codes to RDKit Mol fragments
            use_brics: If True, use BRICS type-safe fragment matching
            brics_cache: Pre-built BRICS action cache (avoids recomputing per env)
            (other args inherited from BaseMoleculeEnv)
        """
        # Initialize base class first
        super().__init__(
            env_index=env_index,
            n_agents=n_agents,
            prop_list=prop_list,
            prop_bounds=prop_bounds,
            small_reward=small_reward,
            big_reward=big_reward,
            repeat_penalty=repeat_penalty,
            novelty_reward=novelty_reward,
            multiplier=multiplier,
            ep_length=ep_length,
            use_scorer=use_scorer,
            scorer=scorer,
            session=session,
            render_mode=render_mode,
            credit_window=credit_window,
            shared_credit_bonus=shared_credit_bonus,
            reward_types=reward_types,
            reward_args=reward_args,
            novelty_mode=novelty_mode,
            buffer_size=buffer_size,
            sampling_type=sampling_type,
            addition_strategy=addition_strategy,
            min_props_satisfied=min_props_satisfied,
            prop_types=prop_types,
            invalid_penalty_mult=invalid_penalty_mult,
            terminate_on_invalid=terminate_on_invalid,
        )
        
        # Fragment-specific attributes
        self.n_actions = n_actions
        self.max_swap = max_swap
        self.decodings = decodings
        self.use_brics = use_brics
        self._brics_cache = brics_cache  # Pre-built cache (shared across envs)
        self.freeze_encodings = set(freeze_encodings) if freeze_encodings else set()
        
        # Initialize buffer with molecule array
        self._initialize_buffer(molecule_array)
        
        # Set observation and action spaces
        d1, d2 = self.buffer.molecule_array[0].shape
        self.observation_space = gym.spaces.MultiBinary([d2, d1])
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        # Fragment tracking
        self.frag_removed = None
        self.frag_added = None
        
        # Pre-compute valid action mask
        self.valid_action_cache = self._build_valid_action_cache()
    
    def _initialize_buffer(self, molecule_array):
        """Initialize the molecule buffer with starting molecules."""
        array_length = min(len(molecule_array), 20)
        if self.env_index is not None:
            rng = random.Random(self.env_index)
            sampled_indices = rng.sample(range(len(molecule_array)), array_length)
        else:
            sampled_indices = random.sample(range(len(molecule_array)), array_length)
        
        initial_mols = np.array([molecule_array[i] for i in sampled_indices])
        self.buffer = MoleculeBuffer(
            initial_mols, 
            max_size=self.buffer_size, 
            sampling_type=self.sampling_type
        )
    
    def _get_observation_space(self):
        """Return the observation space."""
        return self.observation_space
    
    def _get_action_space(self):
        """Return the action space."""
        return self.action_space
    
    def _build_valid_action_cache(self):
        """
        Pre-compute which bit flips lead to valid fragment codes.
        
        If use_brics is enabled, also checks BRICS attachment type compatibility
        to ensure fragments can be safely joined.
        
        Returns:
            dict: {binary_code_string: set of valid swap bit indices}
        """
        if self.use_brics:
            # Use pre-built cache if available (shared across all envs)
            if self._brics_cache is not None:
                return self._brics_cache
            
            # Otherwise build it (fallback, shouldn't happen in normal usage)
            cache = build_brics_valid_action_cache(self.decodings, self.max_swap)
            
            # Log stats if env_index is 0 (only once)
            if self.env_index == 0:
                stats = get_brics_compatibility_stats(self.decodings)
                total = len(self.decodings)
                n_sigs = stats['n_unique_signatures']
                print(f"[BRICS] Fragment library: {total} fragments, {n_sigs} unique attachment signatures")
                print(f"[BRICS] Type distribution: {stats['type_counts']}")
            
            return cache
        else:
            # Standard validity check (only checks if code exists)
            cache = {}
            for code in self.decodings.keys():
                valid_swaps = set()
                for bit in range(self.max_swap):
                    flipped = self._flip_bit_code(code, bit)
                    if flipped in self.decodings:
                        valid_swaps.add(bit)
                cache[code] = valid_swaps
            return cache
    
    def _flip_bit_code(self, code, bit_pos):
        """Flip bit at position (from right) in binary code string."""
        bits = list(code)
        idx = -(1 + bit_pos)
        bits[idx] = '1' if bits[idx] == '0' else '0'
        return ''.join(bits)
    
    def encode_molecule(self, mol) -> np.ndarray:
        """
        Convert RDKit Mol to fragment array representation.
        
        Note: This is typically done externally via build_encoding.py
        during data preparation, not at runtime.
        """
        # This would require the reverse of decode_molecule
        # For now, we expect molecules to already be in array format
        raise NotImplementedError(
            "Direct molecule encoding not implemented. "
            "Use build_encoding.py for data preparation."
        )
    
    def decode_molecule(self, representation) -> Chem.Mol:
        """Convert fragment array to RDKit Mol."""
        return self._decode(representation, self.decodings)
    
    def _decode(self, x, translation):
        """
        Convert array encoding to rdkit.Mol.
        
        Automatically detects whether fragments use:
        - BRICS-typed attachments ([n*])
        - REINVENT-style attachments ([*:n])
        - DeepFMPO-style rare element attachments ([Yb], [Lu], etc.)
        and uses the appropriate joining method.
        """
        enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
        
        if not enc:
            return None
        
        # Get fragments from translation table
        try:
            fs = [Chem.Mol(translation[e]) for e in enc]
        except (KeyError, TypeError) as e:
            return None
        
        if not fs:
            return None
        
        # Detect attachment type from first fragment
        first_frag = fs[0]
        first_smi = Chem.MolToSmiles(first_frag)
        
        # Check for different attachment types
        import re
        brics_pattern = re.compile(r'\[\d+\*\]')  # BRICS: [1*], [3*], etc.
        reinvent_pattern = re.compile(r'\[\*:\d+\]')  # REINVENT: [*:0], [*:1], etc.
        
        is_brics = bool(brics_pattern.search(first_smi))
        is_reinvent = bool(reinvent_pattern.search(first_smi))
        
        try:
            if is_reinvent:
                # Use REINVENT joining with numbered attachment matching
                from encoders.fmpo_utils.mol_utils_reinvent import join_fragments_reinvent
                return join_fragments_reinvent(fs)
            elif is_brics:
                # Use BRICS joining with RDKit's native BRICSBuild
                from encoders.fmpo_utils.mol_utils_brics import join_fragments_brics
                return join_fragments_brics(fs)
            else:
                # Use DeepFMPO joining for rare element attachments
                return mol_utils.join_fragments(fs)
        except:
            return None
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Return boolean mask of valid actions for current state.
        
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
                # Skip frozen fragments — no actions allowed on them
                if code in self.freeze_encodings:
                    continue
                # Get valid swaps for this code
                valid_swaps = self.valid_action_cache.get(code, set())
                for swap_bit in valid_swaps:
                    action = frag_idx * self.max_swap + swap_bit
                    valid[action] = True
        
        return valid
    
    def apply_action(self, action: int, agent_id: int) -> tuple:
        """
        Apply a fragment swap action.
        
        Args:
            action: Action index
            agent_id: ID of the agent taking the action
            
        Returns:
            Tuple of (success: bool, info: str, is_noop: bool)
        """
        current_agent = self.agents[agent_id]
        
        if action == self.n_actions - 1:
            # No-op action
            self.frag_removed = None
            self.frag_added = None
            
            if self.mol_made:
                infostr = 'No-op after mol is made'
            elif self.prop_indicator[current_agent.number] == 1.0:
                infostr = 'No-op waiting'
            else:
                infostr = 'No-op in wrong state'
            
            return True, infostr, True  # success, info, is_noop
        
        # Fragment swap action
        frag_index = int(action // self.max_swap)
        swap_bit = action % self.max_swap
        
        # Save current state for potential revert
        saved_mol = self.curr_mol.copy()
        
        if self.curr_mol[frag_index, 0] != 1:
            # Tried to modify empty slot
            self.frag_removed = None
            self.frag_added = None
            return False, 'Did not find fragment', False
        
        try:
            # Apply the fragment swap
            self._make_fragmol(frag_index, swap_bit, current_agent)
            
            # Try to decode the new molecule
            new_mol = self.decode_molecule(self.curr_mol)
            if new_mol is None:
                # Invalid molecule - revert
                self.curr_mol = saved_mol
                return False, 'invalid joining', False
            
            infostr = f'Agent {current_agent.number} swapped bit {swap_bit} at fragment {frag_index}'
            return True, infostr, False
            
        except Exception:
            # Error during swap - revert
            self.curr_mol = saved_mol
            self.frag_removed = None
            self.frag_added = None
            return False, 'invalid joining', False
    
    def _make_fragmol(self, frag_index, swap_bit, current_agent):
        """Apply fragment swap and track removed/added fragments."""
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
        temp_mol[frag_index] = self._modify_fragment(temp_mol[frag_index], swap_bit)
        self.curr_mol[frag_index] = temp_mol[frag_index]
        
        # Get added fragment encoding and convert to SMILES
        frag_added_bits = temp_mol[frag_index].copy()[1:]
        frag_added_key = "".join([str(int(y)) for y in frag_added_bits])
        if frag_added_key in self.decodings:
            self.frag_added = Chem.MolToSmiles(Chem.Mol(self.decodings[frag_added_key]))
        else:
            self.frag_added = None
    
    def _modify_fragment(self, f, swap):
        """Flip a bit in the fragment encoding."""
        f[-(1 + swap)] = (f[-(1 + swap)] + 1) % 2
        return f
    
    def _evaluate_property_rewards(self, agent_id) -> tuple:
        """Evaluate reward based on property improvement."""
        if self.prev_mol is None:
            return 0.0, False
        
        # Skip property reward if bounds are trivialized (validity warmup)
        bounds = self.agents[agent_id].score_bounds
        if bounds[0] <= -1e5 and bounds[1] >= 1e5:
            return 0.0, True

        prev_molecule = self.decode_molecule(self.prev_mol)
        current_molecule = self.decode_molecule(self.curr_mol)
        
        if prev_molecule is None or current_molecule is None:
            return 0.0, False
        
        mol_tuple = [prev_molecule, current_molecule]
        prop_index = self.prop_list.index(self.agents[agent_id].eval_prop)
        
        current_agent = self.agents[agent_id]
        
        # Check if this is a counter agent with invalid action
        if current_agent.agent_type == 'counter':
            if not discrete_action_validity(self.frag_removed, self.frag_added, current_agent.eval_prop):
                return -10, False
        
        # Build reward function kwargs
        ds_kwargs = {}
        if current_agent.reward_type == "double_sigmoid" and current_agent.reward_args:
            ds_kwargs = {
                "ds_k": current_agent.reward_args.get("k", 0.0),
                "ds_k_left": current_agent.reward_args.get("k_left", 1.0),
                "ds_k_right": current_agent.reward_args.get("k_right", 1.0),
            }
        elif current_agent.reward_type == "gaussian" and current_agent.reward_args:
            ds_kwargs = {
                "gaussian_center": current_agent.reward_args.get("target", None),
                "gaussian_sigma": current_agent.reward_args.get("sigma", None),
            }
        
        prop_reward, improved = differentiable_prop_improvement_reward(
            self.rest_session, self.scorer, self.prop_list, prop_index,
            self.agents[agent_id].score_bounds, mol_tuple, 
            self.frag_removed, self.frag_added,
            reward_type=current_agent.reward_type, 
            prop_types=self.prop_types, 
            **ds_kwargs
        )
        
        return prop_reward, improved
    
    def reset(self, seed=None, options=None):
        """Reset the environment with a randomly chosen lead molecule."""
        super().reset(seed=seed)
        
        # Sample starting molecule from buffer
        index = self.buffer.sample_index()
        self.mol_id = self.buffer.name_list[index]
        self.starting_molecule = self.buffer.get(index).copy()
        self.curr_mol = self.starting_molecule.copy()
        
        # Reset fragment tracking
        self.frag_removed = None
        self.frag_added = None
        
        # Evaluate starting molecule
        og_mol = self.decode_molecule(self.curr_mol)
        og_smiles = Chem.MolToSmiles(og_mol) if og_mol else ""
        
        targ_call, scores, mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.get_active_bounds(), og_mol, prop_types=self.prop_types
        )
        self.prop_indicator = [float(x) for x in targ_call]
        self.mol_made = all(targ_call)
        self.add_to_mol_dict()
        
        observations = self._make_obs(self.curr_mol)
        
        infos = {
            'molindex': self.mol_id,
            'og_smiles': og_smiles,
            'agent_id': 0,
            'indicators': self.prop_indicator,
            'unique_found': 0,
            'agent_ordering': self.agent_ordering,
            'target': self.mol_made,
            'current_molecule_count': 1
        }
        
        # Store reset info for external access
        self.reset_info = self.agent_ordering
        
        return observations, infos
    
    def _build_info_dict(
        self, current_agent, action, start_smiles, end_smiles, count,
        no_op_reward, base_reward, invalid_penalty, target_bonus,
        repeat_penalty, novelty_reward, deferred_bonus_collected,
        shared_credit_info, total_reward, is_noop, is_invalid,
        is_prop_improved, scores, mpo_score, infostr
    ):
        """Build info dict with fragment-specific fields."""
        info = super()._build_info_dict(
            current_agent, action, start_smiles, end_smiles, count,
            no_op_reward, base_reward, invalid_penalty, target_bonus,
            repeat_penalty, novelty_reward, deferred_bonus_collected,
            shared_credit_info, total_reward, is_noop, is_invalid,
            is_prop_improved, scores, mpo_score, infostr
        )
        
        # Add fragment-specific fields
        info['frag_added'] = self.frag_added
        info['frag_removed'] = self.frag_removed
        
        return info
