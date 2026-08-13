#!/usr/bin/env python
# coding: utf-8

"""
SMILES token-based molecule environment.

Molecules are represented as sequences of SMILES tokens.
Actions are token replacements at specific positions.
"""

import random
import json
import numpy as np
import gymnasium as gym
from rdkit import Chem

from environment.base_mol_env import BaseMoleculeEnv
from utils.reward_functions import (
    target_check, differentiable_prop_improvement_reward,
    discrete_action_validity
)
from smiles_utils.vocabulary import Vocabulary, SMILESTokenizer, create_vocabulary, START_TOKEN, STOP_TOKEN


class SmilesTokenMoleculeEnv(BaseMoleculeEnv):
    """
    SMILES token-based molecule optimization environment.
    
    Molecules are represented as 1D arrays of token indices.
    Actions are (position, new_token) pairs that replace a token.
    
    Action space: position × vocab_size + 1 (includes no-op)
    Observation: Padded token index array
    """
    
    def __init__(
        self,
        env_index=None,
        smiles_list=None,  # List of starting SMILES strings
        vocabulary=None,   # Pre-built Vocabulary object
        vocab_path=None,   # Path to saved vocabulary JSON
        max_seq_len=100,   # Maximum sequence length (with padding)
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
        Initialize the SMILES token-based molecule environment.
        
        Args:
            smiles_list: List of starting SMILES strings
            vocabulary: Pre-built Vocabulary object (or provide vocab_path)
            vocab_path: Path to saved vocabulary JSON file
            max_seq_len: Maximum sequence length including padding
            (other args inherited from BaseMoleculeEnv)
        """
        # Initialize base class
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
        
        self.max_seq_len = max_seq_len
        self.tokenizer = SMILESTokenizer()
        
        # Load or build vocabulary
        if vocabulary is not None:
            self.vocabulary = vocabulary
        elif vocab_path is not None:
            self.vocabulary = self._load_vocabulary(vocab_path)
        elif smiles_list is not None:
            self.vocabulary = create_vocabulary(smiles_list, self.tokenizer)
        else:
            raise ValueError("Must provide vocabulary, vocab_path, or smiles_list")
        
        self.vocab_size = len(self.vocabulary)
        self.pad_token = self.vocabulary.pad_token
        self.start_token = self.vocabulary[START_TOKEN]
        self.stop_token = self.vocabulary[STOP_TOKEN]
        
        # Action space: position × vocab_size + 1 (no-op)
        self.n_actions = self.max_seq_len * self.vocab_size + 1
        
        # Initialize buffer with SMILES list
        if smiles_list is not None:
            self._initialize_buffer(smiles_list)
        
        # Set observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=self.vocab_size - 1,
            shape=(self.max_seq_len,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        # Track previous SMILES for property comparison
        self.prev_smiles = None
        self.curr_smiles = None
    
    def _load_vocabulary(self, vocab_path):
        """Load vocabulary from JSON file."""
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        return Vocabulary.load_from_dictionary(vocab_dict)
    
    def _initialize_buffer(self, smiles_list):
        """Initialize the molecule buffer with starting SMILES."""
        # Sample subset for this environment instance
        array_length = min(len(smiles_list), 50)
        if self.env_index is not None:
            rng = random.Random(self.env_index)
            sampled_indices = rng.sample(range(len(smiles_list)), array_length)
        else:
            sampled_indices = random.sample(range(len(smiles_list)), array_length)
        
        self.smiles_buffer = [smiles_list[i] for i in sampled_indices]
        self.buffer = None  # We use smiles_buffer for SMILES env
    
    def _get_observation_space(self):
        """Return the observation space."""
        return self.observation_space
    
    def _get_action_space(self):
        """Return the action space."""
        return self.action_space
    
    def encode_molecule(self, mol) -> np.ndarray:
        """
        Convert RDKit Mol to token index array.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            1D array of token indices, padded to max_seq_len
        """
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return self._smiles_to_tokens(smiles)
    
    def _smiles_to_tokens(self, smiles: str) -> np.ndarray:
        """Convert SMILES string to padded token array."""
        tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=True)
        
        # Encode to indices
        indices = np.full(self.max_seq_len, self.pad_token, dtype=np.int32)
        
        for i, token in enumerate(tokens[:self.max_seq_len]):
            if token in self.vocabulary:
                indices[i] = self.vocabulary[token]
            else:
                # Unknown token - use stop token as fallback
                indices[i] = self.stop_token
        
        return indices
    
    def _tokens_to_smiles(self, token_indices: np.ndarray) -> str:
        """Convert token array to SMILES string."""
        tokens = []
        for idx in token_indices:
            idx = int(idx)
            if idx == self.pad_token:
                continue
            if idx in self.vocabulary:
                token = self.vocabulary[idx]
                tokens.append(token)
        
        return self.tokenizer.untokenize(tokens)
    
    def decode_molecule(self, representation) -> Chem.Mol:
        """Convert token array to RDKit Mol."""
        smiles = self._tokens_to_smiles(representation)
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Return boolean mask of valid actions for current state.
        
        For SMILES tokens, we allow editing any non-padding position
        with any token. Validity is checked after the edit.
        
        Returns:
            np.ndarray: Boolean mask of shape (n_actions,)
        """
        valid = np.zeros(self.n_actions, dtype=bool)
        valid[-1] = True  # No-op always valid
        
        if self.curr_mol is None:
            return valid
        
        # Find actual sequence length (non-padding positions)
        seq_len = 0
        for i, idx in enumerate(self.curr_mol):
            if idx != self.pad_token:
                seq_len = i + 1
        
        # Allow editing any position within sequence (except start/stop tokens)
        # Position 0 is START_TOKEN, so we skip it
        for pos in range(1, seq_len - 1):  # Skip start and stop tokens
            for token_id in range(self.vocab_size):
                # Skip if same as current token (no change)
                if token_id != self.curr_mol[pos]:
                    action = pos * self.vocab_size + token_id
                    if action < self.n_actions - 1:  # Leave room for no-op
                        valid[action] = True
        
        return valid
    
    def apply_action(self, action: int, agent_id: int) -> tuple:
        """
        Apply a token replacement action.
        
        Args:
            action: Action index
            agent_id: ID of the agent taking the action
            
        Returns:
            Tuple of (success: bool, info: str, is_noop: bool)
        """
        current_agent = self.agents[agent_id]
        
        if action == self.n_actions - 1:
            # No-op action
            if self.mol_made:
                infostr = 'No-op after mol is made'
            elif self.prop_indicator[current_agent.number] == 1.0:
                infostr = 'No-op waiting'
            else:
                infostr = 'No-op in wrong state'
            
            return True, infostr, True  # success, info, is_noop
        
        # Decode action to position and new token
        position = action // self.vocab_size
        new_token_id = action % self.vocab_size
        
        # Validate position
        if position >= self.max_seq_len:
            return False, 'Position out of bounds', False
        
        # Save current state for potential revert
        saved_mol = self.curr_mol.copy()
        self.prev_mol = saved_mol.copy()
        self.prev_smiles = self.curr_smiles
        
        # Check if position is in padding region
        if self.curr_mol[position] == self.pad_token:
            return False, 'Cannot edit padding position', False
        
        # Apply the edit
        old_token_id = self.curr_mol[position]
        self.curr_mol[position] = new_token_id
        
        # Try to parse the new SMILES
        new_smiles = self._tokens_to_smiles(self.curr_mol)
        new_mol = Chem.MolFromSmiles(new_smiles)
        
        if new_mol is None:
            # Invalid SMILES - revert
            self.curr_mol = saved_mol
            self.curr_smiles = self.prev_smiles
            
            old_token = self.vocabulary[old_token_id] if old_token_id in self.vocabulary else '?'
            new_token = self.vocabulary[new_token_id] if new_token_id in self.vocabulary else '?'
            return False, f'Invalid SMILES: {old_token} -> {new_token} at pos {position}', False
        
        # Valid edit
        self.curr_smiles = new_smiles
        old_token = self.vocabulary[old_token_id] if old_token_id in self.vocabulary else '?'
        new_token = self.vocabulary[new_token_id] if new_token_id in self.vocabulary else '?'
        infostr = f'Agent {current_agent.number} edited pos {position}: {old_token} -> {new_token}'
        
        return True, infostr, False
    
    def _evaluate_property_rewards(self, agent_id) -> tuple:
        """Evaluate reward based on property improvement."""
        if self.prev_smiles is None or self.curr_smiles is None:
            return 0.0, False
        
        prev_mol = Chem.MolFromSmiles(self.prev_smiles)
        curr_mol = Chem.MolFromSmiles(self.curr_smiles)
        
        if prev_mol is None or curr_mol is None:
            return 0.0, False
        
        mol_tuple = [prev_mol, curr_mol]
        prop_index = self.prop_list.index(self.agents[agent_id].eval_prop)
        
        current_agent = self.agents[agent_id]
        
        # For SMILES edits, we don't have frag_removed/frag_added
        # Counter agents work differently here
        if current_agent.agent_type == 'counter':
            # For counter agents, check if atom count changed appropriately
            # This is a simplified check - could be enhanced
            pass
        
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
        
        # For SMILES, we don't track frag_removed/frag_added
        prop_reward, improved = differentiable_prop_improvement_reward(
            self.rest_session, self.scorer, self.prop_list, prop_index,
            self.agents[agent_id].score_bounds, mol_tuple,
            None, None,  # No fragment info for SMILES
            reward_type=current_agent.reward_type,
            prop_types=self.prop_types,
            **ds_kwargs
        )
        
        return prop_reward, improved
    
    def reset(self, seed=None, options=None):
        """Reset the environment with a randomly chosen starting molecule."""
        super().reset(seed=seed)
        
        # Sample starting SMILES from buffer
        if self.smiles_buffer:
            idx = random.randint(0, len(self.smiles_buffer) - 1)
            start_smiles = self.smiles_buffer[idx]
        else:
            start_smiles = "C"  # Fallback to methane
        
        self.mol_id = idx if self.smiles_buffer else 0
        self.curr_smiles = start_smiles
        self.prev_smiles = None
        
        # Convert to tokens
        self.curr_mol = self._smiles_to_tokens(start_smiles)
        
        # Evaluate starting molecule
        og_mol = Chem.MolFromSmiles(start_smiles)
        
        targ_call, scores, mpo_score = target_check(
            self.rest_session, self.scorer, self.prop_list,
            self.prop_score_bounds, og_mol, prop_types=self.prop_types
        )
        self.prop_indicator = [float(x) for x in targ_call]
        self.mol_made = all(targ_call)
        self.add_to_mol_dict()
        
        observations = self._make_obs(self.curr_mol)
        
        infos = {
            'molindex': self.mol_id,
            'og_smiles': start_smiles,
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
        """Build info dict with SMILES-specific fields."""
        info = super()._build_info_dict(
            current_agent, action, start_smiles, end_smiles, count,
            no_op_reward, base_reward, invalid_penalty, target_bonus,
            repeat_penalty, novelty_reward, deferred_bonus_collected,
            shared_credit_info, total_reward, is_noop, is_invalid,
            is_prop_improved, scores, mpo_score, infostr
        )
        
        # Add SMILES-specific fields
        if action != self.n_actions - 1:  # Not no-op
            position = action // self.vocab_size
            new_token_id = action % self.vocab_size
            info['edit_position'] = position
            info['new_token'] = self.vocabulary[new_token_id] if new_token_id in self.vocabulary else None
        
        return info


def build_smiles_vocabulary(smiles_list, save_path=None):
    """
    Utility function to build vocabulary from a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        save_path: Optional path to save vocabulary JSON
        
    Returns:
        Vocabulary object
    """
    tokenizer = SMILESTokenizer()
    vocabulary = create_vocabulary(smiles_list, tokenizer)
    
    if save_path:
        vocab_dict = vocabulary.get_dictionary()
        with open(save_path, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        print(f"Saved vocabulary with {len(vocabulary)} tokens to {save_path}")
    
    return vocabulary
