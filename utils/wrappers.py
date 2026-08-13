from collections import deque
import itertools
import random
from time import perf_counter
from typing import Iterable

import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np
from utils.train_utils import get_outputs, load_model
import torch

from encoders.gnn_encoder import SMILES_to_Graph, GraphConv,GraphEncoder,SharedGNNEncoder

def normalize_score(score, lower, upper):
    """Normalize a property score to 0-1 based on distance to target bounds.
    
    Returns 1.0 if within bounds, decays linearly outside.
    """
    if lower <= score <= upper:
        return 1.0
    center = (lower + upper) / 2
    half_width = (upper - lower) / 2
    distance = abs(score - center) - half_width  # Distance outside bounds
    # Decay: 1.0 at boundary, 0.0 at 2x the half_width distance outside
    decay_scale = half_width if half_width > 0 else 1.0
    normalized = max(0.0, 1.0 - distance / decay_scale)
    return normalized


# Plausible value ranges per property for bound normalization.
# Each entry: property_name → (min_plausible, max_plausible)
PROPERTY_RANGES = {
    'logp':                (-3.0, 7.0),
    'molweight':           (100.0, 800.0),
    'tpsa':                (0.0, 250.0),
    'qed':                 (0.0, 1.0),
    'num_rotatable_bonds': (0.0, 15.0),
    'num_aromatic_rings':  (0.0, 6.0),
    'fluorine_count':      (0.0, 10.0),
    'sa_score':            (1.0, 10.0),
    'num_h_donors':        (0.0, 10.0),
    'num_h_acceptors':     (0.0, 15.0),
    'hERG_pIC50':        (0.0, 10.0),
    'logD_74':       (-3.0, 7.0),
    'caco2_permeability':       (-2.0, 3.0),
    'cyp3a4_pIC50':    (0.0, 10.0),
    'hlm_clearance': (0.0, 100.0),
}


def normalize_bound(value, prop_name, original_bounds_pair):
    """Normalize a bound value to [0,1] using the property's plausible range.
    
    Trivialized bounds (±1e6) are clamped to sentinel values (-1 / 2).
    """
    if value <= -1e5:
        return -1.0  # Sentinel: "no lower constraint"
    if value >= 1e5:
        return 2.0   # Sentinel: "no upper constraint"
    
    if prop_name in PROPERTY_RANGES:
        pmin, pmax = PROPERTY_RANGES[prop_name]
    else:
        pmin, pmax = original_bounds_pair[0], original_bounds_pair[1]
        span = pmax - pmin
        pmin -= span
        pmax += span
    
    span = pmax - pmin
    if span <= 0:
        return 0.5
    return (value - pmin) / span


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=10,use_gnn=False,gnn_model=None, goal_conditioned=False):
        super().__init__(env)
        self.t0 = perf_counter()
        self.n_agents=getattr(env,'n_agents')
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = getattr(env,'ep_length',0)
        
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.use_gnn=use_gnn
        self.gnn_model_path=gnn_model
        # Get property bounds for score normalization
        self.prop_bounds = getattr(env, 'prop_score_bounds', None)
        # Goal-conditioned: append normalized bounds to observation
        self.goal_conditioned = goal_conditioned
        self.prop_names = getattr(env, 'prop_list', [])
        # Track previous action for coordination signal
        self.prev_agent_id = 0.0
        self.prev_action = 0.0
        # Get n_actions for action normalization
        n_actions = getattr(env, 'n_actions', None)
        if n_actions is None:
            n_actions = env.action_space.n if hasattr(env, 'action_space') else 1
        self.n_actions = n_actions
        if self.use_gnn and self.gnn_model_path is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            node_feature_dim=10  # number of atom types in SMILES_to_Graph
            hidden_dim=128
            output_dim=10
            #model=GraphEncoder(node_feature_dim=10, hidden_dim=128,output_dim=3, num_layers=2).to(device)
            gnn_model=SharedGNNEncoder(node_feature_dim, hidden_dim, output_dim).to(device)
            self.gnn_model = load_model(gnn_model, self.gnn_model_path, device)
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.gnn_model=None

    def _get_goal_bounds_obs(self):
        """Get normalized goal bounds as a flat array [lo0, hi0, lo1, hi1, ...]."""
        if not self.goal_conditioned:
            return np.array([], dtype=np.float32)
        active_bounds = self.env.get_active_bounds()
        orig_bounds = self.prop_bounds if self.prop_bounds is not None else active_bounds
        goal_arr = []
        for i, bounds in enumerate(active_bounds):
            prop_name = self.prop_names[i] if i < len(self.prop_names) else ''
            orig_pair = orig_bounds[i] if i < len(orig_bounds) else bounds
            goal_arr.append(normalize_bound(bounds[0], prop_name, orig_pair))
            goal_arr.append(normalize_bound(bounds[1], prop_name, orig_pair))
        return np.array(goal_arr, dtype=np.float32)

    def reset(self, **kwargs):
        observations, infos = super().reset(**kwargs)
        self.episode_reward = np.zeros(self.n_agents)
        #self.action_seq = np.zeros(self.n_agents)
        self.episode_length = getattr(self.env, 'ep_length', 0)
        self.t0 = perf_counter()
        self.reset_info=infos['agent_ordering']
        # Generalize agent cycling for n_agents
        agent_id = infos['agent_id']
        
        # Reset previous action tracking for new episode
        self.prev_agent_id = 0.0
        self.prev_action = 0.0
        
        current_mol_count = infos['current_molecule_count']
        is_repeat = 1 if current_mol_count>1 else 0
        target = infos['target'] #is the molecule a target
        
        # Get raw scores and normalize them (gives gradient signal vs binary indicators)
        raw_scores = infos.get('scores', [0.0] * self.n_agents)
        if self.prop_bounds is not None and raw_scores is not None:
            norm_scores = [
                normalize_score(score, bounds[0], bounds[1])
                for score, bounds in zip(raw_scores, self.prop_bounds)
            ]
        else:
            # Fallback to binary indicators if no scores available
            norm_scores = infos['indicators']
        
        # flatten observation
        flat_obs = observations.flatten() if isinstance(observations, np.ndarray) else np.array(observations).flatten()
        scores_arr = np.array(norm_scores, dtype=np.float32).flatten()
        is_repeat_arr = np.array([is_repeat], dtype=np.float32).flatten()
        target_arr = np.array([target], dtype=np.float32).flatten()
        prev_action_arr = np.array([self.prev_agent_id, self.prev_action], dtype=np.float32)

        goal_bounds_arr = self._get_goal_bounds_obs()

        if self.use_gnn and self.gnn_model is not None:
             in_smiles=infos['og_smiles']
             out_smiles=infos['og_smiles']
             in_frags=[0]*10
             out_frags=[0]*10
             
             gnn_obs=get_outputs(in_smiles,out_smiles,in_frags,out_frags, self.gnn_model,self.device)
             combined_obs =  np.concatenate([flat_obs, gnn_obs.reshape(-1), scores_arr, is_repeat_arr, target_arr, prev_action_arr, goal_bounds_arr])
        else:
             combined_obs = np.concatenate([flat_obs, scores_arr, is_repeat_arr, target_arr, prev_action_arr, goal_bounds_arr])
        observation = combined_obs

        return observation, infos

    def step(self, action):
        observations, reward, done,truncated, info = super().step(action)
        active_agent=info['agent_id']
        self.episode_reward[active_agent]=+reward
        #self.action_seq[active_agent]=action
        #self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        #if all(done):
        if done or truncated:
            info["episode_reward"] = self.episode_reward.copy()
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0
            #info["action_seq"]=self.action_seq
            self.reward_queue.append(self.episode_reward.copy())
            self.length_queue.append(self.episode_length)

        agent_id=info['agent_id']
        current_mol_count = info['current_molecule_count']
        is_repeat = 1 if current_mol_count>1 else 0
        
        # Store current action info for next agent's observation
        prev_agent_normalized = agent_id / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0
        prev_action_normalized = info['actions'] / max(1, self.n_actions - 1) if self.n_actions > 1 else 0.0
        
        # Get raw scores and normalize them (gives gradient signal vs binary indicators)
        raw_scores = info.get('scores', [0.0] * self.n_agents)
        if self.prop_bounds is not None and raw_scores is not None:
            norm_scores = [
                normalize_score(score, bounds[0], bounds[1])
                for score, bounds in zip(raw_scores, self.prop_bounds)
            ]
        else:
            # Fallback to binary indicators if no scores available
            norm_scores = info['indicators']

        # flatten observation
        flat_obs = observations.flatten() if isinstance(observations, np.ndarray) else np.array(observations).flatten()
        scores_arr = np.array(norm_scores, dtype=np.float32).flatten()
        is_repeat_arr = np.array([is_repeat], dtype=np.float32).flatten()
        target_arr = np.array([info['target']], dtype=np.float32).flatten()
        prev_action_arr = np.array([self.prev_agent_id, self.prev_action], dtype=np.float32)
        
        goal_bounds_arr = self._get_goal_bounds_obs()

        if self.use_gnn and self.gnn_model is not None:
            in_smiles=info['start smiles']
            out_smiles=info['end smiles']
            in_frags=info['frag_removed']
            out_frags=info['frag_added']
            gnn_obs=get_outputs(in_smiles,out_smiles,in_frags,out_frags, self.gnn_model,self.device)
            combined_obs =  np.concatenate([flat_obs, gnn_obs.reshape(-1), scores_arr, is_repeat_arr, target_arr, prev_action_arr, goal_bounds_arr])
        else:
            combined_obs = np.concatenate([flat_obs, scores_arr, is_repeat_arr, target_arr, prev_action_arr, goal_bounds_arr])
        observation = combined_obs
        
        # Update prev action tracking for next step
        self.prev_agent_id = prev_agent_normalized
        self.prev_action = prev_action_normalized
            
        return observation, reward, done,truncated, info


# class RecordEpisodeStatistics(gym.Wrapper):
#     """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

#     def __init__(self, env, deque_size=10):
#         super().__init__(env)
#         self.t0 = perf_counter()
#         self.episode_reward = np.zeros(self.n_agents)
#         self.episode_length = 0
#         self.reward_queue = deque(maxlen=deque_size)
#         self.length_queue = deque(maxlen=deque_size)

#     def reset(self, **kwargs):
#         observation = super().reset(**kwargs)
#         self.episode_reward = np.zeros(self.n_agents)
#         self.episode_length = 0
#         self.t0 = perf_counter()

#         return observation

#     def step(self, action):
#         observation, reward, done, truncated,info = super().step(action)
#         self.episode_reward += np.array(reward, dtype=np.float64)
#         self.episode_length += 1
#         #print(done)
#         if done or truncated:
#             info["episode_reward"] = self.episode_reward.copy()
#             for i, agent_reward in enumerate(self.episode_reward):
#                 info[f"agent{i}/episode_reward"] = agent_reward
#             info["episode_length"] = self.episode_length
#             info["episode_time"] = perf_counter() - self.t0

#             self.reward_queue.append(self.episode_reward.copy())
#             self.length_queue.append(self.episode_length)
#         return observation, reward, done,truncated,info


class ConcatDictObservation(ObservationWrapper):
    """
    Wrapper which concatenates flattened image and feature vector of agents
    """
    def __init__(self, env):
        super(ConcatDictObservation, self).__init__(env)

        assert all([isinstance(obs_space, spaces.Dict) for obs_space in self.observation_space])
        assert all(["image" in obs_space.spaces and "features" in obs_space.spaces for obs_space in self.observation_space])
        self.image_spaces = [obs_space["image"] for obs_space in self.observation_space]

        ma_spaces = []
        for sa_obs in env.observation_space:
            flat_image_dim = spaces.flatdim(sa_obs["image"])
            features_dim = spaces.flatdim(sa_obs["features"])
            flatdim = flat_image_dim + features_dim

            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                np.concatenate(
                    [
                        spaces.flatten(image_space, obs["image"]),
                        obs["features"],
                    ]
                )
                for image_space, obs in zip(self.image_spaces, observation)
            ]
        )


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []
        d2=9
        d1=9
        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            # ma_spaces += [
            #     spaces.Box(
            #         low=-float("inf"),
            #         high=float("inf"),
            #         shape=(flatdim,),
            #         dtype=np.float32,
            #     )
            # ]
            ma_spaces+=[
                        gym.spaces.MultiBinary([d2,d1])
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, info = self.env.step(action)
        return observation, info

# class TimeLimit(gym.wrappers.TimeLimit):
#     def __init__(self, env, max_episode_steps=None):
#         super().__init__(env)
#         if max_episode_steps is None and self.env.spec is not None:
#             max_episode_steps = env.spec.max_episode_steps
#         # if self.env.spec is not None:
#         #     self.env.spec.max_episode_steps = max_episode_steps
#         self._max_episode_steps = max_episode_steps
#         self._elapsed_steps = None

#     def step(self, action):
#         assert (
#             self._elapsed_steps is not None
#         ), "Cannot call env.step() before calling reset()"
#         observation, reward, done, info = self.env.step(action)
#         self._elapsed_steps += 1
#         if self._elapsed_steps >= self._max_episode_steps:
#             info["TimeLimit.truncated"] = not all(done)
#             done = len(observation) * [True]
#         return observation, reward, done, info
        
class TimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env,max_episode_steps)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        #print('Is timelimit before or after epistats', self._max_episode_steps)
    def step(self, action):
     
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done,truncated,info = self.env.step(action)
        #print(observation,self._elapsed_steps,done,info)
        self._elapsed_steps += 1
        done=False
        if self._elapsed_steps >= self._max_episode_steps:
            #truncated=True
            #info["TimeLimit.truncated"] = not done
            #info["elapsed_steps"]=self._elapsed_steps
            #done = len(observation) * [True]
            done=True

        return observation, reward, done,truncated, info


class GeneralisationWrapper(gym.Wrapper):
    """
    General wrapper to create random training task out of given distribution on each reset
    """
    def __init__(self, env, base_name, log=False, **kwargs):
        super().__init__(env)
        if isinstance(base_name, str):
            base_name = [base_name]
        self.base_names = base_name
        self.arguments = kwargs
        self.log = log

        # get all argument combinations in sequence
        argument_values = [v if isinstance(v, Iterable) else [v] for v in kwargs.values()]
        self.arguments_list = []
        for argument_tuple in itertools.product(*argument_values):
            self.arguments_list.append(
                {k: v for k, v in zip(kwargs.keys(), argument_tuple)}
            )

    def __len__(self):
        return len(self.arguments_list) * len(self.base_names)
    
    def domain_randomisation(self):
        """
        Start new environment with random arguments
        """
        args = random.choice(self.arguments_list)
        name = random.choice(self.base_names)
        #print(name,args)
        self.env = gym.make(name, **args)

    def get_all_envs(self):
        """
        Get list of all possible environments
        """
        envs = []
        for name in self.base_names:
            for args in self.arguments_list:
                envs.append(gym.make(name, **args))
        return envs
    
    def reset(self,seed,**kwargs):
        self.domain_randomisation()
        obss = self.env.reset(seed=seed,**kwargs)
        return obss
