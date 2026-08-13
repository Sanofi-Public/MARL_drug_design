"""
TrainingRunner — Generic MARL training orchestrator.

Encapsulates the full training lifecycle:
    1. Model initialization (IAA2C with proper observation/action spaces)
    2. Environment creation (via EnvironmentFactory)
    3. Training loop (multi-agent rollout collection + policy updates)
    4. Periodic evaluation
    5. Model saving and CSV export

Usage:
    from pipeline import TrainingRunner

    runner = TrainingRunner(cfg, train_mols, eval_mols, decodings, logger, scorer)
    runner.run()
"""

import logging
from collections import deque, OrderedDict
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import torch

from pipeline.env_factory import EnvironmentFactory
from utils.utils import split_batch
from utils.train_utils import initialize_training_params
from utils.proc_utils import (
    save_agent_smiles, rewrite_dict, add_to_df,
    save_dflist_to_csv, save_models,
)

logger = logging.getLogger(__name__)


def _make_ordering(agent_idx: int, n_agents: int) -> List[int]:
    """Create cyclic agent ordering starting from agent_idx."""
    return [(agent_idx + offset) % n_agents for offset in range(n_agents)]


class TrainingRunner:
    """
    Orchestrates the full MARL training pipeline.

    Parameters
    ----------
    cfg : dict
        Full training configuration.
    train_mols : np.ndarray
        Encoded training molecules.
    eval_mols : np.ndarray
        Encoded evaluation molecules.
    decodings : dict
        Fragment code → RDKit Mol mapping.
    logger : FileSystemLogger
        Logger instance.
    scorer : str or None
        Scorer type ('rest' or None).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        train_mols: np.ndarray,
        eval_mols: np.ndarray,
        decodings: Dict[str, Any],
        train_logger: Any,
        scorer: Any = None,
    ):
        self.cfg = cfg
        self.train_mols = train_mols
        self.eval_mols = eval_mols
        self.decodings = decodings
        self.train_logger = train_logger
        self.scorer = scorer

        # Parsed from config in _init_params()
        self.n_steps = None
        self.num_env_steps = None
        self.n_agents = None
        self.parallel_envs = None
        self.device = None
        self.recurrent = None
        self.n_actions = None
        self.max_ep = None
        self.log_interval = None
        self.num_updates = None
        self.seed = None
        self.input_dim = None
        self.actor_dim = None
        self.obs_shape = None
        self.gamma = None
        self.model = None  # IAA2C instance

        # Environments
        self.train_env = None
        self.eval_env = None
        self.rest_session = None

        # Tracking
        self.total_steps = 0
        self.completed_episodes = 0

    # ─── Public API ───────────────────────────────────────────

    def run(self) -> None:
        """Execute the full training pipeline: init → train → final eval → save."""
        self._init_params()
        self._init_environments()
        self._init_batches()
        self._training_loop()
        self._final_evaluation()
        self._save_results()
        self.train_env.close()

    # ─── Initialization ───────────────────────────────────────

    def _init_params(self) -> None:
        """Initialize training parameters and model from config."""
        (
            self.n_steps, self.num_env_steps, self.n_agents,
            self.parallel_envs, self.device, self.recurrent,
            self.n_actions, self.max_ep, self.log_interval,
            self.num_updates, self.seed, self.input_dim,
            self.actor_dim, self.model, self.obs_shape, self.gamma,
        ) = initialize_training_params(self.cfg, self.eval_mols)

        self.train_logger.info(f"Using {self.parallel_envs} parallel environments")
        self.train_logger.info(f"Each agent can perform {self.n_actions} actions")
        self.train_logger.info(f"Observation dim: {self.input_dim}")

        # Evaluation settings
        self.evaluate_model = self.cfg['algorithm']['evaluate']
        self.evaluate_every = self.cfg['algorithm']['eval_interval']
        self.eval_episodes = self.cfg['algorithm']['episodes_per_eval']

        # Per-agent DataFrames for tracking
        self.list_df = [pd.DataFrame(columns=['Step']) for _ in range(self.n_agents)]
        self.list_eval_df = [pd.DataFrame(columns=['Step']) for _ in range(self.n_agents)]

    def _init_environments(self) -> None:
        """Create train and eval parallel environments."""
        self.rest_session = requests.Session()
        env_factory = EnvironmentFactory(
            self.cfg, scorer=self.scorer, rest_session=self.rest_session,
        )
        self.train_env = env_factory.create_envs(self.train_mols, self.decodings)
        self.eval_env = env_factory.create_envs(self.eval_mols, self.decodings)

    def _init_batches(self) -> None:
        """Initialize observation batches and split functions."""
        obs = self.train_env.reset()
        self.obs_shape = (self.input_dim,)

        agent_ordering = self.train_env.env_method('get_agent_ordering')
        self.init_ordering = (
            agent_ordering[0] if isinstance(agent_ordering, list) else agent_ordering
        )

        # Place initial observations according to agent ordering
        obs_init = np.zeros(
            (self.parallel_envs, self.n_agents, self.input_dim), dtype=np.float32,
        )
        for i in range(self.parallel_envs):
            first_agent = (
                agent_ordering[i][0]
                if isinstance(agent_ordering[i], (list, tuple))
                else agent_ordering[i]
            )
            obs_init[i, first_agent, :] = obs[i]

        stacked = torch.from_numpy(obs_init).reshape(
            self.parallel_envs, self.input_dim * self.n_agents,
        ).float()

        pe = self.parallel_envs
        na = self.n_agents
        ns = self.n_steps
        idim = self.input_dim
        dev = self.device

        self.batch_obs = torch.zeros(ns + 1, pe, idim * na).to(dev)
        self.batch_obs[0, :] = stacked
        self.batch_done = torch.zeros(ns + 1, pe, na).to(dev)
        self.batch_act = torch.zeros(ns, pe, na).to(dev)
        self.batch_rew = torch.zeros(ns, pe, na).to(dev)
        self.batch_targ = torch.zeros(ns + 1, pe, na ** 2)

        if self.recurrent:
            self.batch_hiddens = torch.zeros(
                ns + 1, pe, self.actor_dim * na,
            ).to(dev)

        # Split functions for multi-agent batches
        obs_split_dim = -1
        self.split_obs = split_batch([idim] * na, dev, obs_split_dim)
        self.split_act = split_batch(na * [1], dev)
        self.split_rew = split_batch(na * [1], dev)
        self.split_done = split_batch(na * [1], dev)
        if self.recurrent:
            self.split_hiddens = split_batch([self.actor_dim] * na, dev)

    # ─── Training Loop ────────────────────────────────────────

    def _training_loop(self) -> None:
        """Run the main training loop: collect rollouts → update policy."""
        all_infos = deque(maxlen=20)
        last_log_t = 0
        last_eval = 0

        self.train_logger.info(
            "Bits swappable: {} out of {}".format(
                self.cfg['deepfmpo']['MAX_SWAP'],
                self.cfg['deepfmpo']['MAX_SWAP'] + 1,
            )
        )

        for n_update in range(1, self.num_updates + 1):
            agent_dict = {}

            for n in range(self.n_steps):
                step_data = self._collect_step(n)
                agent_dict[n] = step_data['agent_smiles']

                # Log completed episodes
                for i, info in enumerate(step_data['infos']):
                    if 'episode_reward' in info:
                        self.completed_episodes += 1
                        info['completed_episodes'] = self.completed_episodes
                        all_infos.append(info)
                        self.init_ordering = self.train_env.get_attr('reset_info')[0]
                        if self.recurrent:
                            self.batch_hiddens[n + 1, i, :].zero_()

                self.total_steps += self.parallel_envs

                # Periodic evaluation
                if (
                    self.evaluate_model
                    and self.total_steps >= last_eval + self.evaluate_every
                ):
                    self.train_logger.info(f"Evaluating at step {self.total_steps}")
                    eval_infos, eval_dict = self._evaluate()
                    self.train_logger.log_progress(
                        eval_infos, n_update, self.total_steps,
                        self.num_env_steps, None, label='Eval',
                    )
                    last_eval = self.total_steps
                    eval_d = OrderedDict(sorted(eval_dict.items()))
                    self.list_eval_df = add_to_df(
                        self.list_eval_df, eval_d, n_update, self.cfg, self.scorer,
                    )

            # Process agent trajectories for this update
            redone_dict = rewrite_dict(agent_dict)
            d = OrderedDict(sorted(redone_dict.items()))
            self.list_df = add_to_df(self.list_df, d, n_update, self.cfg, self.scorer)

            # Policy update
            loss_dict = self.model.update(
                self.split_obs(self.batch_obs),
                self.split_act(self.batch_act),
                self.split_rew(self.batch_rew),
                self.split_done(self.batch_done),
                self.split_hiddens(self.batch_hiddens) if self.recurrent else None,
            )
            loss_dict['updates'] = n_update

            # Carry over last step to next rollout
            self.batch_obs[0, :, :] = self.batch_obs[-1, :, :]
            self.batch_done[0, :] = self.batch_done[-1, :]
            if self.recurrent:
                self.batch_hiddens[0, :, :] = self.batch_hiddens[-1, :, :]

            # Periodic logging
            if self.total_steps >= last_log_t + self.log_interval and len(all_infos) > 1:
                self.train_logger.log_progress(
                    all_infos, n_update, self.total_steps,
                    self.num_env_steps, loss_dict, label='Train',
                )
                all_infos.clear()
                last_log_t = self.total_steps

    def _collect_step(self, step_n: int) -> Dict[str, Any]:
        """Collect one step of multi-agent rollout data."""
        obs = self.split_obs(self.batch_obs[step_n, :])
        hiddens = (
            self.split_hiddens(self.batch_hiddens[step_n, :, :])
            if self.recurrent
            else [None] * self.n_agents
        )

        tensor_obs = torch.stack([obs[i] for i in range(len(obs))])
        corrector_obs = torch.zeros_like(tensor_obs)

        actions_list = []
        rewards_list = []
        masks_list = []
        infos_list = []
        targets_list = []
        hiddens = list(hiddens)

        current_ordering = self.init_ordering

        for step_idx in range(self.n_agents):
            i = current_ordering[step_idx]
            next_agent_idx = current_ordering[(step_idx + 1) % self.n_agents]

            # Valid action masks
            action_masks = self.train_env.env_method('get_valid_actions')
            action_mask_tensor = torch.tensor(np.stack(action_masks), dtype=torch.bool)

            actions, new_hiddens = self.model.act(
                tensor_obs[i], hiddens[i], i,
                evaluation=False, action_mask=action_mask_tensor,
            )
            next_obs, rewards, dones, infos = self.train_env.step(actions)

            rewards = torch.tensor(rewards).view(self.parallel_envs, 1)
            dones = torch.tensor(dones).view(self.parallel_envs, 1)
            targets = [info['indicators'] for info in infos]

            next_obs = torch.stack([
                torch.from_numpy(o).float().reshape(self.obs_shape) for o in next_obs
            ])
            masks = torch.FloatTensor(
                [[0.0] if d else [1.0] for d in dones]
            ).to(self.device)

            corrector_obs[i] = tensor_obs[i]
            tensor_obs[next_agent_idx] = next_obs
            hiddens[i] = new_hiddens

            rewards_list.append(torch.Tensor(rewards))
            masks_list.append(masks)
            infos_list.append(infos)
            actions_list.append(actions)
            targets_list.append(torch.Tensor(targets))

            self.train_logger.update_action_stats([infos])

            if step_idx == self.n_agents - 1:
                self.init_ordering = _make_ordering(next_agent_idx, self.n_agents)

        # Concatenate agent data
        actions = torch.cat(actions_list, dim=1)
        rewards = torch.cat(rewards_list, dim=1)
        masks = torch.cat(masks_list, dim=1)
        targets = torch.cat(targets_list, dim=1)

        if self.recurrent:
            self.batch_hiddens[step_n + 1, :] = torch.cat(hiddens, dim=-1)

        next_tensor_obs = torch.cat(
            [tensor_obs[i, :] for i in range(self.n_agents)], dim=1,
        ).reshape(self.parallel_envs, self.input_dim * self.n_agents)
        correct_prev_obs = torch.cat(
            [corrector_obs[i, :] for i in range(self.n_agents)], dim=1,
        ).reshape(self.parallel_envs, self.input_dim * self.n_agents)

        self.batch_obs[step_n, :] = correct_prev_obs
        self.batch_obs[step_n + 1, :] = next_tensor_obs
        self.batch_act[step_n, :] = actions
        self.batch_done[step_n + 1, :] = masks
        self.batch_rew[step_n, :] = rewards
        self.batch_targ[step_n, :] = targets

        return {
            'infos': infos,
            'agent_smiles': save_agent_smiles(infos_list),
        }

    # ─── Evaluation ───────────────────────────────────────────

    def _evaluate(self):
        """Run evaluation episodes and return info dicts + agent SMILES data."""
        parallel_envs = self.eval_env.num_envs
        obs = self.eval_env.reset()
        obs_shape = (self.input_dim,)

        agent_ordering = self.eval_env.env_method('get_agent_ordering')
        init_ordering = (
            agent_ordering[0] if isinstance(agent_ordering, list) else agent_ordering
        )

        obs_init = np.zeros((parallel_envs, self.n_agents, self.input_dim), dtype=np.float32)
        for i in range(parallel_envs):
            first_agent = (
                agent_ordering[i][0]
                if isinstance(agent_ordering[i], (list, tuple))
                else agent_ordering[i]
            )
            obs_init[i, first_agent, :] = obs[i]

        tensor_obs = torch.from_numpy(obs_init).float().permute(1, 0, 2)
        hiddens = [torch.zeros(parallel_envs, self.actor_dim) for _ in range(self.n_agents)]

        all_infos = []
        eval_agent_dict = {}
        completed = 0
        step_counter = 0
        max_steps = self.eval_episodes * parallel_envs * 100

        while completed < self.eval_episodes and step_counter < max_steps:
            infos_list = []
            current_ordering = init_ordering

            for step_idx in range(self.n_agents):
                i = current_ordering[step_idx]
                next_agent_idx = current_ordering[(step_idx + 1) % self.n_agents]

                action_masks = self.eval_env.env_method('get_valid_actions')
                action_mask_tensor = torch.tensor(np.stack(action_masks), dtype=torch.bool)

                with torch.no_grad():
                    actions, new_hiddens = self.model.act(
                        tensor_obs[i], hiddens[i], i,
                        evaluation=True, action_mask=action_mask_tensor,
                    )

                next_obs, rewards, dones, infos = self.eval_env.step(actions)
                next_obs = torch.stack([
                    torch.from_numpy(o).float().reshape(obs_shape) for o in next_obs
                ])

                tensor_obs[next_agent_idx] = next_obs
                hiddens[i] = new_hiddens
                infos_list.append(infos)

                if self.train_logger is not None:
                    self.train_logger.update_action_stats([infos])

                if step_idx == self.n_agents - 1:
                    init_ordering = _make_ordering(next_agent_idx, self.n_agents)

            eval_agent_dict[step_counter] = save_agent_smiles(infos_list)
            step_counter += 1

            for i, info in enumerate(infos):
                if 'episode_reward' in info:
                    all_infos.append(info)
                    completed += 1
                    if self.recurrent:
                        for h in hiddens:
                            h[i, :].zero_()
                    new_ordering = self.eval_env.get_attr('reset_info')
                    if new_ordering and new_ordering[0]:
                        init_ordering = new_ordering[0]

        return all_infos, rewrite_dict(eval_agent_dict)

    # ─── Finalization ─────────────────────────────────────────

    def _final_evaluation(self) -> None:
        """Run final evaluation and log results."""
        eval_infos, eval_dict = self._evaluate()
        self.train_logger.log_progress(
            eval_infos, self.num_updates, self.total_steps,
            self.num_env_steps, None, label='Eval',
        )
        eval_d = OrderedDict(sorted(eval_dict.items()))
        self.list_eval_df = add_to_df(
            self.list_eval_df, eval_d, self.num_updates, self.cfg, self.scorer,
        )

    def _save_results(self) -> None:
        """Save training CSVs and model checkpoint."""
        name = self.cfg['name']
        save_dflist_to_csv(self.list_df, name)
        save_dflist_to_csv(self.list_eval_df, name + '_eval')

        model_name = (
            f"{self.num_env_steps}_{self.parallel_envs}_{self.n_steps}_"
            f"{self.max_ep}_{self.gamma}_{name}"
        )
        save_models(self.model, model_name)
        self.train_logger.info(f"Model saved as {model_name}")
