import os
import torch
import torch.nn as nn

from utils.algorithm import Algorithm
from utils.models_new import Actor, Critic, RecurrentActor, FragmentEmbedding
from utils.utils import soft_update, concat_shapes

from gymnasium.spaces import flatdim


torch.set_num_threads(1)

class IAA2C(Algorithm):
    def __init__(
        self,cfgparams,**kwargs):

        super(IAA2C,self).__init__(cfgparams)
       
        observation_space=cfgparams['observation_space']
        action_space=cfgparams['action_space']
        n_agents=cfgparams['n_agents']
        actor_dim=cfgparams['actor_dim']
        critic_dim=cfgparams['critic_dim']
        self.privileged_dim = cfgparams.get('privileged_dim', 0)

        # Fragment embedding (optional)
        frag_cfg = cfgparams.get('fragment_embedding', None)
        if frag_cfg and frag_cfg.get('use', False):
            max_frags = frag_cfg['max_frags']
            enc_bits = frag_cfg['enc_bits']
            embed_dim = frag_cfg['embed_dim']
            self.frag_embed = FragmentEmbedding(
                max_frags, enc_bits, embed_dim
            ).to(cfgparams['model_device'])
            self.mol_flat_dim = max_frags * (1 + enc_bits)
        else:
            self.frag_embed = None
            self.mol_flat_dim = 0
        
        actor_fn = RecurrentActor if cfgparams['recurrent'] else Actor
        self.actors = [
            actor_fn(
                actor_dim,
                act_space.n,
                cfgparams['actor'],
                cfgparams['activation'],
            ).to(cfgparams['model_device'])
            for obs_space, act_space in zip(observation_space, action_space)
        ]

        self.critics = [
            Critic(
                critic_dim,
                cfgparams['critic'],
                cfgparams['activation'],
            ).to(cfgparams['model_device'])
            for obs_space in observation_space
        ]
        
        self.target_critics = [
            Critic(
                critic_dim,
                cfgparams['critic'],
                cfgparams['activation'],
            ).to(cfgparams['model_device'])
            for obs_space in observation_space
        ]
        for target_critic, critic in zip(self.target_critics, self.critics):
            soft_update(target_critic, critic, 1.0)

        params = []
        for actor, critic in zip(self.actors, self.critics):
            params += list(actor.parameters())
            params += list(critic.parameters())
        if self.frag_embed is not None:
            params += list(self.frag_embed.parameters())

        
        self.optimiser = torch.optim.Adam(params, self.lr)

        self.saveables = {"optimiser": self.optimiser}
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            self.saveables[f"actor_{i+1}"] = actor
            self.saveables[f"critic_{i+1}"] = critic
        if self.frag_embed is not None:
            self.saveables["frag_embed"] = self.frag_embed

    def reinitialize_optimizer(self, lr=None, pretrained_indices=None, new_agent_lr_multiplier=5.0):
        """Reinitialize optimizer with fresh state, optionally with new learning rate.
        
        Args:
            lr: Base learning rate (if None, use self.lr)
            pretrained_indices: List of agent indices that are pre-trained.
                               If provided, new agents get lr * new_agent_lr_multiplier.
            new_agent_lr_multiplier: LR multiplier for non-pretrained agents (default: 5.0)
        """
        if lr is not None:
            self.lr = lr
        
        if pretrained_indices is not None and len(pretrained_indices) < len(self.actors):
            # Differential learning rates: new agents learn faster
            pretrained_params = []
            new_params = []
            
            for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
                if i in pretrained_indices:
                    pretrained_params += list(actor.parameters()) + list(critic.parameters())
                else:
                    new_params += list(actor.parameters()) + list(critic.parameters())
            
            if self.frag_embed is not None:
                pretrained_params += list(self.frag_embed.parameters())
            
            param_groups = [
                {'params': pretrained_params, 'lr': self.lr},
                {'params': new_params, 'lr': self.lr * new_agent_lr_multiplier}
            ]
            self.optimiser = torch.optim.Adam(param_groups)
            print(f"  Pretrained agents {pretrained_indices}: lr={self.lr}")
            print(f"  New agents: lr={self.lr * new_agent_lr_multiplier}")
        else:
            self.optimiser = torch.optim.Adam(self.parameters, self.lr)
        
        self.saveables["optimiser"] = self.optimiser

    @property
    def parameters(self):
        params = []
        for actor, critic in zip(self.actors, self.critics):
            params += list(actor.parameters())
            params += list(critic.parameters())
        if self.frag_embed is not None:
            params += list(self.frag_embed.parameters())
        return params

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path, reset_optimizer=True, new_agent_lr_multiplier=5.0):
        """
        Load model weights from checkpoint.
        
        Args:
            path: Path to checkpoint directory
            reset_optimizer: If True, reinitialize optimizer (recommended for fine-tuning)
            new_agent_lr_multiplier: LR multiplier for non-pretrained agents (default: 5.0)
        """
        import json
        
        checkpoint = torch.load(os.path.join(path, "models.pt"), weights_only=False)
        for k, v in self.saveables.items():
            if k == "optimiser" and reset_optimizer:
                # Skip loading optimizer state for fine-tuning
                continue
            if k not in checkpoint:
                print(f"{k} not found in {path}")
                continue
            v.load_state_dict(checkpoint[k].state_dict())
        
        # Sync target critics with loaded critic weights
        for target_critic, critic in zip(self.target_critics, self.critics):
            soft_update(target_critic, critic, 1.0)
        
        # Check for composite model metadata
        pretrained_indices = None
        metadata_path = os.path.join(path, "composite_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            pretrained_indices = metadata.get("pretrained_agent_indices", None)
            if pretrained_indices:
                print(f"Loaded composite model metadata: pretrained agents = {pretrained_indices}")
        
        if reset_optimizer:
            self.reinitialize_optimizer(pretrained_indices=pretrained_indices, 
                                        new_agent_lr_multiplier=new_agent_lr_multiplier)
            print(f"Optimizer reinitialized with base lr={self.lr}")

    def hidden_dims(self):
        return [actor.rnn_hidden_dim if actor.recurrent else 1 for actor in self.actors]

    def _embed_obs(self, obs):
        """Replace the mol-encoding portion of *obs* with fragment embeddings.

        If ``self.frag_embed`` is ``None`` the observation is returned unchanged.
        The mol flat portion occupies the first ``self.mol_flat_dim`` columns;
        everything after that (GNN features, indicators, etc.) is kept as-is.
        """
        if self.frag_embed is None:
            return obs
        mol_part = obs[..., :self.mol_flat_dim]
        extra = obs[..., self.mol_flat_dim:]
        embedded = self.frag_embed(mol_part)
        return torch.cat([embedded, extra], dim=-1)

    def act_tuple (self, obss,hiddenss,agent_id_tuple,evaluation=False):
        actions=[]
        hiddens=[]
        #print(agent_id_tuple)
        for i in range(len(agent_id_tuple)):
            observations=obss[i]
            agent_hiddens=hiddenss[i]
            action, hidden = self.act(observations,agent_hiddens,agent_id_tuple[i],evaluation)
            actions.append(action)
            hiddens.append(hidden)
        return actions, hiddens
    def act(self, obss, hiddenss, agent_id, evaluation=False, action_mask=None):
        """
        Choose action for agent given observation (always uses stochastic policy greedy)
        We evaluate our actions for each actor individually, since this our env works sequentially.
        This is done through the use of actor_id variable
        :param obss: observation of each agent (num_agents, parallel_envs, obs_space)
        :param hiddenss: hidden states of each agent (num_agents, parallel_envs, hidden_dim)
        :param evaluation: boolean whether action selection is for evaluation
        :param action_mask: optional boolean tensor of valid actions (parallel_envs, n_actions)
        :return: actions (num_agents, parallel_envs, 1), hiddens (num_agents, parallel_envs, hidden_dim)
        """
        
        actions = []
        hiddens = []
        greedy_evaluation = False
        
        obss = self._embed_obs(obss)
        if self.privileged_dim > 0:
            actor_obss = obss[..., :-self.privileged_dim]
        else:
            actor_obss = obss
        with torch.no_grad():
             actions, hiddens = self.actors[agent_id].act(
                 actor_obss,
                 hiddenss,
                 deterministic=evaluation if greedy_evaluation else False,
                 action_mask=action_mask,
             )
        return actions, hiddens

    def _compute_returns(self, last_obs, rew, done_mask):
        
        """
        Compute n-step returns for all agents
        :param last_obs: batch of observations at last step for each agent (n_agents) x (parallel_envs, obs_shape)
        
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks for each agent (n_agents) x (n_step + 1, parallel_envs, 1)
        """
        
        obs_shape = last_obs[0].shape[1:]

        joint_obs = torch.cat(last_obs, dim=0)
  
        with torch.no_grad():
            next_value = [
                target_critic(self._embed_obs(last_obs[i])) for i, target_critic in enumerate(self.target_critics)
            ]
        
        next_value = torch.stack(next_value)
        rew = torch.stack(rew)

        n_step=rew[0].shape[0]
        returns = [next_value]

        done_mask=list(done_mask)

        for i in range(n_step - 1, -1, -1):
            eval_done=torch.stack([t[i] for t in done_mask])

            ret = rew[:, i] + self.gamma * returns[0] * eval_done
            
            returns.insert(0, ret)
        return torch.stack(returns[:-1], dim=1)
    #add targ after done_mask if indicator targ
    def update(self, obs, act, rew, done_mask, hiddens):
        """
        Compute and execute update
        :param obs: batch of observations for each agent (n_agents) x (n_step + 1, parallel_envs, obs_shape)
        :param act: batch of actions for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks (joint for all agents) (n_step + 1, parallel_envs)

        :param hiddens: batch of hiddens for each agent (n_agents) x (n_step + 1, parallel_envs, hidden_dim)
        :return: dictionary of losses
        """
        #done_mask = done_mask.unsqueeze(-1)

        # standardise rewards
        if self.standardise_rewards:
            rew = list(rew)
            for i in range(self.n_agents):
                rew[i] = (rew[i] - rew[i].mean()) / (rew[i].std() + 1e-5)

        

        returns = self._compute_returns([o[-1] for o in obs], rew, done_mask)
        #print('Here')
        loss_dict = {}

        obs_shape = obs[0].shape[2:]
        
        self.optimiser.zero_grad()

        total_loss = 0

        for i in range(self.n_agents):
            actor = self.actors[i]
            critic = self.critics[i]
    
        
            agent_obs = self._embed_obs(obs[i][:-1])
            if self.privileged_dim > 0:
                actor_obs = agent_obs[..., :-self.privileged_dim]
            else:
                actor_obs = agent_obs
            critic_obs = agent_obs
            
            agent_act = act[i]
  
            agent_hidden = hiddens[i][:-1] if self.model_recurrent else None
        

            agent_ret = returns[i]
            
            values = critic(critic_obs)
            
            action_log_probs, entropy, _ = actor.evaluate_actions(actor_obs,agent_hidden, agent_act)

            advantages = agent_ret - values

            value_loss = advantages.pow(2).mean()
            actor_loss = -(advantages.detach() * action_log_probs).mean()

            loss = (
                actor_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
            )

            total_loss += loss
        
            loss_dict.update({
                f"agent_{i+1}/actor_loss": actor_loss.item(),
                f"agent_{i+1}/value_loss": value_loss.item(),
                f"agent_{i+1}/entropy": entropy.item(),
            })

        total_loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
            nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimiser.step()

        # update target networks
        for critic, target_critic in zip(self.critics, self.target_critics):
            soft_update(target_critic, critic, self.tau)

        return loss_dict

