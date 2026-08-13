import numpy as np
import torch
from torch._C import Value
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m


class FragmentEmbedding(nn.Module):
    """Convert flat binary fragment observation into learned embeddings.

    The flat mol observation has shape ``(batch, max_frags * (1 + enc_bits))``.
    Each fragment occupies ``1 + enc_bits`` floats: an *exists* flag followed
    by ``enc_bits`` binary digits that encode the fragment identity.

    This module:
      1. Reshapes to ``(batch, max_frags, 1 + enc_bits)``
      2. Converts the binary bits to an integer token index
      3. Looks up the token in a learnable ``nn.Embedding``
      4. Multiplies each embedding by its *exists* flag (zero for empty slots)
      5. Flattens back to ``(batch, max_frags * embed_dim)``

    The caller is responsible for concatenating any extra features (GNN,
    indicators, etc.) that follow the mol portion of the observation.

    Parameters
    ----------
    max_frags : int
        Maximum number of fragment slots (e.g. 9).
    enc_bits : int
        Number of binary encoding bits per fragment (e.g. 8 → 256 tokens).
    embed_dim : int
        Dimensionality of each fragment embedding vector.
    """

    def __init__(self, max_frags, enc_bits, embed_dim):
        super().__init__()
        self.max_frags = max_frags
        self.enc_bits = enc_bits
        self.embed_dim = embed_dim
        self.frag_width = 1 + enc_bits          # exists flag + bits
        self.mol_flat_dim = max_frags * self.frag_width
        n_vocab = 2 ** enc_bits + 1             # +1 for "empty" token
        self.empty_token = n_vocab - 1
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # Pre-compute powers of 2 for binary→int conversion (not a parameter)
        powers = 2 ** torch.arange(enc_bits - 1, -1, -1, dtype=torch.long)
        self.register_buffer("powers", powers)

    @property
    def output_dim(self):
        """Flat output size: ``max_frags * embed_dim``."""
        return self.max_frags * self.embed_dim

    def forward(self, flat_mol_obs):
        """
        Parameters
        ----------
        flat_mol_obs : Tensor  (..., mol_flat_dim)
            The molecule-encoding portion of the observation (no extras).

        Returns
        -------
        Tensor  (..., max_frags * embed_dim)
        """
        leading = flat_mol_obs.shape[:-1]
        x = flat_mol_obs.reshape(*leading, self.max_frags, self.frag_width)
        exists = x[..., 0:1]                            # (..., max_frags, 1)
        bits = x[..., 1:].long()                         # (..., max_frags, enc_bits)
        token_ids = (bits * self.powers).sum(dim=-1)     # (..., max_frags)
        # Empty slots get the dedicated empty token
        token_ids = torch.where(exists.squeeze(-1) > 0.5, token_ids,
                                torch.full_like(token_ids, self.empty_token))
        emb = self.embedding(token_ids)                  # (..., max_frags, embed_dim)
        emb = emb * exists                               # zero out empty slots
        return emb.reshape(*leading, self.output_dim)

def build_sequential(num_inputs, hiddens, activation="relu", output_activation=False):
    modules = []

    if activation == "relu":
        nonlin = nn.ReLU
    elif activation == "tanh":
        nonlin = nn.Tanh
    else:
        raise ValueError(f"Unknown activation option {activation}!")
    
    assert len(hiddens) > 0
    modules.append(_init_layer(nn.Linear(num_inputs, hiddens[0])))
    for i in range(len(hiddens) - 1):
        modules.append(nonlin())
        modules.append(_init_layer(nn.Linear(hiddens[i], hiddens[i + 1])))
    if output_activation:
        modules.append(nonlin())
    return nn.Sequential(*modules)


class Actor(nn.Module):
    def __init__(self, actor_input_dim, actor_output_dim, actor_hiddens, activation):
        
        super(Actor, self).__init__()
        #print(type(actor_input_dim), task_emb_dim, type(actor_output_dim), type(actor_hiddens), activation)
        input_dim = actor_input_dim
        #another temp change making actor_hiddens to a list in call 
        #to build sequential need to figure this one in hydra yaml configs.
        
        self.num_outputs = actor_output_dim
        self.actor = build_sequential(input_dim, [actor_hiddens] + [actor_output_dim], activation)
        self.recurrent = False

    def forward(self, inputs):
        raise NotImplementedError
    
    def init_hidden(self, batch_size=1):
        return None
    
    def _get_dist(self, actor_features, action_mask=None):
        """Create categorical distribution, optionally masking invalid actions.
        
        Args:
            actor_features: logits tensor of shape (..., num_actions)
            action_mask: optional boolean tensor where True = valid action
        """
        if action_mask is not None:
            # Apply mask: set invalid action logits to large negative value
            masked_logits = actor_features.clone()
            masked_logits[~action_mask] = float('-inf')
            return Categorical(logits=masked_logits)
        return Categorical(logits=actor_features)

    def act(self, inputs, hiddens, deterministic=False, action_mask=None):
        """Select action from policy.
        
        Args:
            inputs: observation tensor
            hiddens: hidden state (unused for non-recurrent)
            deterministic: if True, select mode; else sample
            action_mask: optional boolean tensor of valid actions
        """
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features, action_mask)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.unsqueeze(-1), hiddens

    def evaluate_actions(self, inputs, hiddens, action, action_mask=None):
        """Evaluate log probability of actions.
        
        Args:
            inputs: observation tensor
            hiddens: hidden state (unused for non-recurrent)
            action: actions to evaluate
            action_mask: optional boolean tensor of valid actions
        """
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features, action_mask)

        action_log_probs = dist.log_prob(action.squeeze()).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy, hiddens
    
    def evaluate_policy_distribution(self, inputs,  hiddens):
        
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features)

        policy_probs = []
        for a in range(self.num_outputs):
            actions = torch.ones(inputs.shape[0],).to(inputs.device) * a
            action_log_probs = dist.log_prob(actions)
            policy_probs.append(action_log_probs)
        policy_log_probs = torch.stack(policy_probs, dim=1).squeeze()
        
        return policy_log_probs, hiddens


class RecurrentActor(nn.Module):
    def __init__(self, actor_input_dim,  actor_output_dim, actor_hiddens, activation):
        super(RecurrentActor, self).__init__()
        input_dim = actor_input_dim

        self.num_outputs = actor_output_dim
        self.actor_hiddens = actor_hiddens
        self.rnn_hidden_dim = actor_hiddens[0]
        self.recurrent = True

        if len(actor_hiddens) > 1:
            # at least 2 hidden layers --> 1 hidden layer before RNN, rest after
            self.input_fc = build_sequential(input_dim, [actor_hiddens[0]], activation, output_activation=True)
            self.rnn = nn.GRUCell(actor_hiddens[0], actor_hiddens[0])
            self.output_fc = build_sequential(actor_hiddens[0], actor_hiddens[1:] + [actor_output_dim], activation)
        else:
            # only 1 hidden layer --> first RNN before rest
            self.input_fc = None
            self.rnn = nn.GRUCell(input_dim, actor_hiddens[0])
            self.output_fc = build_sequential(actor_hiddens[0], actor_hiddens + [actor_output_dim], activation)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def _actor_base(self, inputs, hiddens):
        
        if self.input_fc is not None:
            x = self.input_fc(inputs)
        else:
            x = inputs
        # flatten hiddens if needed
        if len(hiddens.shape) > 2:
            hiddens_shape = hiddens.shape
            hiddens = hiddens.reshape(-1, self.rnn_hidden_dim)
            x = x.reshape(-1, x.shape[-1])
        else:
            hiddens_shape = None
        hiddens = self.rnn(x, hiddens)
        # if flattened before, unflatten again
        if hiddens_shape is not None:
            hiddens = hiddens.view(hiddens_shape)
        x = self.output_fc(hiddens)
        return x, hiddens

    def forward(self, inputs):
        raise NotImplementedError
    
    def _get_dist(self, actor_features, action_mask=None):
        """Create categorical distribution, optionally masking invalid actions."""
        if action_mask is not None:
            masked_logits = actor_features.clone()
            masked_logits[~action_mask] = float('-inf')
            return Categorical(logits=masked_logits)
        return Categorical(logits=actor_features)

    def act(self, inputs, hiddens, deterministic=False, action_mask=None):
        """Select action with optional action masking."""
        actor_features, hiddens = self._actor_base(inputs, hiddens)
        dist = self._get_dist(actor_features, action_mask)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.unsqueeze(-1), hiddens

    def evaluate_actions(self, inputs, hiddens, action, action_mask=None):
        """Evaluate log probability with optional action masking."""
        actor_features, hiddens = self._actor_base(inputs, hiddens)
        dist = self._get_dist(actor_features, action_mask)

        action_log_probs = dist.log_prob(action.squeeze()).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy, hiddens
    
    def evaluate_policy_distribution(self, inputs, hiddens, action_mask=None):
        actor_features, hiddens = self._actor_base(inputs, hiddens)
        dist = self._get_dist(actor_features, action_mask)

        policy_probs = []
        for a in range(self.num_outputs):
            actions = torch.ones(inputs.shape[0],).to(inputs.device) * a
            action_log_probs = dist.log_prob(actions)
            policy_probs.append(action_log_probs)
        policy_log_probs = torch.stack(policy_probs, dim=1).squeeze()

        return policy_log_probs, hiddens


class Critic(nn.Module):
    def __init__(self, critic_input_dim, critic_hiddens, activation):
        super(Critic, self).__init__()
        input_dim = critic_input_dim

        #converting critic_hiddens to list for build. need to fix with yaml configs
        self.critic = build_sequential(input_dim, critic_hiddens + [1], activation)

    def forward(self, inputs):
       
        return self.critic(inputs)