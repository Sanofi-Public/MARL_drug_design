

# Configure logging
# Start the REST app as a subprocess
import os
import sys
import socket
import subprocess
import gymnasium as gym
import subprocess
from gymnasium.spaces import flatdim,MultiBinary, Tuple,Discrete
import requests
from urllib.parse import urlparse

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_api_url(url: str) -> str:
    """Reject plaintext HTTP unless the target is a loopback address."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS:
        return url
    raise ValueError(
        f"Refusing to call {url!r}: use HTTPS for non-loopback hosts to prevent MitM attacks."
    )


def _is_api_reachable(url: str, connect_timeout: float = 5.0) -> bool:
    """Liveness probe: check the host:port accepts a TCP connection.

    This is a pure TCP handshake — no HTTP payload is sent (CWE-319 N/A) and
    no TLS handshake is performed. The single socket is released in a
    ``finally`` block (CWE-404 N/A). For HTTPS URLs this only proves the port
    is listening; that is the sole intent of a liveness probe.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port

    sock = None
    try:
        try:
            sock = socket.create_connection((host, port), timeout=connect_timeout)
            return True
        except (OSError, socket.timeout):
            return False
    finally:
        if sock is not None:
            sock.close()


import time
import numpy as np
from collections import OrderedDict
import torch
from marl_algorithms.a2c.iaa2c_v1 import IAA2C
from encoders.gnn_encoder import SMILES_to_Graph, GraphConv,GraphEncoder

def load_model (gnn_model, model_path, device):
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.to(device)
    gnn_model.eval()
    return gnn_model

def get_outputs(in_smiles,out_smiles,in_frags,out_frags,gnn_model,device):
    in_node_features, in_adj = SMILES_to_Graph(max_atoms=100).featurize(in_smiles)
    out_node_features, out_adj = SMILES_to_Graph(max_atoms=100).featurize(out_smiles)
    #check if frags are tensors, if not convert to tensors
    if not isinstance(in_frags, torch.Tensor):
        in_frags = torch.tensor(in_frags, dtype=torch.float)
        out_frags = torch.tensor(out_frags, dtype=torch.float)
    delta, outputs=gnn_model(in_node_features.unsqueeze(0).to(device), in_adj.unsqueeze(0).to(device), out_node_features.unsqueeze(0).to(device), out_adj.unsqueeze(0).to(device))
    return outputs.detach().cpu().numpy()


def wait_for_rest_api(url, timeout=30):
    """Wait until the REST API is responsive or timeout is reached."""
    url = _validate_api_url(url)
    start_time = time.time()
    while True:
        if _is_api_reachable(url):
            print("REST API is ready.")
            return True

        if time.time() - start_time > timeout:
            raise TimeoutError("REST API did not start within the timeout period.")
        time.sleep(1)

def start_rest_subprocess():
    """Start the REST API in the background.

    Reads ``REST_PYTHON``, ``REST_APP_PATH`` and ``REST_PORT`` from the
    environment so no user- or machine-specific paths are baked into the repo.
    ``REST_APP_PATH`` is required.
    """
    rest_python = os.environ.get("REST_PYTHON", sys.executable)
    rest_app = os.environ.get("REST_APP_PATH")
    rest_port = os.environ.get("REST_PORT", "2000")
    if not rest_app:
        raise RuntimeError(
            "REST_APP_PATH environment variable is not set. "
            "Point it to your REST scoring server script."
        )
    return subprocess.Popen([rest_python, rest_app, "--port", rest_port])


def start_rest_uwsgi():
    """Start the REST API under uWSGI.

    ``REST_UWSGI`` (path to the ``uwsgi`` binary) and ``REST_UWSGI_INI``
    (path to the ini config) must be provided via environment variables.
    """
    rest_uwsgi = os.environ.get("REST_UWSGI")
    rest_uwsgi_ini = os.environ.get("REST_UWSGI_INI")
    if not rest_uwsgi or not rest_uwsgi_ini:
        raise RuntimeError(
            "REST_UWSGI and REST_UWSGI_INI environment variables must be set."
        )
    return subprocess.Popen([rest_uwsgi, "--ini", rest_uwsgi_ini])


def initialize_training_params(cfg, test_mols):
        """Initialize training parameters from config and test molecules.
        
        Args:
            cfg (dict): Configuration dictionary containing training parameters.
            test_mols (list): List of test molecules for determining observation space.
            
        Returns:
            tuple: Contains initialized parameters and model instance.
        """
        n_steps = cfg['algorithm']['n_steps']
        num_env_steps = cfg['algorithm']['num_env_steps']
        n_agents = cfg['algorithm']['model']['n_agents']
        parallel_envs = cfg['env']['parallel_envs']
        device = cfg['algorithm']['model']['model_device']
        recurrent = cfg['algorithm']['model']['recurrent']
        n_actions = cfg['deepfmpo']['MAX_FRAGMENTS'] * cfg['deepfmpo']['MAX_SWAP'] + 1
        max_ep = cfg['env']['max_ep_length']
        log_interval = cfg['algorithm']['model']['log_interval']
        num_updates = (int(num_env_steps) // n_steps // parallel_envs)
        seed = cfg['seed']
        gamma=cfg['algorithm']['model']['gamma']
        pre_train=cfg['algorithm']['pre_train']
        pre_train_path=cfg['algorithm']['pre_train_path']
       

        d1, d2 = test_mols[0].shape
        goal_conditioned = cfg.get('curriculum', {}).get('goal_conditioned', False)
        goal_dim = 2 * n_agents if goal_conditioned else 0  # lo + hi per agent
        privileged_dim = n_agents + 4 + goal_dim  # scores + is_repeat + target + prev_agent + prev_action + goal_bounds
        if cfg['gnn']['use_gnn']:
            embedding_dim = cfg['gnn']['embedding_dim']
            extra_features = embedding_dim + privileged_dim
        else:
            extra_features = privileged_dim
        input_dim = cfg["algorithm"]["model"]["nn_dim"] = d1 * d2 + extra_features
        
        observation_space = MultiBinary([d2, d1])
        print("The observation space is {}".format(observation_space))
        action_space = Discrete(n_actions)
        
        # Fragment embedding: replaces flat mol bits with learned embeddings
        # The raw observation keeps the same shape (input_dim) — embedding
        # happens inside the model. But actor/critic input dims change.
        frag_embed_cfg = cfg['algorithm']['model'].get('fragment_embedding', None)
        use_frag_embed = frag_embed_cfg is not None and frag_embed_cfg.get('use', False)
        if use_frag_embed:
            max_frags = d1  # MAX_FRAGMENTS
            enc_bits = d2 - 1  # encoding bits (total cols minus exists flag)
            frag_embed_dim = frag_embed_cfg['embed_dim']
            # Store derived values so IAA2C can build the module
            frag_embed_cfg['max_frags'] = max_frags
            frag_embed_cfg['enc_bits'] = enc_bits
            # Embedded mol dim replaces flat mol dim in the network input
            embedded_mol_dim = max_frags * frag_embed_dim
            network_input_dim = embedded_mol_dim + extra_features
        else:
            network_input_dim = input_dim
        
        privileged_critic = cfg['algorithm']['model'].get('privileged_critic', False)
        if privileged_critic:
            actor_input_dim = network_input_dim - privileged_dim
            critic_input_dim = network_input_dim
        else:
            actor_input_dim = network_input_dim
            critic_input_dim = network_input_dim
        
        cfg['algorithm']['model']['observation_space'] = Tuple([observation_space for _ in range(n_agents)])
        cfg['algorithm']['model']['action_space'] = Tuple(tuple(n_agents * [action_space]))
        cfg['algorithm']['model']['actor_dim'] = actor_input_dim
        cfg['algorithm']['model']['critic_dim'] = critic_input_dim
        cfg['algorithm']['model']['privileged_dim'] = privileged_dim if privileged_critic else 0
        actor_dim = cfg['algorithm']['model']['actor'][0]
        
        obs_shape = d1 * d2 + cfg["algorithm"]["to_pad"]
        test_n = IAA2C(cfg['algorithm']['model'])
        if pre_train:
            print(f"Loading pre-trained model from {pre_train_path}")
            test_n.restore(pre_train_path,reset_optimizer=True)
            #test_n.reinitialize_optimizer()
        return (n_steps, num_env_steps, n_agents, parallel_envs, device, recurrent, 
                n_actions, max_ep, log_interval, num_updates, seed, input_dim, 
                actor_dim, test_n,obs_shape, gamma)
