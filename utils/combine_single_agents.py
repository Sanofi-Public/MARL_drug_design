#!/usr/bin/env python
"""
Combine Single-Agent Models into Multi-Agent MPO

Takes N single-agent models (each trained on one property) and combines them
into a multi-agent model for multi-property optimization.

Example:
    python combine_single_agents.py \
        --source_models ia2c_models/logd_agent ia2c_models/caco_agent ia2c_models/herg_agent \
        --source_configs configs/logd.json configs/caco.json configs/herg.json \
        --target_config configs/mpo_3_agents.json \
        --output ia2c_models/combined_mpo_model
"""

import argparse
import torch
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gymnasium.spaces import MultiBinary, Tuple, Discrete
from marl_algorithms.a2c.iaa2c_v1 import IAA2C


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_model(cfg, n_agents, input_dim, n_actions):
    """Initialize IAA2C model with specified dimensions."""
    cfg['algorithm']['model']['actor_dim'] = input_dim
    cfg['algorithm']['model']['critic_dim'] = input_dim
    
    observation_space = MultiBinary([input_dim])
    action_space = Discrete(n_actions)
    
    cfg['algorithm']['model']['observation_space'] = Tuple([observation_space for _ in range(n_agents)])
    cfg['algorithm']['model']['action_space'] = Tuple(tuple(n_agents * [action_space]))
    cfg['algorithm']['model']['n_agents'] = n_agents
    
    model = IAA2C(cfg['algorithm']['model'])
    return model


def expand_first_layer_weights(state_dict, old_n_agents, new_n_agents, base_dim, extra_features=4):
    """
    Expand first layer weights for larger agent score encoding.
    
    Input structure: [mol_features (base_dim), scores (n_agents), extra_features]
    """
    if new_n_agents <= old_n_agents:
        return state_dict
    
    # Find the input weight key
    weight_key = None
    for k in state_dict.keys():
        if 'input_fc.0.weight' in k:
            weight_key = k
            break
        elif 'actor.0.weight' in k:
            weight_key = k
            break
        elif 'critic.0.weight' in k:
            weight_key = k
            break
    
    if weight_key is None:
        print(f"  Warning: Could not find first layer weight key")
        return state_dict
    
    old_weight = state_dict[weight_key]
    hidden_dim = old_weight.shape[0]
    
    new_input_dim = base_dim + new_n_agents + extra_features
    new_weight = torch.zeros(hidden_dim, new_input_dim)
    
    # Copy only mol_features part; zero-init all privileged features
    # (scores, is_repeat, target, prev_action) since their semantics change in MARL
    new_weight[:, :base_dim] = old_weight[:, :base_dim]
    
    state_dict[weight_key] = new_weight
    print(f"    Expanded {weight_key}: {old_weight.shape} -> {new_weight.shape} (privileged features zeroed)")
    
    return state_dict


def load_single_agent_checkpoint(model_path, source_n_agents=1, source_extra_features=None):
    """Load single-agent checkpoint and extract state dicts.
    
    Args:
        model_path: Path to model directory
        source_n_agents: Number of agents in source model (default 1)
        source_extra_features: Number of extra features in source obs.
            If None, auto-detect from config or default to 4.
    """
    model_file = os.path.join(model_path, "models.pt")
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    
    # Single agent model has actor_1, critic_1
    actor_sd = ckpt['actor_1'].state_dict()
    critic_sd = ckpt['critic_1'].state_dict()
    
    # Get dimensions
    input_dim = actor_sd['input_fc.0.weight'].shape[1]
    n_actions = actor_sd['output_fc.2.weight'].shape[0]
    
    if source_extra_features is None:
        # Auto-detect: try to find the mol encoding shape from config
        # Default to 4 (is_repeat + target + prev_agent + prev_action)
        source_extra_features = 4
    
    base_dim = input_dim - source_n_agents - source_extra_features
    
    print(f"    input_dim={input_dim}, base_dim={base_dim}, source_extra={source_extra_features}, n_actions={n_actions}")
    
    return actor_sd, critic_sd, base_dim, n_actions


def combine_single_agents(source_model_paths, source_config_paths, target_config_path, output_path):
    """
    Combine multiple single-agent models into a multi-agent MPO model.
    
    Args:
        source_model_paths: List of paths to single-agent model directories
        source_config_paths: List of paths to single-agent config JSONs
        target_config_path: Path to target multi-agent config JSON
        output_path: Path to save combined model
    """
    print("=" * 70)
    print("Combining Single-Agent Models into Multi-Agent MPO")
    print("=" * 70)
    
    n_sources = len(source_model_paths)
    assert n_sources == len(source_config_paths), "Must have same number of models and configs"
    
    # Load target config
    target_cfg = load_config(target_config_path)
    target_props = target_cfg['properties']['names']
    target_n_agents = len(target_props)
    
    print(f"\nTarget config: {os.path.basename(target_config_path)}")
    print(f"  Target properties ({target_n_agents}): {target_props}")
    
    # Load source configs to get property names
    source_props = []
    for cfg_path in source_config_paths:
        cfg = load_config(cfg_path)
        props = cfg['properties']['names']
        assert len(props) == 1, f"Expected single-agent config but got {len(props)} properties in {cfg_path}"
        source_props.append(props[0])
    
    print(f"\nSource models ({n_sources}):")
    for i, (model_path, prop) in enumerate(zip(source_model_paths, source_props)):
        print(f"  [{i}] {prop} <- {model_path}")
    
    # Create property mapping: target_idx -> source_idx
    mapping = {}
    unmapped_targets = []
    
    print("\nProperty mapping:")
    for target_idx, target_prop in enumerate(target_props):
        if target_prop in source_props:
            source_idx = source_props.index(target_prop)
            mapping[target_idx] = source_idx
            print(f"  Target[{target_idx}] '{target_prop}' <- Source[{source_idx}]")
        else:
            unmapped_targets.append(target_idx)
            print(f"  Target[{target_idx}] '{target_prop}' <- RANDOM INIT (no matching source)")
    
    # Load all source checkpoints
    print("\nLoading source checkpoints...")
    actor_state_dicts = []
    critic_state_dicts = []
    base_dim = None
    n_actions = None
    extra_features = 4  # is_repeat + target + prev_agent + prev_action
    
    for i, model_path in enumerate(source_model_paths):
        print(f"  Loading {model_path}...")
        actor_sd, critic_sd, bd, na = load_single_agent_checkpoint(model_path)
        
        if base_dim is None:
            base_dim = bd
            n_actions = na
        else:
            # Verify compatibility
            assert bd == base_dim, f"Incompatible base_dim: {bd} vs {base_dim}"
            assert na == n_actions, f"Incompatible n_actions: {na} vs {n_actions}"
        
        # Expand first layer for multi-agent scoring
        actor_sd = expand_first_layer_weights(actor_sd, 1, target_n_agents, base_dim, extra_features)
        critic_sd = expand_first_layer_weights(critic_sd, 1, target_n_agents, base_dim, extra_features)
        
        actor_state_dicts.append(actor_sd)
        critic_state_dicts.append(critic_sd)
    
    # Calculate target input dimension
    target_input_dim = base_dim + target_n_agents + extra_features
    print(f"\nTarget dimensions: input_dim={target_input_dim}, n_actions={n_actions}")
    
    # Initialize target model (with random weights)
    print(f"\nInitializing target model with {target_n_agents} agents...")
    target_model = setup_model(target_cfg, target_n_agents, target_input_dim, n_actions)
    
    # Load state dicts into target model
    print("\nAssembling combined model:")
    pretrained_indices = []
    
    for target_idx in range(target_n_agents):
        prop_name = target_props[target_idx]
        
        if target_idx in mapping:
            source_idx = mapping[target_idx]
            
            # Load source state dicts into target agent
            target_model.actors[target_idx].load_state_dict(actor_state_dicts[source_idx])
            target_model.critics[target_idx].load_state_dict(critic_state_dicts[source_idx])
            target_model.target_critics[target_idx].load_state_dict(critic_state_dicts[source_idx])
            
            pretrained_indices.append(target_idx)
            print(f"  Agent[{target_idx}] '{prop_name}': loaded from source[{source_idx}]")
        else:
            print(f"  Agent[{target_idx}] '{prop_name}': randomly initialized")
    
    # Save combined model
    os.makedirs(output_path, exist_ok=True)
    
    # Create saveables dict in IAA2C format
    params = []
    for actor, critic in zip(target_model.actors, target_model.critics):
        params += list(actor.parameters())
        params += list(critic.parameters())
    optimiser = torch.optim.Adam(params, lr=target_cfg['algorithm']['model']['lr'])
    
    saveables = {"optimiser": optimiser}
    for i, (actor, critic) in enumerate(zip(target_model.actors, target_model.critics)):
        saveables[f"actor_{i+1}"] = actor
        saveables[f"critic_{i+1}"] = critic
    
    torch.save(saveables, os.path.join(output_path, "models.pt"))
    
    # Save metadata
    metadata = {
        "pretrained_agent_indices": pretrained_indices,
        "source_models": source_model_paths,
        "source_properties": source_props,
        "target_properties": target_props,
        "property_mapping": {str(k): v for k, v in mapping.items()}
    }
    with open(os.path.join(output_path, "composite_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved combined model to: {output_path}")
    print(f"  models.pt")
    print(f"  composite_metadata.json")
    
    # Verification
    print("\nVerifying saved model...")
    verify_model = setup_model(target_cfg, target_n_agents, target_input_dim, n_actions)
    verify_model.restore(output_path, reset_optimizer=True)
    print("Verification successful!")
    
    print("\n" + "=" * 70)
    print("Combined model ready for multi-agent fine-tuning!")
    print("=" * 70)
    print(f"\nNext step: Fine-tune using:")
    print(f"  python train.py --config {target_config_path}")
    print(f"  (with pre_train=true and pre_train_path={output_path})")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Combine single-agent models into multi-agent MPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine 3 single-agent models
  python combine_single_agents.py \\
      --source_models ia2c_models/logd_agent ia2c_models/caco_agent ia2c_models/herg_agent \\
      --source_configs configs/logd.json configs/caco.json configs/herg.json \\
      --target_config configs/mpo_3_agents.json \\
      --output ia2c_models/combined_mpo

  # Then fine-tune with:
  python train_v5.py --config configs/mpo_3_agents.json
  (set pre_train=true and pre_train_path in config)
        """
    )
    
    parser.add_argument('--source_models', nargs='+', required=True,
                        help='Paths to single-agent model directories')
    parser.add_argument('--source_configs', nargs='+', required=True,
                        help='Paths to single-agent config JSONs')
    parser.add_argument('--target_config', required=True,
                        help='Path to target multi-agent config JSON')
    parser.add_argument('--output', required=True,
                        help='Path to save combined model')
    
    args = parser.parse_args()
    
    combine_single_agents(
        source_model_paths=args.source_models,
        source_config_paths=args.source_configs,
        target_config_path=args.target_config,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
