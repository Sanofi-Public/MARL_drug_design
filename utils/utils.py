import collections
import math

import gymnasium as gym
import torch
from torch._C import Value
from torch.autograd import Variable
import numpy as np


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def split_batch(splits, device, dim=-1):
    def thunk(batch):
        return torch.split(batch.to(device), splits, dim=dim)
    return thunk
    
def split_dims(gym_spaces):
    """
    Extract dimensions for splitting
    :param gym_spaces (List[gym.spaces]): gym spaces
    :return (List[int]): splitting dims
    """
    is_multibinary = lambda x: isinstance(x, gym.spaces.MultiBinary)
    all_multibinary = all([is_multibinary(space) for space in gym_spaces])
    none_multibinary = all([not is_multibinary(space) for space in gym_spaces])
    assert all_multibinary or none_multibinary

    if none_multibinary:
        return [gym.spaces.flatdim(space) for space in gym_spaces]
    else:
        # all MultiBinary --> have shape property
        return [space.shape[0] * space.shape[1] +2 for space in gym_spaces]


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    
    keep_list = ["target_attained", "mol_id","episode_reward","episode_length","episode_time"]
    keys_to_keep = {key for key in keys if key in keep_list}
    
    for key in keys_to_keep:
        values = [d[key] for d in info if key in d]
        
        if key == 'target_attained':
            summation = np.sum(values)
            new_info[key] = summation
        elif key == 'mol_id':
            new_info[key] = values

        else:

            mean = np.mean(values, 0)
            new_info[key] = mean
    
    return new_info



def evaluate(
    parallel_envs,
    envs,
    agent,
    ae,
    device,
    episodes_per_eval,
    split_obs,
):
    obs = envs.reset()
    completed_episodes=0

    n_agents=3

    current_obs=torch.cat([torch.from_numpy(o).float() for o in obs], dim=1).reshape(parallel_envs,81*n_agents) 
    
    hiddens = [
        torch.zeros(parallel_envs, hidden_dim).to(device) for hidden_dim in agent.hidden_dims()
    ]
    if ae:
        task_embs = [torch.zeros(parallel_envs, agent.task_emb_dim) for _ in range(n_agents)]
        ae_hiddens = [
            torch.zeros(parallel_envs, hidden_dim).to(device) for hidden_dim in ae.hidden_dims()
        ]
    else:
        task_embs = [None for _ in range(n_agents)]
    
    all_infos = []
    while len(all_infos) < episodes_per_eval:
        #tensor_obs=torch.stack([torch.from_numpy(current_obs[i]).float() for i in range(len(current_obs))])
       
        obs = split_obs(current_obs)
        with torch.no_grad():
            actions, hiddens = agent.act(obs, task_embs, hiddens, evaluation=True)
        env_actions = torch.cat(actions, dim=1)
        next_obs, rew, done, infos = envs.step(env_actions.tolist())
    
        rew = list(torch.stack([torch.from_numpy(r).float() for r in rew], dim=-1).unsqueeze(-1))
        if ae:
            task_embs, ae_hiddens = ae.encode(obs, actions, rew, ae_hiddens, no_grads=True)

        next_obs = [torch.from_numpy(o).float() for o in next_obs]
        current_obs = torch.stack(next_obs, dim=1).reshape(parallel_envs, 9*9*n_agents)
        
        for i, info in enumerate(infos):
                infodict=infos[i]
                #print(infodict)
                #list_indices=[x['mol_id'] for x in all_infos]
                #if infodict['mol_id'] not in list_indices:
                    
                if 'episode_reward' in infodict:
                        completed_episodes += 1
                        infodict["completed_episodes"] = completed_episodes
                        if  infodict['target_achieved']:
                            infodict['target_attained']=1

                        else:
                            infodict['target_attained']=0
                        all_infos.append(infodict)
                        #logger.log_episode(total_steps, infodict)
                        for hidden in hiddens:
                            hidden[i, :].zero_()
                        if ae:
                            for ae_hidden in ae_hiddens:
                                ae_hidden[i, :].zero_()

    #list_indices=[x['mol_id'] for x in all_infos]
    #print(list_indices)
    return all_infos

# def split_dims(gym_spaces):
#     """
#     Extract dimensions for splitting
#     :param gym_spaces (List[gym.spaces]): gym spaces
#     :return (List[int]): splitting dims
#     """
#     is_box = lambda x: isinstance(x, gym.spaces.Box)
#     all_box = all([is_box(space) for space in gym_spaces])
#     none_box = all([not is_box(space) for space in gym_spaces])
#     assert all_box or none_box

#     if none_box:
#         return [gym.spaces.flatdim(space) for space in gym_spaces]
#     else:
#         # all box --> have shape property
#         return [space.shape[0] for space in gym_spaces]


def concat_shapes(shapes):
    """
    Concatenate shape of multiple shapes
    :param shapes (List[Tuple[int]]): list of shapes
    :return: concatenated shape
    """
    # all need to have same length and same shape aside from first entry
    assert len(shapes) >= 1
    shape = shapes[0]
    if not all([len(s) == len(shape) for s in shapes[1:]]):
        raise ValueError("All shapes for concatenation need to have same dimensionality.")
    if not all([s[1:] == shape[1:] for s in shapes[1:]]):
        raise ValueError("All shapes for concatenation need to have values in all dims but 0.")
    cat_dim = sum([s[0] for s in shapes])
    return tuple([cat_dim] + list(shape[1:]))


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)