import ast
import torch
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors,PandasTools
from encoders.fmpo_utils.build_encoding import read_decodings
from utils.reward_functions import target_check, check_bounds_all
import json

prop_list = [
            "macrocycle",
            "has_not_substructure",
            "has_substructure",
            "n_chiral",
            "n_element",
            "n_cycles",
            "n_heavy_atoms",
            "molweight",
            "numhdonors",
            "numhacceptors",
            "tpsa",
            "n_rotable_bonds",
            "fingerprint_similarity",
            "logp",
            "count_radicals",
            "qed",
            "avoid_known_molecules",
            "cns_mpo_old",
            "add_heavy_atoms",
            "free_positions",
            "drug_likeness",
            # "subtan", # NOTE subtan returns inconsistent results in different environments
            "shape3d",
            "alerts",
            "tanimoto",
            "unwanted_frags",
        ]
def create_scoring_function_config(sample_path,cfg):
    """
    Create a configuration dictionary for scoring functions based on the provided JSON sample.
    
    Args:
        sample_json (dict): Basic template of scoring function calculator for modification.
        properties (list): List of properties to be included in the scoring function.
        bounds (dict): Dictionary containing bounds for each property.
    Returns:
        updated_json (dict): Updated JSON with scoring function configuration.
    """
    with open(sample_path, 'r') as f:       
        # Load the sample JSON configuration
        # This should be a basic template of scoring function calculator for modification
        sample_json=json.load(f)
    
    property_dict=cfg["properties"]
    prop_list=property_dict["names"]
    property_bounds=property_dict["bounds"]
    property_args=property_dict["args"]
    property_types=property_dict["types"]
    # Initialize the scoring function configuration
    for i in range(len(prop_list)):
        prop=prop_list[i]
        
        bounds=property_bounds[i]
        lower_bound=bounds[0]
        upper_bound=bounds[1]
        prop_args=property_args[i]
        property_type=property_types[i]
        if property_type is not None:

            type=property_type
        else:
            type=None
        
        if type is None:
            if prop not in prop_list:
                
                raise ValueError(f"Property {prop} is not in the predefined property list.")
                
        if prop not in sample_json:
        
            sample_json[prop] = {}
        try:
            sample_json[prop]['weight'] = 1
            sample_json[prop]['range'] = [lower_bound, upper_bound]
            sample_json[prop]['score_range']=[0.01, 1]
            sample_json[prop]['minimize'] = False 
            sample_json[prop]['args']=prop_args
            sample_json[prop]['scaling'] = 'quadratic'
            if type is not None:
                sample_json[prop]['type'] = type
        except KeyError:
            print(f"KeyError: Please check the JSON structure.")
            
    #if file exists delete it
    if os.path.exists('configs/marl_scoring.json'):
        os.remove('configs/marl_scoring.json')
    #save the updated configuration to a JSON file
    with open('configs/marl_scoring.json', 'w') as f:
        json.dump(sample_json, f, indent=4)
    

def draw_smiles_properties(smiles):
   
    scores_list = []
    leg_list = []
    prop_list = ['MolWt', 'TPSA', 'LogP']
    smiles_mol=[]
    decodings=read_decodings("data/decodings.txt")
    mol_list=[]
    
    start_list=[]
    for i in range(len(smiles)):
        mol =Chem.MolFromSmiles(smiles[i])
        smiles_mol.append(Chem.MolToSmiles)
        scores = [Descriptors.MolWt(mol), TPSA(mol), Descriptors.MolLogP(mol)]
                # Determine color based on criteria

        props =  'Prop'+ ', '.join(f'{prop}: {scores[j]:.1f}' for j, prop in enumerate(prop_list))
        leg_list.append(props)
        scores_list.append(scores)
        mol_list.append(mol)

    img = Draw.MolsToGridImage(mol_list, legends=leg_list, molsPerRow=5, maxMols=999999, subImgSize=(250, 250))

    return  img


def draw_properties(mol_arr):
    scores_list = []
    leg_list = []
    prop_list = ['MolWt', 'TPSA', 'LogP']
    smiles_mol=[]
    decodings=read_decodings("data/decodings.txt")
    mol_list=[]
    
    start_list=[]
    for i in range(len(mol_arr)):
        mol =decode(mol_arr[i],decodings)
        smiles_mol.append(Chem.MolToSmiles)
        scores = [Descriptors.MolWt(mol), TPSA(mol), Descriptors.MolLogP(mol)]
                # Determine color based on criteria

        props =  'Prop'+ ', '.join(f'{prop}: {scores[j]:.1f}' for j, prop in enumerate(prop_list))
        leg_list.append(props)
        scores_list.append(scores)
        mol_list.append(mol)

    img = Draw.MolsToGridImage(mol_list, legends=leg_list, molsPerRow=5, maxMols=999999, subImgSize=(250, 250))

    return  img

def calculate_and_draw_properties(df):
    scores_list = []
    leg_list = []
    prop_list = ['MolWt', 'TPSA', 'LogP']
    smiles_mol=[]
    
    max_step = df['Step'].max()
    step_df=df[df['Step'] == max_step]
    
    # Add RDKit molecule column to DataFrame
    PandasTools.AddMoleculeColumnToFrame(step_df, 'Smiles', 'Molecule')
    start_list=[]
    for idx, row in step_df.iterrows():
        mol = row['Molecule']
        agent = row['Agent']
        start_mol=row['Starting_mol']
        smiles_mol.append(row['Smiles'])
        scores = [Descriptors.MolWt(mol), TPSA(mol), Descriptors.MolLogP(mol)]
                # Determine color based on criteria
        colors = ['green' if (prop == 'MolWt' and 420>scores[j] > 320) or 
                            (prop == 'TPSA' and 60>scores[j] >50) or 
                            (prop == 'LogP' and 3>scores[j] >2) else 'black' 
                  for j, prop in enumerate(prop_list)]
        start_list.append(start_mol)
        # props = f'<font color="black">Ag: {agent}</font>, ' + \
        #         f'<font color="black">St_mol: {start_mol}</font>, ' + \
        #         ', '.join(f'<font color="{colors[j]}">{prop}: {scores[j]:.1f}</font>' for j, prop in enumerate(prop_list))
        # print(props)
        # props = f'<font color="blue">Ag: {agent}</font>, ' + \
        #         f'<font color="green">St_mol: {start_mol}</font>, ' + \
        #         ', '.join(f'<font color="red">{prop}: {scores[j]:.1f}</font>' for j, prop in enumerate(prop_list))
        props = f'Ag: {agent}, ' +f'St_mol: {start_mol}, '+ ', '.join(f'{prop}: {scores[j]:.1f}' for j, prop in enumerate(prop_list))
        leg_list.append(props)
        scores_list.append(scores)
    #print(len(leg_list))
    #print(len(scores_list))
    img = Draw.MolsToGridImage(step_df['Molecule'], legends=leg_list, molsPerRow=5, maxMols=999999, subImgSize=(250, 250))

    return scores_list, leg_list,start_list,smiles_mol, img

def retrieve_mols(df):

    #non_target_df=df.loc[df['Target']==False]
    non_smiles=list(df['Smiles'])

    target_df=df.loc[df['Target']==True]
    target_smiles=list(target_df['Smiles'])
    return non_smiles,target_smiles
    

def step_evaluate(df):
    target_list=[]
    max_value = df['Step'].max()
    print(max_value)
    for i in range(1,max_value):
        step=i
        step_df=df.loc[df['Step'] == step]
        #print(step_df.head(5))
        #target_count=step_df['Target'].value_counts()[True]
        target_count=step_df['Target'].values.sum()
        other_count=(~step_df['Target']).values.sum()
        target_list.append([target_count,other_count])
    return target_list




def retrieve_targets(df,index):
    target_df=df.loc[df['Target']==True]
    #print(mol_df)
    agent_list=[index]*len(target_df)
    #mol_df['Agent']=agent_list
    target_df.loc[:, 'Agent'] = agent_list
    #unique_combinations = mol_df.groupby(['Smiles','Starting_mol'])
    unique_combinations = target_df.drop_duplicates(subset=['Smiles','Starting_mol'])

    return unique_combinations,target_df
    

def mol_weight(mol):
    return Descriptors.ExactMolWt(mol)


def TPSA(mol):

    score=0
    try:
        Chem.SanitizeMol(mol)
        mol.GetRingInfo()
        score=Descriptors.TPSA(mol)

    except:
        score=-1e5

    return score

def logp(mol):

    return Descriptors.MolLogP(mol)


def decode(x, translation):
        
        enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]

        fs = [Chem.Mol(translation[e]) for e in enc]
        try:
            import preprocessing.mol_utils as mol_utils
            return mol_utils.join_fragments(fs)
        except:
            raise RuntimeError("Something went wrong when joining fragments.")


def save_models(alg,t):
    
    save_at = os.path.join("ia2c_models", f"t_{t}")
    os.makedirs(save_at, exist_ok=True)               
    alg.save(save_at)

        # info = {
        #     'env_index': self.env_index,
        #     'agent_id': current_agent.number,
        #     'all rewards': reward,
        #     'actions': action,
        #     'indicators': self.prop_indicator,
        #     'start smiles': start_smiles,
        #     'end smiles': end_smiles,
        #     'current_molecule_count': count,
        #     'unique_found': len(self.target_dict),
        #     # Reward components for tracking
        #     'no_op_reward': no_op_reward,
        #     'base_reward': base_reward,
        #     'invalid_penalty': invalid_penalty,
        #     'target_bonus': target_bonus,
        #     'repeat_penalty': repeat_penalty,
        #     'novelty_reward': novelty_reward,
        #     'deferred_bonus_collected': deferred_bonus_collected,
        #     'shared_credit_distributed': shared_credit_info,
        #     'total_reward': total_reward,
        #     # Action flags
        #     'is_noop': is_noop,
        #     'is_invalid': is_invalid,
        #     'is_prop_improved': is_prop_improved,
        #     # Other info
        #     'qualified_agents': list(self.get_qualified_agents()),
        #     'ep_reward': current_agent.rewards,
        #     'frag_added': added_frag,
        #     'frag_removed': removed_frag,
        #     'ep_step': self.current_step,
        #     'target': self.mol_made,
        #     'episode_number': ep_num,
        #     'scores': scores,
        #     'mpo_score': mpo_score,
        #     'infostr': infostr,
        #     'mol_id': self.mol_id

def save_agent_smiles(data):
    agent_smiles_dict = {}

    for d in data:
        for entry in d:
            env_index = entry['env_index']
            agent_id = int(entry['agent_id'])
            all_rewards = entry['all rewards']
            actions = entry['actions']
            indicators = entry['indicators']
            start_smiles = entry['start smiles']
            end_smiles = entry['end smiles']
            current_molecule_count = entry['current_molecule_count']
            unique_found = entry['unique_found']
            # Reward components
            no_op_reward = entry['no_op_reward']
            base_reward = entry['base_reward']
            invalid_penalty = entry['invalid_penalty']
            target_bonus = entry['target_bonus']
            repeat_penalty = entry['repeat_penalty']
            novelty_reward = entry['novelty_reward']
            deferred_bonus_collected = entry['deferred_bonus_collected']
            shared_credit_distributed = entry['shared_credit_distributed']
            total_reward = entry['total_reward']
            # Action flags
            is_noop = entry['is_noop']
            is_invalid = entry['is_invalid']
            is_prop_improved = entry['is_prop_improved']
            # Other info
            qualified_agents = entry['qualified_agents']
            ep_reward = entry['ep_reward']
            frag_removed = entry['frag_removed']
            frag_added = entry['frag_added']
            ep_step = entry['ep_step']
            target = entry['target']
            episode_number = entry['episode_number']
            scores = entry['scores']
            mpo_score = entry['mpo_score']
            mol_id = entry['mol_id']
            prop_indicator = indicators[agent_id]

            if agent_id not in agent_smiles_dict:
                agent_smiles_dict[agent_id] = []

            agent_smiles_dict[agent_id].append([
                env_index,                    # 0
                start_smiles,                 # 1
                end_smiles,                   # 2
                actions,                      # 3
                no_op_reward,                 # 4
                base_reward,                  # 5
                invalid_penalty,              # 6
                target_bonus,                 # 7
                repeat_penalty,               # 8
                novelty_reward,               # 9
                deferred_bonus_collected,     # 10
                shared_credit_distributed,    # 11
                total_reward,                 # 12
                is_noop,                      # 13
                is_invalid,                   # 14
                is_prop_improved,             # 15
                qualified_agents,             # 16
                ep_step,                      # 17
                unique_found,                 # 18
                current_molecule_count,       # 19
                prop_indicator,               # 20
                indicators,                   # 21
                all_rewards,                  # 22
                ep_reward,                    # 23
                frag_removed,                 # 24
                frag_added,                   # 25
                target,                       # 26
                episode_number,               # 27
                scores,                       # 28
                mpo_score,                    # 29
                mol_id                        # 30
            ])

    return agent_smiles_dict

def rewrite_dict(nested_dict):
    # Dynamically collect all agent keys from the nested dict
    all_keys = set()
    for inner_dict in nested_dict.values():
        all_keys.update(inner_dict.keys())
    new_dict = {k: [] for k in sorted(all_keys)}
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, smiles_list in inner_dict.items():
            new_dict[inner_key].extend(smiles_list)
    return new_dict

def add_to_df(df_list, t_dict, step, cfg, scorer):
    prop_dict = cfg["properties"]
    prop_names = prop_dict["names"]
    prop_bounds = prop_dict["bounds"]

    for i in range(len(df_list)):
        df = df_list[i]
        # Unpack all fields based on new structure
        env_index_list = [x[0] for x in t_dict[i]]
        start_smiles_list = [x[1] for x in t_dict[i]]
        end_smiles_list = [x[2] for x in t_dict[i]]
        actions_list = [x[3] for x in t_dict[i]]
        no_op_reward_list = [x[4] for x in t_dict[i]]
        base_reward_list = [x[5] for x in t_dict[i]]
        invalid_penalty_list = [x[6] for x in t_dict[i]]
        target_bonus_list = [x[7] for x in t_dict[i]]
        repeat_penalty_list = [x[8] for x in t_dict[i]]
        novelty_reward_list = [x[9] for x in t_dict[i]]
        deferred_bonus_collected_list = [x[10] for x in t_dict[i]]
        shared_credit_distributed_list = [x[11] for x in t_dict[i]]
        total_reward_list = [x[12] for x in t_dict[i]]
        is_noop_list = [x[13] for x in t_dict[i]]
        is_invalid_list = [x[14] for x in t_dict[i]]
        is_prop_improved_list = [x[15] for x in t_dict[i]]
        qualified_agents_list = [x[16] for x in t_dict[i]]
        ep_step_list = [x[17] for x in t_dict[i]]
        unique_found_list = [x[18] for x in t_dict[i]]
        current_molecule_count_list = [x[19] for x in t_dict[i]]
        prop_indicator_list = [x[20] for x in t_dict[i]]
        indicators_list = [x[21] for x in t_dict[i]]
        all_rewards_list = [x[22] for x in t_dict[i]]
        ep_reward_list = [x[23] for x in t_dict[i]]
        frag_removed_smiles_list = [x[24] for x in t_dict[i]]
        frag_added_smiles_list = [x[25] for x in t_dict[i]]
        target_list = [x[26] for x in t_dict[i]]
        episode_number_list = [x[27] for x in t_dict[i]]
        scores_list = [x[28] for x in t_dict[i]]
        mpo_score_list = [x[29] for x in t_dict[i]]
        mol_id_list = [x[30] for x in t_dict[i]]
        step_list = [step] * len(start_smiles_list)

        new_df = pd.DataFrame({
            'Env_Index': env_index_list,
            'Step': step_list,
            'Episode_number': episode_number_list,
            'Mol_ID': mol_id_list,
            'Start_Smiles': start_smiles_list,
            'End_Smiles': end_smiles_list,
            'Actions': actions_list,
            'Target': target_list,
            # Reward components
            'No_Op_Reward': no_op_reward_list,
            'Base_Reward': base_reward_list,
            'Invalid_Penalty': invalid_penalty_list,
            'Target_Bonus': target_bonus_list,
            'Repeat_Penalty': repeat_penalty_list,
            'Novelty_Reward': novelty_reward_list,
            'Deferred_Bonus_Collected': deferred_bonus_collected_list,
            'Shared_Credit_Distributed': shared_credit_distributed_list,
            'Total_Reward': total_reward_list,
            # Action flags
            'Is_Noop': is_noop_list,
            'Is_Invalid': is_invalid_list,
            'Is_Prop_Improved': is_prop_improved_list,
            # Other info
            'Qualified_Agents': qualified_agents_list,
            'Prop_Indicator': prop_indicator_list,
            'All_Indicators': indicators_list,
            'All_Rewards': all_rewards_list,
            'Episode_Reward': ep_reward_list,
            'Episode_Step': ep_step_list,
            'Unique_Found': unique_found_list,
            'Current_Molecule_count': current_molecule_count_list,
            'Frag_Removed_SMILES': frag_removed_smiles_list,
            'Frag_Added_SMILES': frag_added_smiles_list,
            'Scores': scores_list,
            'MPO_Score': mpo_score_list
        })

        df_list[i] = pd.concat([df, new_df], ignore_index=True)

    return df_list


def save_dflist_to_csv(df_list, config_name):
    folder_name='ia2c_results'+'/'+config_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder {folder_name} already exists. Files will be overwritten.")
        rm_files = [f for f in os.listdir(folder_name) if f.startswith('agent_') and f.endswith('.csv')]
        for f in rm_files:
            os.remove(os.path.join(folder_name, f))
    for i in range(len(df_list)):
        
        df=df_list[i]
        
        f_path=folder_name+'/agent_'+str(i)+'.csv'
        with open(f_path, "a") as f:
            df.to_csv(f,index=False)

