import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}


import numpy as np
import pandas as pd  # Add pandas import

import encoders.fmpo_utils.file_io as file_io
import encoders.fmpo_utils.build_encoding as build_encoding
import encoders.fmpo_utils.visualization as visualization
import encoders.fmpo_utils.mol_utils as mol_utils
import encoders.fmpo_utils.global_parameters as gl
from encoders.fmpo_utils.mol_utils_brics import get_fragments_brics
# from encoders.fmpo_utils.mol_utils_reinvent import get_fragments_reinvent

from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')


def log_unfragmentable_molecules(lead_mols_total, used, output_path):
    """
    Log molecules that could not be fragmented.
    
    Args:
        lead_mols_total: list of RDKit mol objects
        used: list of booleans indicating if molecule was successfully fragmented
        output_path: path to save the unfragmentable molecules
    """
    unfragmentable = []
    for i, (mol, was_used) in enumerate(zip(lead_mols_total, used)):
        if not was_used:
            smi = Chem.MolToSmiles(mol) if mol else "Invalid"
            unfragmentable.append({
                'index': i,
                'smiles': smi
            })
    
    if unfragmentable:
        df = pd.DataFrame(unfragmentable)
        unfrag_path = os.path.join(output_path, "unfragmentable_molecules.csv")
        df.to_csv(unfrag_path, index=False)
        print(f"\n{'='*60}")
        print(f"WARNING: {len(unfragmentable)} molecules could not be fragmented!")
        print(f"Saved to: {unfrag_path}")
        print(f"{'='*60}")
        for item in unfragmentable[:10]:  # Show first 10
            print(f"  Index {item['index']}: {item['smiles']}")
        if len(unfragmentable) > 10:
            print(f"  ... and {len(unfragmentable) - 10} more")
        print(f"{'='*60}\n")
    
    return unfragmentable


def encoding_maker(cfg):
    #the reference function for this is the main function of the Main.py from DEEPFMPO code.
    #takes jsonconfig as input and outputs encodings/decodings after making a tree
    #code follows DEEPFMPO implementation of the same, with no changes in algorithm
    #only change is in the variables used, for ease of calculation
    #we devise two modes, if save=True, we save the generated encodings/decodings etc.
    
    # create the scoring function with kwargs
 #   sf = scoring_functions.get_scoring_function(gl.PARAMS["SCORING_FUNCTION"], **gl.PARAMS["SF_KWARGS"])

    lead_fragments=cfg['deepfmpo']['lead_frags'] #replaces gl.PARAMS["LEAD_FRAGMENTS"]
    lead_file=cfg['deepfmpo']['lead_file']  #replaces gl.PARAMS["LEAD_FILE"]
    more_output=cfg['deepfmpo']['more_output'] #replaces gl.PARAMS["MORE_OUTPUT] #true,false
    output_path=cfg['deepfmpo']['output_path'] #path to save encoding outputs
    write_decodings='/decodings' #additional path to save decodings
    return_mode=cfg['deepfmpo']['return_mode'] #if true it returns the lead_codes, decodings and freeze_encodings else only saves
    train_file=cfg['deepfmpo']['train_file']  #file with training molecules
    eval_file=cfg['deepfmpo']['eval_file']    #file with evaluation
    use_brics=cfg['deepfmpo'].get('use_brics', False)  #whether to use brics fragmentation
    use_reinvent=cfg['deepfmpo'].get('use_reinvent', False)  #whether to use reinvent fragmentation
    # Set freeze frags from config into global params before fragmentation
    gl.PARAMS["FREEZE_FRAGS"] = cfg['deepfmpo'].get('freeze_frags', [])
    frag_var=0
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print("the fragment files is {}".format(lead_fragments) if lead_fragments is not None else "None")
    print( "the lead file is {}".format(lead_file) if lead_file is not None else "None")
    # get lead molecules and fragment them
    if lead_fragments is not None:  # read leads as fragments
        lead_sets = file_io.read_fragments(lead_fragments)
        #print(lead_sets)
        lead_frags = {} 
        lead_mols = []
        lead_smiles = []
        for lead_set in lead_sets:
            lead_frags.update(mol_utils.mols_to_frags(lead_set))
            # lead_mol = mol_utils.join_fragments(lead_set)
            # smi = Chem.MolToSmiles(lead_mol)
            # if smi not in lead_smiles:
            #     lead_mols.append(lead_mol)
            #     lead_smiles.append(smi)
        #print("We have {} lead fragment sets, forming {} lead molecules".format(len(lead_sets), len(lead_mols)))
    else:
        frag_var=1
        lead_frags = {} 
        #("No files with fragments given!")
    
    if lead_file is not None:  # read leads as molecules
        print("Reading lead molecules from file: {}".format(lead_file))
        lead_mols_total = file_io.read_molfile(lead_file,file_type='csv',as_smiles=False)
        
        # Use appropriate fragmentation method (priority: reinvent > brics > deepfmpo)
        if use_reinvent:
            print("Using REINVENT fragmentation with named reactions [*:n]")
            lead_frags_new, used, lead_sets_new = get_fragments_reinvent(
                lead_mols_total, output_path+"/lead_frags.csv"
            )
        elif use_brics:
            print("Using BRICS fragmentation with BRICS-typed attachments [n*]")
            lead_frags_new, used, lead_sets_new = get_fragments_brics(
                lead_mols_total, output_path+"/lead_frags.csv"
            )
        else:
            print("Using DeepFMPO fragmentation with rare element attachments")
            lead_frags_new, used, lead_sets_new = mol_utils.get_fragments(
                lead_mols_total, output_path+"/lead_frags.csv", use_brics=False
            )
        
        # Log unfragmentable molecules
        unfragmentable = log_unfragmentable_molecules(lead_mols_total, used, output_path)
        
        lead_mols = []
        for i, lead in enumerate(lead_mols_total):
            if used[i] == True:
                lead_mols.append(lead)
        print("{} out of {} lead molecules have been fragmented, resulting in {} lead fragment sets".format(
              len(lead_mols), len(lead_mols_total), len(lead_sets_new)))
    else:
        frag_var=2
        #raise Exception("No files with fragments given!")
    if frag_var==2:
        raise Exception("No files with fragments or leads given!")
    # we don't know how to handle duplicates, so for now, just add leads from two sources
    
    if eval_file is not None:
        eval_mols = file_io.read_molfile(eval_file,file_type='csv',as_smiles=False)
        if use_reinvent:
            eval_frags_new, used, eval_sets_new = get_fragments_reinvent(eval_mols, output_path+"/eval_frags.csv")
        elif use_brics:
            eval_frags_new, used, eval_sets_new = get_fragments_brics(eval_mols, output_path+"/eval_frags.csv")
        else:
            eval_frags_new, used, eval_sets_new = mol_utils.get_fragments(eval_mols, output_path+"/eval_frags.csv", use_brics=False)
    else:
        eval_frags_new = {}
        eval_sets_new = []
    
    if train_file is not None:
        train_mols = file_io.read_molfile(train_file,file_type='csv',as_smiles=False)
        if use_reinvent:
            train_frags_new, used, train_sets_new = get_fragments_reinvent(train_mols, output_path+"/train_frags.csv")
        elif use_brics:
            train_frags_new, used, train_sets_new = get_fragments_brics(train_mols, output_path+"/train_frags.csv")
        else:
            train_frags_new, used, train_sets_new = mol_utils.get_fragments(train_mols, output_path+"/train_frags.csv", use_brics=False)
    else:
        train_frags_new = {}
        train_sets_new = []


    print("Lead frags before adding new ones: {}".format(len(lead_frags)))
    lead_frags.update(lead_frags_new)  # add new fragments from lead file
    lead_frags.update(train_frags_new)  # add new fragments from train file
    lead_frags.update(eval_frags_new)   # add new fragments from eval file
    print("Lead frags after adding all sources: {}".format(len(lead_frags)))
    # lead_sets.update(lead_sets_new)  # add new sets to the existing
    lead_smiles_total = [Chem.MolToSmiles(mol) for mol in lead_mols_total]
    
    #print("These are duplicate leads")
    duplicate_smiles = [smi for smi in lead_smiles_total if lead_smiles_total.count(smi) > 1]
    #print(duplicate_smiles)
    #print(len(set(lead_smiles_total)))
    # create decodings and encode freezed fragments
    encodings, decodings = build_encoding.create_decodings(lead_frags)
    print('Encodings/decodings created. {} encodings, {} decodings'.format(len(encodings), len(decodings)), flush=True)
    build_encoding.save_decodings(decodings,filename=output_path+write_decodings)
    
    # Identify fragment codes containing frozen substructures
    freeze_encodings = build_encoding.encode_freeze(decodings)
    if freeze_encodings:
        print('Freeze encodings: {} fragment codes are frozen'.format(len(freeze_encodings)), flush=True)
    more_output=True
    # optional: write lead molecules and fragments (+ their encodings) into file
    if more_output:
        #file_io.create_folder("History")
        lfs_mols = [l[0] for l in lead_frags.values()]  # Always Mol objects now
        lead_smiles = [Chem.MolToSmiles(lf) for lf in lfs_mols]
        lead_keys = [encodings[smi] for smi in lead_smiles]
        print('Saving lead fragments and their encodings to file: {}'.format(output_path+"/lead_frags.smi"), flush=True)
        file_io.smiles_to_smi(lead_smiles, output_path+"/lead_frags.smi", lead_keys)
        print('Lead fragments saved.', flush=True)
        #visualization.mols_to_svg(lead_mols, filename=output_path+"/History/leads.svg")
        #visualization.mols_to_svg(lfs_mols, filename=output_path+"/History/lead_frags.svg", namelist=lead_keys)
    
    # if only decodings are written: exit program here
    #if (gl.PARAMS["WRITE_DECODINGS"]): exit()                                       
        
    # encode lead molecules
    max_frag = max([len(lead_set) for lead_set in lead_sets_new])  # maximum number of fragments in a lead molecule
    num_bits = len(list(decodings.keys())[0])  # this is only encoding for fragment (not this first bit that will be added later)
    print("Size of encodings: {} fragments with {} bits per fragment".format(max_frag, num_bits), flush=True) 
    lead_codes = build_encoding.encode_frags(lead_sets_new, encodings, max_frag)
    print('Lead codes encoded: {}'.format(lead_codes.shape), flush=True)
    train_codes = build_encoding.encode_frags(train_sets_new, encodings, max_frag)
    print('Train codes encoded: {}'.format(train_codes.shape), flush=True)
    eval_codes = build_encoding.encode_frags(eval_sets_new, encodings, max_frag)
    print('Eval codes encoded: {}'.format(eval_codes.shape), flush=True)
    np.save(output_path+'/leadmols.npy',lead_codes)
    # if there are no more lead molecules left nothing can be done
    if len(lead_codes) == 0:
        raise Exception("Sorry, there are no lead molecules left. Leaving program...")

    true_smiles = [Chem.MolToSmiles(mol) for mol in lead_mols]

    #make a dictionary of lead codes and true smiles
    lead_codes_dict = {Chem.MolToSmiles(mol): code for mol, code in zip(lead_mols, lead_codes)}
    if return_mode ==True:
        
        return lead_codes, train_codes,eval_codes,decodings, lead_codes_dict,max_frag,num_bits,freeze_encodings

if __name__ == '__main__':
    starter='Hello'
    print('Starting')
    encoding_maker(starter)
