
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
import Levenshtein
from rdkit.Chem import rdFMCS
from encoders.fmpo_utils.similarity import calculateMCStanimoto
import encoders.fmpo_utils.global_parameters as gl
# ---- Your existing pieces ----
# gl.PARAMS["ETA"] used in Levenshtein-based similarity
# Levenshtein.distance(smi1, smi2) is assumed available (C extension is fast-ish)


def upper_bound_mcs_tanimoto(mol1, mol2):
    n1 = mol1.GetNumAtoms()
    n2 = mol2.GetNumAtoms()
    return min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0.0

def upper_bound_lev_sim(s1, s2, eta):
    return 1.0 - eta * abs(len(s1) - len(s2))

def exact_similarity(smi1, smi2, mol1, mol2, eta):
    # Your exact definition: max(edit-sim, MCS tanimoto)
    d1 = 1.0 - eta * Levenshtein.distance(smi1, smi2)
    mcs_res = rdFMCS.FindMCS(
        [mol1, mol2],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchValences=True
    )
    numCommonAtoms = mcs_res.numAtoms
    n1 = float(mol1.GetNumAtoms())
    n2 = float(mol2.GetNumAtoms())
    d2 = numCommonAtoms / ((n1 + n2) - numCommonAtoms) if (n1 + n2) > numCommonAtoms else 0.0
    return max(d1, d2)

def fast_similarity_with_prescreen(smi1, smi2, mol1, mol2, fp1, fp2, eta, tau, tau_fp):
    # Upper bounds
    
    ub_mcs = upper_bound_mcs_tanimoto(mol1, mol2)
    ub_lev = upper_bound_lev_sim(smi1, smi2, eta)
    ub = max(ub_mcs, ub_lev)

    # Quick fingerprint tanimoto (proxy)
    fp_tan = DataStructs.TanimotoSimilarity(fp1, fp2)

    # If none can possibly exceed tau, return the best cheap proxy (or ub)
    if ub < tau and fp_tan < tau_fp:
        # Choose what you prefer to return here; fp_tan is a good proxy.
        return fp_tan

    # Otherwise compute exact similarity, but with early exit: try cheap Levenshtein first
    d1 = 1.0 - eta * Levenshtein.distance(smi1, smi2)
    if d1 >= tau:
        return d1  # no need to run FMCS
    else:
        d2 = calculateMCStanimoto(mol1, mol2)[0]
        return max(d1, d2)
       # Only then run FMCS (expensive) if it's still potentially useful
