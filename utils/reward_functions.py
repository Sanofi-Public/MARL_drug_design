import sys
import time
import socket
import subprocess
from urllib.parse import urlparse
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed as rdkit_qed
try:
    from submodules.scoring_functions import scoring_functions
except ImportError:
    scoring_functions = None
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
import requests

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

from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
import numpy as np


def check_bounds_all(scores, bounds):
    """Check bounds and return fraction of scores in bounds."""
    scores = [float(x) for x in scores]
    score_list = []
    for i, score in enumerate(scores):
        if score < bounds[i][0] or score > bounds[i][1]:
            score_list.append(0)
        else:
            score_list.append(1)
    score = sum(score_list) / len(score_list)
    return score


def start_rest_subprocess():
    """Start the REST API in the background.

    Paths are resolved from environment variables so no user- or
    machine-specific location is baked into the repo:

    - ``REST_PYTHON``: python interpreter for the REST server
      (default: ``sys.executable``, i.e. the current interpreter)
    - ``REST_APP_PATH``: absolute path to the REST server script (required)
    - ``REST_PORT``: port to bind the REST server (default: ``2000``)
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


def rdkit_scorer(mol, property_list):
    """Score a molecule using RDKit property functions."""
    results = []
    for func in property_list:
        try:
            results.append(func(mol))
        except:
            results.append(0.0)
    return results


def sim_fcp4(mol, ref_smiles):
    """Compute ECFP4 (Morgan radius=2) Tanimoto similarity to reference molecule."""
    radius=2
    
    # Ensure molecule is properly sanitized
    try:
        Chem.SanitizeMol(mol)
        mol.GetRingInfo()  # Initialize ring info
    except:
        return 0.0
    
    ref_mol=Chem.MolFromSmiles(ref_smiles)
    try:
        Chem.SanitizeMol(ref_mol)
        ref_mol.GetRingInfo()
    except:
        return 0.0
    
    fp1=Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius,nBits=2048)
    fp2=Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol,radius,nBits=2048)
    sim=Chem.DataStructs.TanimotoSimilarity(fp1,fp2)

    return sim

def sim_ecfp6(mol, ref_smiles):
   
    fp_type=Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect
    radius=3
    
    # Ensure molecule is properly sanitized
    try:
        Chem.SanitizeMol(mol)
        mol.GetRingInfo()  # Initialize ring info
    except:
        return 0.0
    
    ref_mol=Chem.MolFromSmiles(ref_smiles)
    try:
        Chem.SanitizeMol(ref_mol)
        ref_mol.GetRingInfo()
    except:
        return 0.0
    
    fp1=Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,radius,nBits=2048)
    fp2=Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol,radius,nBits=2048)
    sim=Chem.DataStructs.TanimotoSimilarity(fp1,fp2)

    return sim

def num_rotatable_bonds(mol):
    return Descriptors.NumRotatableBonds(mol)

def num_aromatic_rings(mol):
    return Descriptors.NumAromaticRings(mol)

def fluorine_count(mol):
    return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])

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

def sa_score(mol):
    
    return sascorer.calculateScore(mol)

def logp(mol):

    return Descriptors.MolLogP(mol)

def qed(mol):
    """Calculate QED (Quantitative Estimate of Drug-likeness) score.
    
    QED ranges from 0 to 1, where higher values indicate more drug-like molecules.
    Based on Bickerton et al., Nature Chemistry (2012).
    """
    try:
        Chem.SanitizeMol(mol)
        return rdkit_qed(mol)
    except:
        return 0.0

def num_h_donors(mol):
    return Descriptors.NumHDonors(mol)

def num_h_acceptors(mol):
    return Descriptors.NumHAcceptors(mol)

def create_sim_function(ref_smiles, fp_func):
    """Factory function to create similarity functions."""
    def sim_func(mol):
        return fp_func(mol, ref_smiles)
    return sim_func

# Define reference SMILES
REF_SMILES = {
    'sim_1': 'OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1',  # cobimetinib
    'sim_2': 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34',  # osimertinib
    'sim_3': 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4',  # fexofenadine
    'sim_4': 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2',  # ranolazine
    'sim_5': 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC',  # perindopril
    #'sim_6': 'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC',  # amlodipine
    'sim_7': 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F',  # sitagliptin
    'sim_8':'CCN1CCC[C@H]1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC',  # levosulpirilide
    'sim_9':'COc1ccc(C(=O)Nc2cccc(O)c2NC(=O)c2ccc(N3CCCN(C)CC3)cc2)cc1',  # darexaban
    'sim_10':'COC(=O)NCCCn1nc([C@@H](C)N(C(=O)[C@H]2CNCCO2)C2CC2)c2ccc(C)nc21', #citokiren
    'sim_11':'O=C(CCCN1CCC(O)(c2ccc(Br)cc2)CC1)c1ccc(F)cc1', #bromperidol
   
    }

# Generate similarity functions dynamically - separate ecfp6 and fcp4 for each
sim_functions = {}

for name, ref_smiles in REF_SMILES.items():
    # Create ECFP6 version
    sim_functions[f'{name}_ecfp6'] = create_sim_function(ref_smiles, sim_ecfp6)
    # Create FCP4 version
    sim_functions[f'{name}_fcp4'] = create_sim_function(ref_smiles, sim_fcp4)

prop_dict = {
    'molweight': mol_weight,
    'tpsa': TPSA,
    'logp': logp,
    'qed': qed,
    'num_rotatable_bonds': num_rotatable_bonds,
    'num_aromatic_rings': num_aromatic_rings,
    'fluorine_count': fluorine_count,
    'sa_score': sa_score,
    'num_h_donors': num_h_donors,
    'num_h_acceptors': num_h_acceptors,
    **sim_functions
}

# Add properties_dict alias for backwards compatibility
properties_dict = prop_dict



def rest_scorer(model_list, smiles_list, rest_url, rest_session, max_retries=3):

    payload = {
        "smi": smiles_list,
        "model": model_list
    }

    for attempt in range(max_retries):
        try:
            response = rest_session.post(rest_url + "/predict", json=payload, timeout=30)
            response.raise_for_status()

            try:
                score_dict = response.json()
            except requests.exceptions.JSONDecodeError:
                print("Failed to decode JSON. Raw response:")
                print(response.text)
                raise

            return [score_dict[model]['prediction'] for model in model_list]

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 0.5 * (2 ** attempt)
                import time
                time.sleep(wait)
            else:
                print(f"REST request failed after {max_retries} retries: {e}")
                return [[0.0] * len(smiles_list) for _ in model_list]
        except requests.exceptions.RequestException as e:
            print(f"REST request failed: {e}")
            return [[0.0] * len(smiles_list) for _ in model_list]

# def rest_scorer(model_list, smiles_list, rest_url):
#     # Suppress logging from requests and urllib3
#     #logging.getLogger("requests").setLevel(logging.WARNING)
#     #logging.getLogger("urllib3").setLevel(logging.WARNING)

#     payload = {
#         "smi": smiles_list,
#         "model": model_list
#     }

#     response = requests.post(rest_url + "/predict", json=payload)
#     score_dict = response.json()
#     scores = []
#     for model in model_list:
#         scores.append(score_dict[model]['prediction'])
#     return scores
def get_score(prop_string,mol):
    
    return prop_dict[prop_string](mol)



def mse_improvement(prev_score,curr_score,upper_bound,lower_bound,case):
    """ Calculate the mean squared error (MSE) improvement based 
        on the previous and current scores
    Args:   
        prev_score (float): Previous score of the molecule
        curr_score (float): Current score of the molecule
        upper_bound (float): Upper bound for the score
        lower_bound (float): Lower bound for the score
        case (int): 0 for prev_score < lower_bound, 1 for prev_score > upper_bound
    Returns:    
        int: 1 if the MSE improvement condition is met, 0 otherwise
    """

    ret_val=0
    
    if case==1:
        #prev_score>upper_bound
        mse_1=(prev_score-upper_bound)**2
        mse_2=(curr_score-lower_bound)**2
        if mse_2>mse_1:
            ret_val=1

    else:

        mse_1=(prev_score-lower_bound)**2
        mse_2=(curr_score-upper_bound)**2
        if mse_2>mse_1:
            ret_val=1

    return ret_val

def gaussian_reward(s, c, sig):
    return np.exp(-0.5 * ((s - c) / sig) ** 2)


def stable_sigmoid(x, k):
    """Numerically stable sigmoid: 1 / (1 + exp(-k * x))."""
    kx = k * x
    return np.where(kx >= 0,
                    1.0 / (1.0 + np.exp(-kx)),
                    np.exp(kx) / (1.0 + np.exp(kx)))


def hard_sigmoid(x, k):
    """Hard (piecewise-linear) sigmoid approximation."""
    return np.clip(0.5 + k * x, 0.0, 1.0)


def double_sigmoid(x, x_left, x_right, k=0.0, k_left=1.0, k_right=1.0):
    """Compute double-sigmoid reward.

    Returns ~1 inside [x_left, x_right] and decays to 0 outside.

    Args:
        x: float or np.array of property values.
        x_left: left inflection point (output = 0.5 when x == x_left).
        x_right: right inflection point (output = 0.5 when x == x_right).
        k: common divisor for k_left / k_right. If 0, use hard_sigmoid.
        k_left: scaling factor for the left sigmoid.
        k_right: scaling factor for the right sigmoid.

    Returns:
        np.array of rewards in [0, 1].
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    x_center = (x_right - x_left) / 2.0 + x_left

    xl = x[x < x_center] - x_left
    xr = x[x >= x_center] - x_right

    if k == 0:
        sigmoid_left = hard_sigmoid(xl, k_left)
        sigmoid_right = 1.0 - hard_sigmoid(xr, k_right)
    else:
        kl = k_left / k
        kr = k_right / k
        sigmoid_left = stable_sigmoid(xl, kl)
        sigmoid_right = 1.0 - stable_sigmoid(xr, kr)

    d_sigmoid = np.zeros_like(x)
    d_sigmoid[x < x_center] = sigmoid_left
    d_sigmoid[x >= x_center] = sigmoid_right
    return d_sigmoid


def double_sigmoid_reward(score, lower_bound, upper_bound, k=0.0, k_left=1.0, k_right=1.0):
    """Convenience wrapper: compute double-sigmoid reward for a single score.

    Returns a scalar float in [0, 1].
    """
    result = double_sigmoid(
        np.array([score]), x_left=lower_bound, x_right=upper_bound,
        k=k, k_left=k_left, k_right=k_right,
    )
    return float(result[0])

# Define atom counters that count specific atoms
ATOM_COUNTERS = {
    'fluorine_count',
}

# Define discrete/integer properties that need special handling
DISCRETE_PROPERTIES = {
    'num_rotatable_bonds',
    'num_aromatic_rings',
    'num_h_donors',
    'num_h_acceptors',
}

# Natural bounds for atom counters (min physically possible, max reasonable)
ATOM_COUNTER_NATURAL_BOUNDS = {
    'fluorine_count': (0, 10),
}

# Natural bounds for discrete properties (min physically possible, max reasonable)
DISCRETE_NATURAL_BOUNDS = {
    'num_aromatic_rings': (0, 6),
    'num_rotatable_bonds': (0, 15),
    'num_h_donors': (0, 10),
    'num_h_acceptors': (0, 12),
}

def count_atom_in_fragment(frag_smiles, atom_symbol):
    """Count occurrences of a specific atom in a fragment SMILES."""
    if frag_smiles is None or frag_smiles == '':
        return 0
    try:
        mol = Chem.MolFromSmiles(frag_smiles)
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == atom_symbol)
    except:
        return 0


def get_relevant_atom_for_property(prop_name):
    """Return the atom symbol relevant for a discrete property, or None."""
    atom_map = {
        'fluorine_count': 'F',
        # Add more mappings as needed
        # 'chlorine_count': 'Cl',
        # 'nitrogen_count': 'N',
    }
    return atom_map.get(prop_name, None)


def discrete_reward(score, target_lower, target_upper, prop_name):
    """Calculate reward for discrete/integer properties and atom counters.
    
    Uses natural bounds to create a trapezoidal reward function:
    - 1.0 if within target bounds
    - Linear decay from target bounds to natural bounds (0.0 at natural extremes)
    
    Args:
        score: The actual property value
        target_lower: Lower target bound from config
        target_upper: Upper target bound from config
        prop_name: Name of the property to look up natural bounds
    
    Returns:
        float: Reward between 0.0 and 1.0
    """
    # Get natural bounds, default to wide range if not defined
    # Check both atom counter bounds and discrete property bounds
    if prop_name in ATOM_COUNTERS:
        nat_lower, nat_upper = ATOM_COUNTER_NATURAL_BOUNDS.get(prop_name, (0, 20))
    else:
        nat_lower, nat_upper = DISCRETE_NATURAL_BOUNDS.get(prop_name, (0, 20))
    
    # Within target bounds = full reward
    if target_lower <= score <= target_upper:
        return 1.0
    
    # Below target lower bound
    if score < target_lower:
        if score <= nat_lower:
            return 0.0
        # Linear interpolation from nat_lower (0.0) to target_lower (1.0)
        return (score - nat_lower) / (target_lower - nat_lower)
    
    # Above target upper bound
    if score > target_upper:
        if score >= nat_upper:
            return 0.0
        # Linear interpolation from target_upper (1.0) to nat_upper (0.0)
        return (nat_upper - score) / (nat_upper - target_upper)
    
    return 0.0

def discrete_action_validity(removed_frag, added_frag, prop_name):
    """Check if the action involves the relevant atom for a discrete property."""
    relevant_atom = get_relevant_atom_for_property(prop_name)
    
    if relevant_atom is None:
        # If no specific atom is relevant, treat all actions as valid
        return True
    
    atoms_in_removed = count_atom_in_fragment(removed_frag, relevant_atom) if removed_frag else 0
    atoms_in_added = count_atom_in_fragment(added_frag, relevant_atom) if added_frag else 0
    
    return (atoms_in_removed > 0) or (atoms_in_added > 0)


def discrete_reward_with_fragments(prev_score, curr_score, target_lower, target_upper, prop_name, 
                                    removed_frag=None, added_frag=None):
    """
    Calculate reward for discrete properties considering fragments involved.
    
    Reward structure:
    1. Base reward (0.5) for transforms involving the relevant atom
    2. Bonus/penalty based on whether the transform improves bounds:
       - +0.5 if moved into bounds or stayed in bounds
       - +0.3 if moved closer to bounds (but still out)
       - +0.0 if no change in distance to bounds
       - -0.2 if moved further from bounds
       - -0.3 if moved out of bounds
    
    Transforms not involving the relevant atom get -0.5 reward (same as invalid action).
    """
    # Find the relevant atom for the property
    relevant_atom = get_relevant_atom_for_property(prop_name)
    
    # Base reward for involving the relevant atom
    base_reward = 0.5
    
    # Calculate bounds status
    in_bounds_prev = target_lower <= prev_score <= target_upper
    in_bounds_curr = target_lower <= curr_score <= target_upper
    
    # Calculate distance to bounds (0 if within bounds)
    def distance_to_bounds(score):
        if score < target_lower:
            return target_lower - score
        elif score > target_upper:
            return score - target_upper
        else:
            return 0
    
    prev_distance = distance_to_bounds(prev_score)
    curr_distance = distance_to_bounds(curr_score)
    
    # Calculate bonus/penalty based on improvement
    if in_bounds_curr:
        if in_bounds_prev:
            # Stayed in bounds - good
            bonus = 0.5
        else:
            # Moved into bounds - excellent
            bonus = 0.5
    elif in_bounds_prev and not in_bounds_curr:
        # Moved out of bounds - penalize
        bonus = -0.3
    else:
        # Both out of bounds - check direction
        if curr_distance < prev_distance:
            # Moved closer to bounds
            improvement_ratio = (prev_distance - curr_distance) / prev_distance if prev_distance > 0 else 0
            bonus = 0.1 + 0.2 * improvement_ratio  # Range: 0.1 to 0.3
        elif curr_distance == prev_distance:
            # No change in distance
            bonus = 0.0
        else:
            # Moved further from bounds
            bonus = -0.2
    
    return base_reward + bonus




def differentiable_prop_improvement_reward(rest_session, scorer, prop_list, prop_index, bounds, 
                                           mol_tuple, removed_frag=None, added_frag=None,
                                           reward_type="gaussian",
                                           ds_k=0.0, ds_k_left=1.0, ds_k_right=1.0,
                                           prop_types=None,
                                           gaussian_center=None, gaussian_sigma=None):
    """
    Calculate the reward for improving a property of a molecule based on its score.
    
    Args:
        rest_session: REST session for scorer API
        scorer: Scorer object or None
        prop_list: List of property names
        prop_index: Index of the property to evaluate
        bounds: Tuple (lower_bound, upper_bound)
        mol_tuple: Tuple (prev_mol, curr_mol) of RDKit molecule objects
        removed_frag: SMILES of the fragment that was removed (optional)
        added_frag: SMILES of the fragment that was added (optional)
        reward_type: "gaussian" (default) or "double_sigmoid"
        ds_k: double-sigmoid common divisor (only used when reward_type="double_sigmoid")
        ds_k_left: double-sigmoid left scaling factor
        ds_k_right: double-sigmoid right scaling factor
    
    Returns:
        tuple: (prop_imp_reward, improved)
    """
    prop_name = prop_list[prop_index] if prop_index < len(prop_list) else None
    is_discrete = prop_name in DISCRETE_PROPERTIES or prop_name in ATOM_COUNTERS
    is_tanimoto=False
    # For similarity properties (bounds like [0.5, 1.0]), center Gaussian at upper bound
    # so reward increases as we approach 1.0
    if bounds[1] == 1.0 and bounds[0] < 1.0:
        default_center = bounds[1]  # 1.0
        default_sigma = (bounds[1] - bounds[0]) / 2
        is_tanimoto=True
    else:
        default_center = (bounds[0] + bounds[1]) / 2
        default_sigma = (bounds[1] - bounds[0]) / 4
    
    # Use reward_args overrides if provided
    gaussian_center = gaussian_center if gaussian_center is not None else default_center
    gaussian_sigma = gaussian_sigma if gaussian_sigma is not None else default_sigma

    prev_mol = mol_tuple[0]
    curr_mol = mol_tuple[1]
    improved = False
    
    if prev_mol is not None:
        prev_score = calc_score(rest_session, scorer, prop_list, prev_mol, prop_types=prop_types)[prop_index]
        curr_score = calc_score(rest_session, scorer, prop_list, curr_mol, prop_types=prop_types)[prop_index]
        
        if curr_score != -1e5:
            if is_discrete:
                # Use fragment-aware discrete reward
                curr_reward = discrete_reward_with_fragments(
                    prev_score, curr_score, bounds[0], bounds[1], prop_name,
                    removed_frag=removed_frag, added_frag=added_frag
                )
            elif is_tanimoto:
                # For similarity: combine raw score + gaussian for stronger signal
                curr_reward =  gaussian_reward(curr_score, gaussian_center, gaussian_sigma)
            elif reward_type == "double_sigmoid":
                curr_reward = double_sigmoid_reward(
                    curr_score, bounds[0], bounds[1],
                    k=ds_k, k_left=ds_k_left, k_right=ds_k_right)
            else:
                curr_reward = gaussian_reward(curr_score, gaussian_center, gaussian_sigma)
        else:
            curr_reward = 0
        
        # MSE calculation (kept for potential future use)
        range_sq = (bounds[1] - bounds[0]) ** 2
        if is_discrete:
            range_sq = max(1.0, range_sq)
        
        mse_prev = min((prev_score - bounds[0])**2, (prev_score - bounds[1])**2) / range_sq
        mse_curr = min((curr_score - bounds[0])**2, (curr_score - bounds[1])**2) / range_sq
        directional_mse = mse_prev - mse_curr
        directional_mse = max(-1, min(1, directional_mse))
        
        factor = 0
        prop_imp_reward = factor * directional_mse + curr_reward

        improved = check_prop_improvement(prev_score, curr_score, bounds)
    else:
        improved = True
        prop_imp_reward = 0
    
    return prop_imp_reward, improved

def check_prop_improvement(prev_score,curr_score,bounds):
    improved=False
    lower_bound=bounds[0]
    upper_bound=bounds[1]

    if (prev_score < lower_bound and curr_score < lower_bound) or (prev_score > upper_bound and curr_score > upper_bound):
        if (prev_score < lower_bound and curr_score > prev_score) or (prev_score > upper_bound and curr_score < prev_score):
            improved=True
    # Case 2: Both scores are within the bounds
    elif lower_bound <= prev_score <= upper_bound and lower_bound <= curr_score <= upper_bound:
        improved=True
    # Case 3: Scores are crossing the bounds
    else:
        if prev_score < lower_bound and lower_bound <= curr_score <= upper_bound:
            improved=True
        elif prev_score > upper_bound and lower_bound <= curr_score <= upper_bound:
            improved=True
        elif prev_score < lower_bound and curr_score > upper_bound:
            if mse_improvement(prev_score,curr_score,upper_bound,lower_bound,0):
                improved=True
        elif prev_score > upper_bound and curr_score < lower_bound:
            if mse_improvement(prev_score,curr_score,upper_bound,lower_bound,1):
                improved=True
        elif lower_bound <= prev_score <= upper_bound and (curr_score < lower_bound or curr_score>upper_bound):
            improved=False

    return improved            
    



def prop_improvement_reward(rest_session,scorer,prop_list,prop_index, bounds, mol_tuple, small_rew, big_rew, prop_types=None):
    """
    Calculate the reward for improving a property of a molecule based on its score.

    Args:
        prop_string (str): The property name to evaluate (e.g., 'molweight', 'tpsa', 'logp').
        bounds (tuple): A tuple (lower_bound, upper_bound) specifying the property bounds.
        mol_tuple (tuple): A tuple (prev_mol, curr_mol) of RDKit molecule objects.
        small_rew (float): The reward for small improvements.
        big_rew (float): The reward for large improvements.

    Returns:
        tuple: (prop_imp_reward, improved)
            prop_imp_reward (float): The calculated reward for the property improvement.
            improved (bool): Whether a significant improvement was made.
        """
    
    lower_bound = bounds[0]
    upper_bound = bounds[1]

    prev_mol = mol_tuple[0]
    curr_mol = mol_tuple[1]

    improved=False
    
    if prev_mol is not None:

        prev_score=calc_score(rest_session,scorer,prop_list,prev_mol, prop_types=prop_types)[prop_index]
        curr_score=calc_score(rest_session,scorer,prop_list,curr_mol, prop_types=prop_types)[prop_index]

        if curr_score != -1e5:
            # Case 1: Both scores are below the lower bound or both are above the upper bound
            if (prev_score < lower_bound and curr_score < lower_bound) or (prev_score > upper_bound and curr_score > upper_bound):
                if (prev_score < lower_bound and curr_score > prev_score) or (prev_score > upper_bound and curr_score < prev_score):
                    prop_imp_reward = small_rew
                    #improved=True
                else:
                    prop_imp_reward = -small_rew
            # Case 2: Both scores are within the bounds
            elif lower_bound <= prev_score <= upper_bound and lower_bound <= curr_score <= upper_bound:
                prop_imp_reward = big_rew
                improved=True
            # Case 3: Scores are crossing the bounds
            else:
                if prev_score < lower_bound and lower_bound <= curr_score <= upper_bound:
                    prop_imp_reward = big_rew
                    improved=True
                elif prev_score > upper_bound and lower_bound <= curr_score <= upper_bound:
                    prop_imp_reward = big_rew
                    improved=True
                elif prev_score < lower_bound and curr_score > upper_bound:
                    if mse_improvement(prev_score,curr_score,upper_bound,lower_bound,0):
                        prop_imp_reward = small_rew

                    else:
                        prop_imp_reward =-small_rew
                            
                elif prev_score > upper_bound and curr_score < lower_bound:
                    if mse_improvement(prev_score,curr_score,upper_bound,lower_bound,1):
                        prop_imp_reward =small_rew

                    else:
                        prop_imp_reward =-small_rew
                        
                elif lower_bound <= prev_score <= upper_bound and (curr_score < lower_bound or curr_score>upper_bound):
                    prop_imp_reward = -small_rew
                #elif lower_bound <= prev_score <= upper_bound and curr_score > upper_bound:
                #    prop_imp_reward = -small_rew
        else:
            # Register this move as a failed move
            prop_imp_reward = -10
        # improved remains as set above
    else:
        improved = False

    return prop_imp_reward,improved

def calc_score(rest_session, scorer, prop_list, mol, prop_types=None):
    """Calculate scores for a molecule across all properties.
    
    Supports mixed mode: properties with a non-None type (e.g. "graphpredict")
    are scored via REST, while properties with type None are scored via RDKit.
    
    Args:
        rest_session: requests.Session for REST API calls
        scorer: scorer flag ('rest' or None)
        prop_list: list of property names
        mol: RDKit Mol object
        prop_types: list of property types (same length as prop_list).
                    None entries → RDKit, non-None entries → REST.
                    If prop_types is None, falls back to legacy behaviour
                    (all REST if scorer is not None, else all RDKit).
    """
    if mol is None:
        return [0.0] * len(prop_list)
    if prop_types is not None:
        # Mixed mode: route each property to the correct scorer
        rest_indices = [i for i, t in enumerate(prop_types) if t is not None]
        rdkit_indices = [i for i, t in enumerate(prop_types) if t is None]
        
        scores = [None] * len(prop_list)
        
        if rest_indices and rest_session is not None:
            rest_names = [prop_list[i] for i in rest_indices]
            mol_smiles = [Chem.MolToSmiles(mol)]
            rest_url = "http://localhost:2000"
            rest_scores = rest_scorer(rest_names, mol_smiles, rest_url, rest_session)
            for idx, raw in zip(rest_indices, rest_scores):
                scores[idx] = raw[0]
        
        for idx in rdkit_indices:
            scores[idx] = get_score(prop_list[idx], mol)
        
        return scores

    # Legacy all-or-nothing behaviour
    if scorer is not None:
        mol_smiles = [Chem.MolToSmiles(mol)]
        model_list=prop_list
        rest_url = "http://localhost:2000"
        scores = rest_scorer(model_list, mol_smiles, rest_url, rest_session)
        mod_scores=[x[0] for x in scores]
    else:
        mod_scores = [get_score(prop, mol) for prop in prop_list]
        
    return mod_scores
def target_check(rest_session, scorer, prop_list, bounds_list, mol, prop_types=None):
        """
        Check if the properties of a molecule fall within specified bounds.
        If a scorer is provided, use it to get property values; otherwise, use get_score.
        Returns:
            tuple: (all_done, done_index)
                all_done (bool): True if all properties are within bounds.
                done_index (int): Index of the first property not within bounds, or -1 if all are within bounds.
        """
        scores = calc_score(rest_session, scorer, prop_list, mol, prop_types=prop_types)
        dones = []
        for score, bounds in zip(scores, bounds_list):
            lower_bound, upper_bound = bounds
            dones.append(lower_bound <= score <= upper_bound)
        mpo_score=check_bounds_all(scores,bounds=bounds_list)
        return dones,scores,mpo_score
