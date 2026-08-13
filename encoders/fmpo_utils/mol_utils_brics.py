"""
BRICS Molecule Utilities

Fragment-based molecule operations using BRICS approach:
- BRICS bond rules (retrosynthetically interesting cuts)
- Typed attachment points [1*], [3*], [7*], etc. (BRICS environment types L1-L16)
- Uses RDKit's native BRICS.BRICSBuild for reconstruction

This module handles:
- split_molecule_brics_typed() - Fragment molecule using BRICS rules with typed attachments
- join_fragments_brics() - Reconstruct molecule using BRICS.BRICSBuild
- get_fragments_brics() - Extract fragment library from molecules

All functions use Mol objects consistently (same as DeepFMPO).
"""

import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, RWMol, CombineMols
import encoders.fmpo_utils.global_parameters as gl
from encoders.fmpo_utils.brics_utils import BRICS_COMPATIBILITY


# ═══════════════════════════════════════════════════════════════════════════════
# SPLITTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def split_molecule_brics_typed(mol):
    """
    Split molecule using BRICS rules with typed attachment points.
    
    Uses FindBRICSBonds + FragmentOnBonds to cut all BRICS bonds and preserve
    the BRICS environment type labels ([1*], [3*], [7*], etc.).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        List of RDKit Mol objects with BRICS-typed attachments
    """
    avoid_ring_cuts = True
    
    # Find frozen fragment indices
    freeze_indices = []
    for ff_smi in gl.PARAMS["FREEZE_FRAGS"]:
        ff = Chem.MolFromSmiles(ff_smi)
        idxs = mol.GetSubstructMatch(ff)
        if len(idxs) != 0:
            freeze_indices.append(idxs)
    
    def is_frozen_pair(a1, a2):
        for fr in freeze_indices:
            if a1 in fr and a2 in fr:
                return True
        return False

    cuts = []
    brics_labels = {}  # Store BRICS environment labels for each bond
    
    for (a1a2, labels) in BRICS.FindBRICSBonds(mol):
        a1, a2 = a1a2
        if is_frozen_pair(a1, a2):
            continue
        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is None:
            continue
        if avoid_ring_cuts and b.IsInRing():
            continue
        bond_key = tuple(sorted((a1, a2)))
        cuts.append(bond_key)
        # Store BRICS environment types (e.g., ('1', '3') means L1-L3 bond)
        brics_labels[bond_key] = labels
    
    if not cuts:
        # No BRICS bonds found, return whole molecule as single fragment
        return [mol]
    
    # Dedupe
    cuts = sorted(set(cuts))
    bond_indices = []
    bond_labels = []
    
    for b in cuts:
        bond_indices.append(mol.GetBondBetweenAtoms(b[0], b[1]).GetIdx())
        # Use actual BRICS environment types as dummy labels
        labels = brics_labels[b]
        bond_labels.append((int(labels[0]), int(labels[1])))
    
    # Fragment at these bonds with BRICS-typed attachment points
    frag_mol = Chem.FragmentOnBonds(mol, bond_indices, dummyLabels=bond_labels)
    smi_list = Chem.MolToSmiles(frag_mol).split(".")
    
    # Convert SMILES back to Mol objects
    fragments = []
    for smi in smi_list:
        frag = Chem.MolFromSmiles(smi)
        if frag is not None:
            fragments.append(frag)
    
    if not fragments:
        return [mol]
    
    return fragments


# ═══════════════════════════════════════════════════════════════════════════════
# JOINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def join_fragments_brics(fragments, debug=False):
    """
    Join BRICS-typed fragments into a molecule using RDKit's native BRICSBuild.
    
    Uses BRICS.BRICSBuild which handles all the compatibility rules automatically.
    After joining, adds implicit hydrogens to satisfy valences.
    
    Args:
        fragments: List of RDKit Mol objects with BRICS-typed attachments
        debug: Print debug info
        
    Returns:
        RDKit Mol object or None if joining fails
    """
    if not fragments:
        if debug: print("[BRICS JOIN] No fragments provided")
        return None
    
    if debug:
        print(f"[BRICS JOIN] Joining {len(fragments)} fragments:")
        for i, f in enumerate(fragments):
            smi = Chem.MolToSmiles(f)
            print(f"  Fragment {i}: {smi}")
    
    # Handle single fragment - just remove attachment points and add H
    if len(fragments) == 1:
        mol = fragments[0]
        emol = Chem.RWMol(mol)
        
        # Find atoms connected to dummy atoms (they'll need H added)
        atoms_needing_h = set()
        atoms_to_remove = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                atoms_to_remove.append(atom.GetIdx())
                # Get the neighbor that will lose a bond
                for neighbor in atom.GetNeighbors():
                    atoms_needing_h.add(neighbor.GetIdx())
        
        # Remove dummy atoms (highest index first)
        for idx in sorted(atoms_to_remove, reverse=True):
            emol.RemoveAtom(idx)
        
        try:
            # Sanitize and add implicit hydrogens
            Chem.SanitizeMol(emol)
            result = emol.GetMol()
            # Add Hs to satisfy valences, then remove to get implicit H representation
            result = Chem.AddHs(result)
            result = Chem.RemoveHs(result)
            return result
        except Exception as e:
            if debug: print(f"[BRICS JOIN] Single fragment sanitize failed: {e}")
            return None
    
    # Use deterministic pairwise joining instead of BRICSBuild enumeration.
    # We join fragments one at a time: result = frag[0] ⊕ frag[1] ⊕ frag[2] ...
    # Each pairwise join has exactly one free attachment on each side → deterministic.
    try:
        result = fragments[0]
        for i, next_frag in enumerate(fragments[1:], 1):
            joined = _join_two_brics(result, next_frag, debug)
            if joined is None:
                if debug:
                    print(f"[BRICS JOIN] Pairwise join failed at fragment {i}")
                    print(f"  current: {Chem.MolToSmiles(result)}")
                    print(f"  next:    {Chem.MolToSmiles(next_frag)}")
                return None
            result = joined
        
        try:
            Chem.SanitizeMol(result)
            result = Chem.AddHs(result)
            result = Chem.RemoveHs(result)
        except Exception:
            pass

        if debug:
            print(f"[BRICS JOIN] Success: {Chem.MolToSmiles(result)}")
        return result

    except Exception as e:
        if debug:
            print(f"[BRICS JOIN] Pairwise join failed: {e}")
        return None


def _join_two_brics(mol_a, mol_b, debug=False):
    """
    Join two BRICS fragments by connecting their free attachment points.

    Picks the first free attachment point on mol_a and the first compatible
    free attachment point on mol_b, forms a bond, and removes both dummies.
    Deterministic — no enumeration.

    Args:
        mol_a: RDKit Mol with one or more [n*] attachment points
        mol_b: RDKit Mol with one or more [n*] attachment points
        debug: Print debug info

    Returns:
        RDKit Mol with one bond formed between mol_a and mol_b, or None on failure
    """
    # Get all dummy atoms with their BRICS type (stored as isotope: [1*] has isotope=1)
    def _get_dummies(mol):
        """Returns list of (atom_idx, brics_type) for dummy atoms."""
        dummies = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummies.append((atom.GetIdx(), atom.GetIsotope()))
        return dummies

    dummies_a = _get_dummies(mol_a)
    dummies_b = _get_dummies(mol_b)

    if not dummies_a or not dummies_b:
        if debug: print("[BRICS JOIN] One fragment has no free attachment")
        return None

    # BRICS compatibility: types connect to their partner type (1↔2, 3↔4, 7↔7, etc.)
    # Find first compatible pair using the BRICS_COMPATIBILITY rules
    dummy_a_idx = None
    dummy_b_idx = None
    for (da_idx, da_type) in dummies_a:
        for (db_idx, db_type) in dummies_b:
            if db_type in BRICS_COMPATIBILITY.get(da_type, set()):
                dummy_a_idx = da_idx
                dummy_b_idx = db_idx
                break
        if dummy_a_idx is not None:
            break

    # Fallback: if no type match, just use first available (handles edge cases)
    if dummy_a_idx is None:
        dummy_a_idx = dummies_a[0][0]
        dummy_b_idx = dummies_b[0][0]
        if debug:
            types_a = [t for _, t in dummies_a]
            types_b = [t for _, t in dummies_b]
            print(f"[BRICS JOIN] No type match found (types_a={types_a}, types_b={types_b}), using first")

    combined = CombineMols(mol_a, mol_b)
    emol = RWMol(combined)

    # After combining, dummy_b index is offset by len(mol_a.GetAtoms())
    offset = mol_a.GetNumAtoms()
    dummy_b_combined = dummy_b_idx + offset

    # Get the real neighbors of both dummies
    try:
        neigh_a = [n.GetIdx() for n in emol.GetAtomWithIdx(dummy_a_idx).GetNeighbors()]
        neigh_b = [n.GetIdx() for n in emol.GetAtomWithIdx(dummy_b_combined).GetNeighbors()]
    except Exception as e:
        if debug: print(f"[BRICS JOIN] Neighbor lookup failed: {e}")
        return None

    if not neigh_a or not neigh_b:
        if debug: print("[BRICS JOIN] Dummy atom has no neighbor")
        return None

    real_a = neigh_a[0]
    real_b = neigh_b[0]

    # Form bond between the two real atoms
    emol.AddBond(real_a, real_b, Chem.BondType.SINGLE)

    # Remove both dummy atoms (higher index first to preserve indices)
    for idx in sorted([dummy_a_idx, dummy_b_combined], reverse=True):
        emol.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(emol)
        return emol.GetMol()
    except Exception as e:
        if debug: print(f"[BRICS JOIN] Sanitization failed after join: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FRAGMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_brics(fragment):
    """
    Get class (number of attachment points) for BRICS fragment.
    
    Args:
        fragment: RDKit Mol object with BRICS attachments
        
    Returns:
        int: Number of attachment points
    """
    n = 0
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            n += 1
    return n


def should_use_brics(fragment, max_atoms=None, max_attachments=None):
    """
    Check if BRICS fragment meets size constraints.
    
    Rejects fragments that are:
    - Too large (> max_atoms heavy atoms) - avoids fragments that are near-complete molecules
    - Have too many attachments (> max_attachments)
    
    Note: No minimum size limit - small fragments like Cl, F, C are useful.
    
    Args:
        fragment: RDKit Mol object with BRICS attachments
        max_atoms: Maximum heavy atoms (defaults to gl.PARAMS["MAX_ATOMS"])
        max_attachments: Maximum attachments (defaults to gl.PARAMS["MAX_FREE"])
        
    Returns:
        bool: True if fragment is within size limits
    """
    if fragment is None:
        return False
    
    if max_atoms is None:
        max_atoms = gl.PARAMS["MAX_ATOMS"]
    if max_attachments is None:
        max_attachments = gl.PARAMS["MAX_FREE"]
    
    n = 0  # attachment points (dummy atoms)
    m = 0  # heavy atoms
    for a in fragment.GetAtoms():
        if a.GetAtomicNum() == 0:  # Dummy atom [n*]
            n += 1
        else:
            m += 1
    
    # Check constraints
    if n > max_attachments:
        return False
    if m > max_atoms:
        return False
        
    return True


def get_fragments_brics(mols, filename=None, verbose=False):
    """
    Split molecules into BRICS-typed fragments.
    
    Returns fragments as Mol objects (consistent with DeepFMPO).
    
    Args:
        mols: List of RDKit Mol objects
        filename: Optional file to write fragments to (not currently used)
        verbose: Print detailed failure info
        
    Returns:
        Tuple of:
        - fragments: dict{smiles: (Mol, class)}
        - used_mols: np.array of bools
        - fragment_sets: list of fragment Mol lists
    """
    used_mols = np.zeros(len(mols)) != 0
    failure_reasons = {}
    fragments = dict()
    fragment_sets = []
    
    # Get size limits from config
    max_atoms = gl.PARAMS["MAX_ATOMS"]
    max_attachments = gl.PARAMS["MAX_FREE"]
    
    for i, mol in enumerate(mols):
        mol_smi = Chem.MolToSmiles(mol) if mol else "Invalid"
        
        try:
            # Use BRICS-typed splitting - returns Mol objects
            fsets = [split_molecule_brics_typed(mol)]
        except Exception as e:
            print(f"Molecule {i+1} could not be fragmented: {str(e)}")
            failure_reasons[i] = f"Exception: {str(e)}"
            continue
        
        # Check results
        if not fsets or all(len(fs) == 0 for fs in fsets):
            failure_reasons[i] = "No fragments produced"
            continue
        
        # Validate and add fragment sets
        mol_was_used = False
        for fs in fsets:  # fs is list of Mol objects
            # Check each fragment and collect rejection reasons
            all_valid = True
            rejection_details = []
            
            for frag in fs:
                frag_smi = Chem.MolToSmiles(frag)
                n_attach = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() == 0)
                n_heavy = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() != 0)
                
                if n_attach > max_attachments:
                    all_valid = False
                    rejection_details.append(f"'{frag_smi}' too many attachments ({n_attach}>{max_attachments})")
                elif n_heavy > max_atoms:
                    all_valid = False
                    rejection_details.append(f"'{frag_smi}' too large ({n_heavy}>{max_atoms} atoms)")
            
            if all_valid:
                used_mols[i] = True
                mol_was_used = True
                fragment_sets.append(fs)
            elif not mol_was_used:
                failure_reasons[i] = f"Fragment size rejection: {'; '.join(rejection_details[:2])}"
    
    # Print summary
    if failure_reasons or verbose:
        print(f"\n{'='*60}")
        print(f"BRICS FRAGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(mols)}, Success: {sum(used_mols)}, Failed: {len(failure_reasons)}")
        print(f"Size limits: max {max_atoms} heavy atoms, max {max_attachments} attachments")
        
        if failure_reasons:
            reason_counts = {}
            for idx, reason in failure_reasons.items():
                # Simplify for grouping
                if "too large" in reason:
                    simple_reason = "Fragment too large"
                elif "too many attachments" in reason:
                    simple_reason = "Too many attachments"
                elif "Exception" in reason:
                    simple_reason = "Exception"
                else:
                    simple_reason = reason.split(':')[0] if ':' in reason else reason
                reason_counts[simple_reason] = reason_counts.get(simple_reason, 0) + 1
            
            print("\nFailure breakdown:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
        print(f"{'='*60}\n")
    
    # Add fragments to dictionary - store Mol objects (like DeepFMPO)
    for fs in fragment_sets:
        for frag in fs:
            smi = Chem.MolToSmiles(frag)
            cl = get_class_brics(frag)
            fragments[smi] = (frag, cl)  # Store Mol object, not SMILES
    
    return fragments, used_mols, fragment_sets


def fragments_to_mol_brics(fragments):
    """
    Convenience function to convert BRICS fragments to molecule.
    
    Args:
        fragments: List of RDKit Mol objects with BRICS attachments
        
    Returns:
        Tuple of (Mol or None, error_message or None)
    """
    mol = join_fragments_brics(fragments)
    if mol is None:
        return None, "Joining failed - BRICS compatibility issue"
    
    return mol, None
