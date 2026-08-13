"""
REINVENT Library Design Molecule Utilities

Fragment-based molecule operations using REINVENT's named reaction approach:
- Chemically-meaningful cuts based on synthetically-relevant reactions
- Numbered attachment points [*:0], [*:1], etc.
- Joins fragments by matching numbered attachment points

This module handles:
- split_molecule_reinvent() - Fragment molecule using named reactions
- join_fragments_reinvent() - Reconstruct molecule by matching attachment numbers
- get_fragments_reinvent() - Extract fragment library from molecules

All functions use Mol objects consistently (same as DeepFMPO/BRICS).
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, RWMol, BondType
from rdkit.Chem.rdmolops import SanitizeMol, CombineMols

import encoders.fmpo_utils.global_parameters as gl

# Import REINVENT library_design components
# Note: The chemistry package is in the project root (not under 'reinvent' namespace)
from chemistry import conversions, tokens
from chemistry.library_design import FragmentReactions, BondMapper
from chemistry.library_design.reaction_definitions.standard_definitions import StandardDefinitions
from chemistry.library_design import attachment_points as ap


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Path to reaction definitions CSV
_REACTION_DEFS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "chemistry/library_design/reaction_definitions/data/reaction_definitions.csv"
)

# Default set of synthetically-relevant reactions to use for fragmentation
# These are common medicinal chemistry transformations
DEFAULT_REACTIONS = [
    "Amide_coupling_PrimAmine",
    "Schotten-Baumann_amide_PrimAmine",
    "sulfon_amide",
    "Buchwald-Hartwig",
    "Suzuki",
    "Sonogashira",
    "heteroaromatic_nuc_sub",
    "Williamsonether",
    "reductiveamination",
    "urea",
    "Acylation_of_amines",
    "Acylation_of_nucleophiles_with_carboxylic_acid_anhydrides",
    "Acylation_of_nucleophiles_with_carboxylic_acids",
    "Addition_of_hydrogen_cyanide_to_alkynes",
    "Alkyaltion_of_amine_with_alkyl_halide",
    "Alpha-Alkylation_of_beta-keto_esters",
    "Alpha-alkylation_of_esters",
    "Arylation_of_amines",
    "Carbamate_formation_from_isocyanate",
    "Cross_Claisen_condensation",
    "Friedel-crafts_Alkylation",
    "Grignard_alcohol",
    "Heck_non-terminal_vinyl",
    "Heck_terminal_vinyl",
    "Kumada_cross-coupling",
    "Menshutkin_reaction",
    "Mitsunobu_imide",
    "Mitsunobu_phenole",
    "Mitsunobu_sulfonamide",
    "Negishi",
    "Nitroalkane_formation",
    "Nucleophilic_aromatic_substitution_SN2",
    "Pinner_reaction",
    "Stille",
    "Wurtz",
    "nucl_sub_aromatic_ortho_nitro",
    "nucl_sub_aromatic_para_nitro",
]

# Lazy-loaded reaction infrastructure
_fragment_reactions = None
_reaction_dtos = None


def _get_reaction_infrastructure():
    """Lazy-load reaction definitions and FragmentReactions instance."""
    global _fragment_reactions, _reaction_dtos
    
    if _fragment_reactions is None:
        _fragment_reactions = FragmentReactions()
        
        # Load reaction definitions
        std_defs = StandardDefinitions(_REACTION_DEFS_PATH)
        
        # Get SMIRKS for each default reaction
        smirks_list = []
        for rxn_name in DEFAULT_REACTIONS:
            try:
                smirks = std_defs.get_reaction_definition(rxn_name)
                smirks_list.append(smirks)
            except IOError:
                # Reaction not found, skip it
                pass
        
        # Create ReactionDTO objects
        _reaction_dtos = _fragment_reactions.create_reactions_from_smirks(smirks_list)
    
    return _fragment_reactions, _reaction_dtos


# ═══════════════════════════════════════════════════════════════════════════════
# SPLITTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def split_molecule_reinvent(mol: Mol, reactions: List[str] = None) -> List[Mol]:
    """
    Split molecule using REINVENT's named reaction approach, recursively.
    
    Applies retrosynthetic reaction SMIRKS to cut bonds. If a resulting
    fragment is too large (fails should_use_reinvent), it is recursively
    split again. This produces multi-attachment fragments when needed.
    
    Args:
        mol: RDKit Mol object
        reactions: Optional list of reaction names to use (defaults to DEFAULT_REACTIONS)
        
    Returns:
        List of RDKit Mol objects with attachment points [*]
    """
    if mol is None:
        return []
    
    frag_rxn, rxn_dtos = _get_reaction_infrastructure()
    
    def _try_split(fragment):
        """Try to split a single fragment. Returns list of sub-fragments or None if no cut found."""
        try:
            fragment_tuples = frag_rxn.slice_molecule_to_fragments(fragment, rxn_dtos)
        except Exception:
            return None
        
        if not fragment_tuples:
            return None
        
        # Try each possible cut, prefer those producing 2 valid pieces
        for frag_tuple in fragment_tuples:
            if len(frag_tuple) >= 2:
                all_valid = all(f is not None for f in frag_tuple)
                if all_valid:
                    return list(frag_tuple)
        return None
    
    # Initial split
    initial_frags = _try_split(mol)
    if not initial_frags or len(initial_frags) <= 1:
        return [mol]
    
    # Recursive splitting: keep splitting fragments that are too large
    max_depth = 5  # safety limit
    final_fragments = []
    queue = list(initial_frags)
    
    for _ in range(max_depth):
        next_queue = []
        for frag in queue:
            if should_use_reinvent(frag):
                final_fragments.append(frag)
            else:
                # Try to split further
                sub_frags = _try_split(frag)
                if sub_frags and len(sub_frags) >= 2:
                    next_queue.extend(sub_frags)
                else:
                    # Can't split further, keep as-is
                    final_fragments.append(frag)
        
        if not next_queue:
            break
        queue = next_queue
    
    # Any remaining in queue couldn't be split further
    final_fragments.extend(queue)
    
    if len(final_fragments) <= 1:
        return [mol]
    
    # Number attachment points globally across all fragments
    final_fragments = _number_attachment_points_global(final_fragments)
    
    return final_fragments


def _number_attachment_points_global(fragments: List[Mol]) -> List[Mol]:
    """
    Number attachment points globally across all fragments.
    
    Each cut produces two matching [*] atoms (one per fragment).
    We assign the same number to each matching pair so that
    joining can match them: [*:0]↔[*:0], [*:1]↔[*:1], etc.
    
    Since REINVENT reactions produce pairs, each [*] in a fragment
    already has a partner in another fragment. We number them
    sequentially by order of appearance across all fragments.
    
    Args:
        fragments: List of Mol objects with [*] attachment points
        
    Returns:
        List of Mol objects with numbered attachment points [*:n]
    """
    result = []
    ap_counter = 0
    
    # First pass: count total attachment points per fragment
    # Each attachment point gets a unique number; matching happens
    # via the reaction product pairing (same reaction → same number)
    for frag in fragments:
        if frag is None:
            continue
        try:
            smi = Chem.MolToSmiles(frag)
            # Number all [*] in this fragment sequentially from ap_counter
            def _ap_replace(match):
                nonlocal ap_counter
                num = ap_counter
                ap_counter += 1
                return f"[*:{num}]"
            
            numbered_smi = re.sub(tokens.ATTACHMENT_POINT_REGEXP, _ap_replace, smi)
            numbered_mol = Chem.MolFromSmiles(numbered_smi)
            if numbered_mol:
                result.append(numbered_mol)
            else:
                result.append(frag)
        except Exception:
            result.append(frag)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# JOINING FUNCTIONS  
# ═══════════════════════════════════════════════════════════════════════════════

def join_fragments_reinvent(fragments: List[Mol], debug: bool = False) -> Optional[Mol]:
    """
    Join REINVENT-style fragments by matching numbered attachment points.
    
    Fragments with matching [*:n] attachment points are joined together.
    
    Args:
        fragments: List of RDKit Mol objects with numbered attachments
        debug: Print debug info
        
    Returns:
        RDKit Mol object or None if joining fails
    """
    if not fragments:
        if debug: print("[REINVENT JOIN] No fragments provided")
        return None
    
    if debug:
        print(f"[REINVENT JOIN] Joining {len(fragments)} fragments:")
        for i, f in enumerate(fragments):
            smi = Chem.MolToSmiles(f) if f else "None"
            print(f"  Fragment {i}: {smi}")
    
    # Handle single fragment - remove attachment points
    if len(fragments) == 1:
        return _remove_attachment_points(fragments[0], debug)
    
    # Convert fragments to SMILES for joining
    frag_smiles = []
    for f in fragments:
        if f is not None:
            smi = Chem.MolToSmiles(f)
            # Ensure attachment points are numbered
            if '[*]' in smi and '[*:' not in smi:
                smi = ap.add_attachment_point_numbers(smi, canonicalize=True)
            frag_smiles.append(smi)
    
    if len(frag_smiles) < 2:
        if debug: print("[REINVENT JOIN] Not enough valid fragments")
        return _remove_attachment_points(fragments[0], debug) if fragments else None
    
    # Try to join fragments iteratively
    result_mol = None
    scaffold_smi = frag_smiles[0]
    decorations = frag_smiles[1:]
    
    # Simple joining: match attachment points by number
    try:
        result_mol = _join_by_attachment_numbers(scaffold_smi, decorations, debug)
    except Exception as e:
        if debug: print(f"[REINVENT JOIN] Joining failed: {e}")
        return None
    
    if result_mol:
        # Final cleanup - remove any remaining attachment points
        result_mol = _remove_attachment_points(result_mol, debug)
        
        if debug and result_mol:
            print(f"[REINVENT JOIN] Success: {Chem.MolToSmiles(result_mol)}")
    
    return result_mol


def _join_by_attachment_numbers(scaffold_smi: str, decoration_smiles: List[str], debug: bool = False) -> Optional[Mol]:
    """
    Join fragments by matching numbered attachment points.
    
    Args:
        scaffold_smi: SMILES of the scaffold fragment
        decoration_smiles: List of decoration SMILES to attach
        debug: Print debug info
        
    Returns:
        Joined molecule or None
    """
    # Get all attachment points in scaffold
    scaffold_aps = ap.get_attachment_points(scaffold_smi)
    
    if not scaffold_aps and not decoration_smiles:
        # No attachments, just return the scaffold
        return Chem.MolFromSmiles(scaffold_smi)
    
    # Build decoration string with separator
    dec_str = tokens.ATTACHMENT_SEPARATOR_TOKEN.join(decoration_smiles)
    
    # Try REINVENT's native joining
    try:
        from chemistry.library_design.bond_maker import join_scaffolds_and_decorations
        result = join_scaffolds_and_decorations(scaffold_smi, dec_str, keep_labels_on_atoms=False)
        return result
    except Exception as e:
        if debug: print(f"[REINVENT JOIN] Native join failed: {e}")
    
    # Fallback: manual join by matching attachment point numbers
    return _manual_join_fragments(scaffold_smi, decoration_smiles, debug)


def _manual_join_fragments(scaffold_smi: str, decoration_smiles: List[str], debug: bool = False) -> Optional[Mol]:
    """
    Manual fallback for joining fragments by matching attachment numbers.
    """
    # Combine all fragments
    all_smiles = [scaffold_smi] + decoration_smiles
    
    combined_mol = None
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if combined_mol is None:
            combined_mol = mol
        else:
            combined_mol = CombineMols(combined_mol, mol)
    
    if combined_mol is None:
        return None
    
    # Find matching attachment points by atom map number
    emol = RWMol(combined_mol)
    
    # Build map of attachment number -> atom indices
    ap_map: Dict[int, List[int]] = {}
    for atom in emol.GetAtoms():
        if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN:
            if atom.HasProp("molAtomMapNumber"):
                ap_num = int(atom.GetProp("molAtomMapNumber"))
                if ap_num not in ap_map:
                    ap_map[ap_num] = []
                ap_map[ap_num].append(atom.GetIdx())
    
    # Join matching attachment points
    bonds_to_add = []
    atoms_to_remove = set()
    
    for ap_num, ap_idxs in ap_map.items():
        if len(ap_idxs) == 2:
            # Get neighbors of both attachment points
            neighbors = []
            for ap_idx in ap_idxs:
                ap_atom = emol.GetAtomWithIdx(ap_idx)
                for neigh in ap_atom.GetNeighbors():
                    neighbors.append(neigh.GetIdx())
                atoms_to_remove.add(ap_idx)
            
            if len(neighbors) == 2:
                bonds_to_add.append((neighbors[0], neighbors[1]))
    
    # Add bonds
    for a1, a2 in bonds_to_add:
        try:
            emol.AddBond(a1, a2, BondType.SINGLE)
        except:
            pass  # Bond may already exist
    
    # Remove attachment point atoms (in reverse order)
    for idx in sorted(atoms_to_remove, reverse=True):
        try:
            emol.RemoveAtom(idx)
        except:
            pass
    
    # Sanitize
    try:
        SanitizeMol(emol)
        result = emol.GetMol()
        result = Chem.AddHs(result)
        result = Chem.RemoveHs(result)
        return result
    except Exception as e:
        if debug: print(f"[REINVENT JOIN] Manual join sanitize failed: {e}")
        return None


def _remove_attachment_points(mol: Mol, debug: bool = False) -> Optional[Mol]:
    """
    Remove attachment points from a molecule, replacing with H.
    
    Args:
        mol: RDKit Mol object with attachment points
        debug: Print debug info
        
    Returns:
        Mol with attachment points removed
    """
    if mol is None:
        return None
    
    emol = RWMol(mol)
    
    # Find all attachment point atoms
    atoms_to_remove = []
    for atom in emol.GetAtoms():
        if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN or atom.GetAtomicNum() == 0:
            atoms_to_remove.append(atom.GetIdx())
    
    # Remove in reverse order
    for idx in sorted(atoms_to_remove, reverse=True):
        try:
            emol.RemoveAtom(idx)
        except:
            pass
    
    try:
        SanitizeMol(emol)
        result = emol.GetMol()
        result = Chem.AddHs(result)
        result = Chem.RemoveHs(result)
        return result
    except Exception as e:
        if debug: print(f"[REINVENT] Remove attachment points failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FRAGMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_reinvent(fragment: Mol) -> int:
    """
    Get class (number of attachment points) for REINVENT fragment.
    
    Args:
        fragment: RDKit Mol object with REINVENT attachments
        
    Returns:
        int: Number of attachment points
    """
    n = 0
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN or atom.GetAtomicNum() == 0:
            n += 1
    return n


def should_use_reinvent(fragment: Mol, max_atoms: int = None, max_attachments: int = None) -> bool:
    """
    Check if REINVENT fragment meets size constraints.
    
    Rejects fragments that are:
    - Too large (> max_atoms heavy atoms)
    - Have too many attachments (> max_attachments)
    
    Args:
        fragment: RDKit Mol object with attachments
        max_atoms: Maximum heavy atoms (defaults to gl.PARAMS["MAX_ATOMS"])
        max_attachments: Maximum attachments (defaults to gl.PARAMS["MAX_FREE"])
        
    Returns:
        bool: True if fragment is within size limits
    """
    if fragment is None:
        return False
    
    if max_atoms is None:
        max_atoms = gl.PARAMS.get("MAX_ATOMS", 10)
    if max_attachments is None:
        max_attachments = gl.PARAMS.get("MAX_FREE", 4)
    
    n_attach = 0  # attachment points
    n_heavy = 0   # heavy atoms
    
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN or atom.GetAtomicNum() == 0:
            n_attach += 1
        else:
            n_heavy += 1
    
    # Check constraints
    if n_attach > max_attachments:
        return False
    if n_heavy > max_atoms:
        return False
    
    return True


def get_fragments_reinvent(mols: List[Mol], filename: str = None, verbose: bool = False) -> Tuple[Dict, np.ndarray, List]:
    """
    Split molecules into REINVENT-style fragments using named reactions.
    
    Returns fragments as Mol objects (consistent with DeepFMPO/BRICS).
    
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
    
    # REINVENT uses named reactions - no size limits needed since fragments
    # are chemically reasonable by design
    
    for i, mol in enumerate(mols):
        if mol is None:
            failure_reasons[i] = "Invalid molecule"
            continue
            
        mol_smi = Chem.MolToSmiles(mol)
        
        try:
            # Use REINVENT-style splitting
            fsets = [split_molecule_reinvent(mol)]
        except Exception as e:
            failure_reasons[i] = f"Exception: {str(e)}"
            continue
        
        # Check results
        if not fsets or all(len(fs) <= 1 for fs in fsets):
            # No fragmentation possible - molecule stays as one piece
            # This is actually OK for REINVENT - just means no reaction bonds found
            failure_reasons[i] = "No reaction bonds found"
            continue
        
        # Validate and add fragment sets (no size limits for REINVENT)
        mol_was_used = False
        for fs in fsets:
            if len(fs) <= 1:
                continue
                
            # Check each fragment is valid (not None)
            all_valid = True
            for frag in fs:
                if frag is None:
                    all_valid = False
                    break
            
            if all_valid:
                used_mols[i] = True
                mol_was_used = True
                fragment_sets.append(fs)
            elif not mol_was_used:
                failure_reasons[i] = f"Fragment size rejection: {'; '.join(rejection_details[:2])}"
    
    # Collect unique fragments from all fragment sets
    for fs in fragment_sets:
        for frag in fs:
            if frag is None:
                continue
            smi = Chem.MolToSmiles(frag)
            if smi not in fragments:
                frag_class = get_class_reinvent(frag)
                fragments[smi] = (frag, frag_class)
    
    # Print summary
    if failure_reasons or verbose:
        print(f"\n{'='*60}")
        print(f"REINVENT FRAGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(mols)}, Success: {sum(used_mols)}, Failed: {len(failure_reasons)}")
        print(f"Unique fragments: {len(fragments)}")
        print(f"Reactions used: {', '.join(DEFAULT_REACTIONS[:5])}...")
        
        if failure_reasons:
            reason_counts = {}
            for idx, reason in failure_reasons.items():
                # Simplify for grouping
                if "No reaction bonds" in reason:
                    simple_reason = "No reaction bonds found"
                elif "Exception" in reason:
                    simple_reason = "Exception during fragmentation"
                else:
                    simple_reason = reason[:50]
                reason_counts[simple_reason] = reason_counts.get(simple_reason, 0) + 1
            
            print(f"\nFailure breakdown:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
        print(f"{'='*60}\n")
    
    return fragments, used_mols, fragment_sets


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_reinvent_fragment(fragment: Mol) -> bool:
    """
    Check if a fragment uses REINVENT-style attachment points [*:n].
    
    Args:
        fragment: RDKit Mol object
        
    Returns:
        bool: True if fragment uses REINVENT attachment style
    """
    if fragment is None:
        return False
    
    smi = Chem.MolToSmiles(fragment)
    # REINVENT uses [*:n] format
    return bool(re.search(r'\[\*:\d+\]', smi))


def get_available_reactions() -> List[str]:
    """Return list of available reaction names."""
    return DEFAULT_REACTIONS.copy()


def set_reactions(reaction_names: List[str]):
    """
    Set which reactions to use for fragmentation.
    
    Args:
        reaction_names: List of reaction names from reaction_definitions.csv
    """
    global DEFAULT_REACTIONS, _reaction_dtos
    DEFAULT_REACTIONS = reaction_names
    _reaction_dtos = None  # Force reload
