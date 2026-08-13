#!/usr/bin/env python
# coding: utf-8

"""
BRICS type-safe fragment matching utilities.

BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)
defines 16 environments (L1-L16) with specific connection rules.
This module ensures fragments are only swapped when their attachment
point types are compatible.

BRICS Environment Compatibility Rules:
- L1-L2: Ring carbon - Linker
- L3-L4: Amide N - Carbonyl C
- L5-L6: Amine N - C
- L7-L7: Aromatic C - Aromatic C
- L8-L9: Aromatic C - heteroatom linker
- L10-L10: Aromatic N - Aromatic C
- L11-L11: S/O - C
- L12-L12: tertiary N - C
- L13-L14: C - C (sp3)
- L14-L14: C(sp3) - C(sp3)
- L15-L15: C - C (near ring)
- L16-L16: C - C (aromatic adjacent)
"""

import re
from typing import Dict, Set, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import BRICS

# BRICS compatibility rules: which environment types can connect to each other
# Based on BRICS retrosynthetic rules
BRICS_COMPATIBILITY = {
    1: {2},      # L1 connects to L2
    2: {1},      # L2 connects to L1
    3: {4},      # L3 connects to L4 (amide nitrogen to carbonyl)
    4: {3},      # L4 connects to L3
    5: {6},      # L5 connects to L6 (amine)
    6: {5},      # L6 connects to L5
    7: {7},      # L7 connects to L7 (aromatic carbon)
    8: {9},      # L8 connects to L9
    9: {8},      # L9 connects to L8
    10: {10},    # L10 self-compatible
    11: {11},    # L11 self-compatible
    12: {12},    # L12 self-compatible
    13: {14},    # L13 connects to L14
    14: {13, 14, 15, 16},  # L14 broad compatibility (sp3 carbon)
    15: {14, 15, 16},      # L15 near-ring carbon
    16: {14, 15, 16},      # L16 aromatic-adjacent carbon
}


def extract_attachment_types(smiles: str) -> List[int]:
    """
    Extract BRICS attachment point types from a fragment SMILES.
    
    Attachment points are encoded as [n*] where n is the environment type (1-16).
    Also handles rare element markers ([Yb], [Lu], etc.) used in DeepFMPO.
    
    Args:
        smiles: Fragment SMILES string
        
    Returns:
        List of attachment point types (1-16) found in the fragment
    """
    types = []
    
    # Pattern for numbered attachment points [n*] or [n*:m]
    numbered_pattern = re.compile(r'\[(\d+)\*(?::\d+)?\]')
    matches = numbered_pattern.findall(smiles)
    types.extend([int(m) for m in matches])
    
    # If no numbered types found, check for rare elements (legacy DeepFMPO format)
    # These indicate generic attachment points without BRICS typing
    if not types:
        rare_elements = ['Yb', 'Lu', 'Tm', 'Er', 'Ho', 'Dy', 'Tb', 'Gd']
        for elem in rare_elements:
            if f'[{elem}]' in smiles:
                # Legacy format - treat as type 0 (generic)
                types.append(0)
    
    return types


def get_fragment_attachment_signature(mol_or_smiles) -> Tuple[int, ...]:
    """
    Get a hashable signature of a fragment's attachment point types.
    
    Args:
        mol_or_smiles: RDKit Mol or SMILES string
        
    Returns:
        Sorted tuple of attachment types (can be used as dict key)
    """
    if isinstance(mol_or_smiles, str):
        smiles = mol_or_smiles
    else:
        smiles = Chem.MolToSmiles(mol_or_smiles)
    
    types = extract_attachment_types(smiles)
    return tuple(sorted(types))


def check_attachment_compatibility(type1: int, type2: int) -> bool:
    """
    Check if two BRICS attachment types are compatible for joining.
    
    Args:
        type1: First attachment type (1-16 or 0 for generic)
        type2: Second attachment type (1-16 or 0 for generic)
        
    Returns:
        True if types can connect, False otherwise
    """
    # Type 0 is generic (legacy) - compatible with anything
    if type1 == 0 or type2 == 0:
        return True
    
    # Check compatibility based on BRICS rules
    return type2 in BRICS_COMPATIBILITY.get(type1, set())


def fragments_are_compatible(frag1_types: Tuple[int, ...], 
                              frag2_types: Tuple[int, ...]) -> bool:
    """
    Check if two fragments have compatible attachment point profiles.
    
    For fragment swapping, the new fragment should have attachment points
    that can connect to the same neighbors as the old fragment.
    This means they should have the same number of attachment points
    with compatible types.
    
    Args:
        frag1_types: Tuple of attachment types for fragment 1
        frag2_types: Tuple of attachment types for fragment 2
        
    Returns:
        True if fragments can be swapped, False otherwise
    """
    # Must have same number of attachment points
    if len(frag1_types) != len(frag2_types):
        return False
    
    # Empty fragments are compatible with each other
    if len(frag1_types) == 0:
        return True
    
    # Check if types are compatible
    # For a swap to work, each attachment point in frag1 must be
    # compatible with the corresponding point in frag2
    # Sort both and compare - compatible types should match
    
    # Simple approach: same types are always compatible
    if frag1_types == frag2_types:
        return True
    
    # Check if types can substitute for each other based on BRICS rules
    # E.g., if frag1 has type 3 and frag2 has type 4, they're not directly
    # swappable because 3 connects to 4, not to itself
    
    # For now, use strict matching: fragments must have identical
    # attachment type signatures for safe swapping
    return frag1_types == frag2_types


def build_brics_compatibility_cache(decodings: Dict[str, bytes]) -> Dict[Tuple[int, ...], Set[str]]:
    """
    Build a cache grouping fragments by their BRICS attachment type signatures.
    
    Args:
        decodings: Dict mapping binary codes to RDKit Mol bytes
        
    Returns:
        Dict mapping attachment signatures to sets of compatible binary codes
    """
    signature_to_codes: Dict[Tuple[int, ...], Set[str]] = {}
    
    for code, mol_bytes in decodings.items():
        try:
            mol = Chem.Mol(mol_bytes)
            smiles = Chem.MolToSmiles(mol)
            signature = get_fragment_attachment_signature(smiles)
            
            if signature not in signature_to_codes:
                signature_to_codes[signature] = set()
            signature_to_codes[signature].add(code)
        except Exception:
            # Skip fragments that can't be processed
            continue
    
    return signature_to_codes


def build_brics_valid_action_cache(decodings: Dict[str, bytes], 
                                    max_swap: int) -> Dict[str, Set[int]]:
    """
    Build action validity cache that only allows BRICS-compatible swaps.
    
    A bit flip is valid only if:
    1. The resulting code exists in decodings
    2. The resulting fragment has compatible BRICS attachment types
    
    Args:
        decodings: Dict mapping binary codes to RDKit Mol bytes
        max_swap: Number of bits that can be swapped per fragment
        
    Returns:
        Dict mapping binary codes to sets of valid swap bit indices
    """
    # First, compute attachment signatures for all fragments
    code_to_signature: Dict[str, Tuple[int, ...]] = {}
    for code, mol_bytes in decodings.items():
        try:
            mol = Chem.Mol(mol_bytes)
            smiles = Chem.MolToSmiles(mol)
            code_to_signature[code] = get_fragment_attachment_signature(smiles)
        except Exception:
            code_to_signature[code] = ()
    
    # Build the validity cache
    cache: Dict[str, Set[int]] = {}
    
    for code in decodings.keys():
        valid_swaps = set()
        current_signature = code_to_signature.get(code, ())
        
        for bit in range(max_swap):
            # Flip bit to get new code
            flipped = _flip_bit_code(code, bit)
            
            # Check if new code exists
            if flipped not in decodings:
                continue
            
            # Check BRICS compatibility
            flipped_signature = code_to_signature.get(flipped, ())
            if fragments_are_compatible(current_signature, flipped_signature):
                valid_swaps.add(bit)
        
        cache[code] = valid_swaps
    
    return cache


def _flip_bit_code(code: str, bit_pos: int) -> str:
    """Flip bit at position (from right) in binary code string."""
    bits = list(code)
    idx = -(1 + bit_pos)
    bits[idx] = '1' if bits[idx] == '0' else '0'
    return ''.join(bits)


def get_brics_compatibility_stats(decodings: Dict[str, bytes]) -> Dict:
    """
    Get statistics about BRICS attachment types in the fragment library.
    
    Args:
        decodings: Dict mapping binary codes to RDKit Mol bytes
        
    Returns:
        Dict with statistics about attachment type distribution
    """
    type_counts: Dict[int, int] = {}
    signature_counts: Dict[Tuple[int, ...], int] = {}
    
    for code, mol_bytes in decodings.items():
        try:
            mol = Chem.Mol(mol_bytes)
            smiles = Chem.MolToSmiles(mol)
            signature = get_fragment_attachment_signature(smiles)
            
            # Count signatures
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            
            # Count individual types
            for t in signature:
                type_counts[t] = type_counts.get(t, 0) + 1
                
        except Exception:
            continue
    
    return {
        'type_counts': type_counts,
        'signature_counts': signature_counts,
        'n_unique_signatures': len(signature_counts),
        'total_fragments': len(decodings),
    }


if __name__ == '__main__':
    # Test with example fragments
    test_fragments = [
        '[1*]c1ccccc1',      # Aromatic ring with L1 attachment
        '[2*]CCCC',          # Chain with L2 attachment
        '[3*]N(C)C(=O)[4*]', # Amide with L3 and L4
        '[7*]c1ccc([7*])cc1', # Benzene ring with two L7
        '[Yb]c1ccccc1',      # Legacy format
    ]
    
    for frag in test_fragments:
        types = extract_attachment_types(frag)
        sig = get_fragment_attachment_signature(frag)
        print(f'{frag}')
        print(f'  Types: {types}')
        print(f'  Signature: {sig}')
        print()
    
    # Test compatibility
    print('Compatibility tests:')
    print(f'  L1-L2: {check_attachment_compatibility(1, 2)}')  # True
    print(f'  L1-L1: {check_attachment_compatibility(1, 1)}')  # False
    print(f'  L7-L7: {check_attachment_compatibility(7, 7)}')  # True
    print(f'  L3-L4: {check_attachment_compatibility(3, 4)}')  # True
