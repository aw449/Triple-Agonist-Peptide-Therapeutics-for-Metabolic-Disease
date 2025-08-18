"""
Peptide Sequence Handler for Genetic Algorithm
Handles peptide sequences with standard and non-standard amino acids.
"""

import re
import random
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import Counter


@dataclass
class PeptideProperties:
    """Properties of a peptide sequence."""
    length: int
    charge: float
    hydrophobicity: float
    molecular_weight: float
    has_modifications: bool
    modification_count: int


class PeptideSequenceHandler:
    """
    Handles parsing, validation, and manipulation of peptide sequences
    with support for non-standard amino acids and modifications.
    Enhanced with biological plausibility constraints and motif preservation.
    """
    
    def __init__(self):
        # Standard amino acids
        self.standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Non-standard amino acids and modifications with their properties
        self.modification_properties = {
            # D-amino acids
            '[dNle]': {'type': 'd_amino', 'base_aa': 'L', 'hydrophobicity': 3.8, 'charge': 0, 'mw': 131},
            '[dAla]': {'type': 'd_amino', 'base_aa': 'A', 'hydrophobicity': 1.8, 'charge': 0, 'mw': 89},
            '[s]': {'type': 'd_amino', 'base_aa': 'S', 'hydrophobicity': -0.8, 'charge': 0, 'mw': 105},
            '[a]': {'type': 'd_amino', 'base_aa': 'A', 'hydrophobicity': 1.8, 'charge': 0, 'mw': 89},
            
            # Unnatural amino acids
            '[Aib]': {'type': 'unnatural', 'base_aa': 'A', 'hydrophobicity': 1.8, 'charge': 0, 'mw': 103},
            '[Nle]': {'type': 'unnatural', 'base_aa': 'L', 'hydrophobicity': 3.8, 'charge': 0, 'mw': 131},
            '[Sar]': {'type': 'unnatural', 'base_aa': 'G', 'hydrophobicity': 0.0, 'charge': 0, 'mw': 89},
            '[hSer]': {'type': 'unnatural', 'base_aa': 'S', 'hydrophobicity': -0.4, 'charge': 0, 'mw': 119},
            '[nVal]': {'type': 'unnatural', 'base_aa': 'V', 'hydrophobicity': 3.8, 'charge': 0, 'mw': 117},
            '[Dap]': {'type': 'unnatural', 'base_aa': 'K', 'hydrophobicity': -2.5, 'charge': 1, 'mw': 104},
            '[MetO]': {'type': 'unnatural', 'base_aa': 'M', 'hydrophobicity': 0.3, 'charge': 0, 'mw': 165},
            
            # Lipidations
            '[K(yE-C16)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': 0.6, 'charge': -1.23, 'mw': 513},
            '[K(yE-yE-C16)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': -2.9, 'charge': -2.22, 'mw': 642},
            '[K(yE-C18)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': 1.6, 'charge': -1.23, 'mw': 541},
            '[K(OEG-OEG-yE-C18DA)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': -3.4, 'charge': -2.22, 'mw': 741.599},
            '[K(OEG-OEG-yE-C20DA)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': -2.4, 'charge': -2.22, 'mw': 769.63},
            '[K(eK-eK-yE-C20DA)]': {'type': 'lipidation', 'base_aa': 'K', 'hydrophobicity': -8.2, 'charge': -0.22, 'mw': 855.606}
        }
        
        # Critical binding motifs from peptide hormone families
        self.critical_motifs = {
            'glucagon_family': [
                {'pattern': ["H", "*", "*", "G", "*", "F", "*", "*", "D", "*", "*", "*", "*", "L", "D"], 'position': 0, 'importance': 0.8, 'name': 'N-terminus_Paralog'},
                {'pattern': [ "*", "*", "*", "*", "*", "D", "F", "V", "*", "W", "L", "*", "*"], 'position': 15, 'importance': 0.8, 'name': 'C-terminus_Paralog'},
                {'pattern': ["S", "Q", "*", "T", "*", "*", "*", "*", "*", "*", "*", "Y", "*", "*"], 'position': 1, 'importance': 0.95, 'name': 'N-terminus_Ortholog'},
                {'pattern': ["A"], 'position': 18, 'importance': 0.95, 'name': 'C-terminus_Ortholog'}
            ],
            'glp1_family': [
                {'pattern': ["H", "*", "E", "G", "*", "F", "T", "*", "D", "*", "*"], 'position': 0, 'importance': 0.9, 'name': 'N_terminal_GLP1'},
                {'pattern': ["F", "I", "*", "*", "L", "*", "*", "*", "R"], 'position': 15, 'importance': 0.8, 'name': 'C_terminal_GLP1'},
                {'pattern': ["H", "*", "*", "G", "*", "F", "*", "*", "D", "*", "*", "*", "*", "L", "*"], 'position': 0, 'importance': 0.8, 'name': 'N-terminus_Paralog'},
                {'pattern': [ "*", "*", "*", "*", "*", "*", "F", "I", "*", "W", "L", "*", "*"], 'position': 15, 'importance': 0.8, 'name': 'C-terminus_Paralog'},
                {'pattern': ["A", "E", "*", "T", "*", "T", "*", "*", "*", "*", "*", "Y", "*", "*"], 'position': 1, 'importance': 0.95, 'name': 'N-terminus_Ortholog'},
                {'pattern': ["*", "*", "A", "A", "K", "E", "*", "*", "*", "*", "*", "*", "*","G"], 'position': 15, 'importance': 0.95, 'name': 'C-terminus_Ortholog'}
            ],
            "gip_family": [
                {'pattern': ["*", "*", "*", "G", "*", "F", "*", "*", "D", "*", "*", "*", "*", "*", "D"], 'position': 0, 'importance': 0.8, 'name': 'N-terminus_Paralog'},
                {'pattern': [ "*", "*", "*", "*", "*", "D", "F", "V", "*", "W", "L", "*", "*"], 'position': 15, 'importance': 0.8, 'name': 'C-terminus_Paralog'},
                {'pattern': ["Y", "*", "E", "*", "*", "*", "*", "S", "*", "*", "S", "*", "*", "*"], 'position': 0, 'importance': 0.95, 'name': 'N-terminus_Ortholog'},
                {'pattern': ["N", "*", "A", "L"], 'position': 23, 'importance': 0.95, 'name': 'C-terminus_Ortholog'}
            ],
            'conserved_core': [
                {'pattern': ['E', 'G', 'T', 'F'], 'position': 2, 'importance': 0.95, 'name': 'essential_core'},
                {'pattern': ["F", "*", "*", "W", "L"], 'position': 21, 'importance': 0.95, 'name': 'essential_core'},
                {'pattern': ["D"], 'position': 9, 'importance': 0.95, 'name': 'essential_core'}
            ]
        }
        
        # Proteolytic cleavage sites to avoid
        self.cleavage_sites = {
            'trypsin': [['K', 'R'], ['K'], ['R']],  # After K or R
            'chymotrypsin': [['F'], ['W'], ['Y'], ['L']],  # After aromatic and Leu
            'pepsin': [['F', '*'], ['L', '*']],  # Various patterns
            'endopeptidases': [['D', 'E'], ['K', 'K'], ['R', 'R']]  # Double charged
        }
        
        # Natural amino acid frequencies in peptide hormones (for plausibility scoring)
        self.natural_frequencies = {
            'A': 0.08, 'C': 0.02, 'D': 0.06, 'E': 0.08, 'F': 0.04,
            'G': 0.09, 'H': 0.03, 'I': 0.05, 'K': 0.07, 'L': 0.10,
            'M': 0.02, 'N': 0.05, 'P': 0.05, 'Q': 0.05, 'R': 0.06,
            'S': 0.08, 'T': 0.06, 'V': 0.07, 'W': 0.02, 'Y': 0.03
        }
        
        # Standard amino acid properties
        self.aa_properties = {
            'hydrophobicity': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            'charge': {
                'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
                'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
                'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
            },
            'molecular_weight': {
                'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
                'Q': 146, 'E': 147, 'G': 75, 'H': 155, 'I': 131,
                'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
                'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117
            }
        }
        
        # Valid amino acids for mutations (including non-standard)
        self.all_valid_residues = list(self.standard_aa) + list(self.modification_properties.keys())
        
        # Probability weights for different residue types during mutation
        self.mutation_weights = {
            'standard': 0.97,  # 85% chance for standard AA
            'd_amino': 0.01,   # 8% chance for D-amino acids
            'unnatural': 0.01, # 5% chance for unnatural AAs
            'lipidation': 0.01 # 2% chance for lipidations
        }
    
    def tokenize_sequence(self, sequence: str) -> List[str]:
        """
        Tokenize a peptide sequence into individual residues/modifications.
        
        Args:
            sequence: Peptide sequence string
            
        Returns:
            List of tokens (standard AAs or modifications)
        """
        if not sequence or sequence.strip() == '':
            return []
        
        sequence = sequence.strip().replace('-', '')  # Remove gaps
        tokens = []
        i = 0
        
        while i < len(sequence):
            if sequence[i] == '[':
                # Find the closing bracket
                close_idx = sequence.find(']', i)
                if close_idx == -1:
                    # Malformed bracket, treat as regular character
                    if sequence[i] in self.standard_aa:
                        tokens.append(sequence[i])
                    i += 1
                else:
                    # Extract the modification
                    modification = sequence[i:close_idx+1]
                    if modification in self.modification_properties:
                        tokens.append(modification)
                    i = close_idx + 1
            elif sequence[i] in self.standard_aa:
                # Standard amino acid
                tokens.append(sequence[i])
                i += 1
            else:
                # Unknown character, skip
                i += 1
                
        return tokens
    
    def tokens_to_sequence(self, tokens: List[str]) -> str:
        """Convert list of tokens back to sequence string."""
        return ''.join(tokens)
    
    def calculate_properties(self, tokens: List[str]) -> PeptideProperties:
        """
        Calculate properties of a peptide sequence.
        
        Args:
            tokens: List of amino acid tokens
            
        Returns:
            PeptideProperties object
        """
        if not tokens:
            return PeptideProperties(0, 0, 0, 0, False, 0)
        
        total_charge = 0
        total_hydrophobicity = 0
        total_mw = 0
        modification_count = 0
        
        for token in tokens:
            if token in self.standard_aa:
                # Standard amino acid
                total_charge += self.aa_properties['charge'][token]
                total_hydrophobicity += self.aa_properties['hydrophobicity'][token]
                total_mw += self.aa_properties['molecular_weight'][token]
            elif token in self.modification_properties:
                # Modified amino acid
                props = self.modification_properties[token]
                total_charge += props['charge']
                total_hydrophobicity += props['hydrophobicity']
                total_mw += props['mw']
                modification_count += 1
            else:
                # Unknown token, use neutral values
                total_mw += 100  # Average AA weight
        
        length = len(tokens)
        avg_hydrophobicity = total_hydrophobicity / length if length > 0 else 0
        
        return PeptideProperties(
            length=length,
            charge=total_charge,
            hydrophobicity=avg_hydrophobicity,
            molecular_weight=total_mw,
            has_modifications=(modification_count > 0),
            modification_count=modification_count
        )
    
    def is_valid_length(self, tokens: List[str], min_length: int = 25, max_length: int = 35) -> bool:
        """Check if sequence length is within valid range."""
        return min_length <= len(tokens) <= max_length
    
    def is_chemically_plausible(self, tokens: List[str], threshold: float = 0.3) -> bool:
        """
        Check if sequence is chemically plausible using enhanced evaluation.
        
        Args:
            tokens: Sequence tokens
            threshold: Minimum plausibility score (0.3 = basic plausibility)
        
        Returns:
            True if sequence meets plausibility threshold
        """
        if not tokens:
            return False
        
        plausibility = self.evaluate_biological_plausibility(tokens)
        return plausibility['overall'] >= threshold
    
    def repair_sequence(self, tokens: List[str], min_length: int = 25, max_length: int = 35) -> List[str]:
        """
        Repair a sequence to make it valid (correct length and biologically plausible).
        Enhanced with motif preservation during repair.
        
        Args:
            tokens: Current sequence tokens
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            Repaired sequence tokens
        """
        if not tokens:
            return self.generate_random_sequence(min_length)
        
        repaired = tokens.copy()
        
        # Fix length issues
        while len(repaired) < min_length:
            # Add random residues, preferring positions that don't disrupt motifs
            position = self._find_safe_insertion_position(repaired)
            repaired.insert(position, self.get_weighted_random_residue())
        
        while len(repaired) > max_length:
            if len(repaired) <= min_length:
                break
            
            # Remove residues, avoiding critical motif positions
            position = self._find_safe_deletion_position(repaired)
            del repaired[position]
        
        # Enhance plausibility through targeted repairs
        max_repair_attempts = 30
        attempts = 0
        
        while attempts < max_repair_attempts:
            plausibility = self.evaluate_biological_plausibility(repaired)
            
            if plausibility['overall'] >= 0.4:  # Acceptable threshold
                break
            
            # Identify and fix the worst component
            worst_component = min(plausibility.items(), key=lambda x: x[1] if x[0] != 'overall' else 1.0)
            
            if worst_component[0] == 'chemical':
                repaired = self._repair_chemical_issues(repaired)
            elif worst_component[0] == 'proteolysis':
                repaired = self._repair_proteolysis_issues(repaired)
            elif worst_component[0] == 'composition':
                repaired = self._repair_composition_issues(repaired)
            else:
                # General mutation
                if repaired:
                    position = random.randint(0, len(repaired) - 1)
                    repaired[position] = self.get_weighted_random_residue()
            
            attempts += 1
        
        return repaired
    
    def _get_critical_positions(self, tokens: List[str]) -> Dict[int, float]:
        """
        Get positions that are critical (non-wildcard) in motifs with their importance weights.
        
        Returns:
            Dict mapping position -> max_importance_score for that position
        """
        if not tokens:
            return {}
        
        critical_positions = {}
        
        for motifs in self.critical_motifs.values():
            for motif in motifs:
                pattern = motif['pattern']
                position = motif['position']
                importance = motif['importance']
                
                # Handle negative positions
                if position < 0:
                    start_pos = len(tokens) + position
                else:
                    start_pos = position
                
                # Check bounds
                if start_pos < 0 or start_pos + len(pattern) > len(tokens):
                    continue
                
                # Mark non-wildcard positions as critical
                for i, pattern_aa in enumerate(pattern):
                    if pattern_aa != '*':  # Only protect non-wildcard positions
                        pos = start_pos + i
                        if 0 <= pos < len(tokens):
                            # Keep the highest importance score for each position
                            current_importance = critical_positions.get(pos, 0.0)
                            critical_positions[pos] = max(current_importance, importance)
        
        return critical_positions
    
    def _find_safe_insertion_position(self, tokens: List[str]) -> int:
        """Find a position for insertion that minimally disrupts motifs."""
        if not tokens:
            return 0
        
        # Get critical positions (non-wildcards only)
        critical_positions = self._get_critical_positions(tokens)
        
        # Calculate safety scores for each insertion position
        position_scores = {}
        
        for i in range(len(tokens) + 1):
            safety_score = 1.0
            
            # Check if inserting here would disrupt critical positions
            # Insertion at position i affects positions i and beyond
            for crit_pos, importance in critical_positions.items():
                if crit_pos >= i:
                    # This insertion would shift this critical position
                    # Apply penalty based on importance
                    if importance > 0.9:  # Very critical
                        safety_score *= 0.05
                    elif importance > 0.8:  # Highly critical  
                        safety_score *= 0.3
                    elif importance > 0.7:  # Moderately critical
                        safety_score *= 0.6
            
            position_scores[i] = safety_score
        
        # Choose position with highest safety score, with some randomness
        safe_positions = [pos for pos, score in position_scores.items() if score > 0.5]
        if safe_positions:
            return random.choice(safe_positions)
        
        # If no safe positions, choose least bad option
        best_pos = max(position_scores.items(), key=lambda x: x[1])[0]
        return best_pos
    
    def _find_safe_deletion_position(self, tokens: List[str]) -> int:
        """Find a position for deletion that minimally disrupts motifs."""
        if not tokens:
            return 0
        
        # Get critical positions
        critical_positions = self._get_critical_positions(tokens)
        
        # Prefer deleting modifications first
        mod_positions = [i for i, token in enumerate(tokens) 
                        if token in self.modification_properties]
        
        # Filter modification positions by safety
        safe_mod_positions = [pos for pos in mod_positions 
                             if pos not in critical_positions or critical_positions[pos] < 0.8]
        if safe_mod_positions:
            return random.choice(safe_mod_positions)
        
        # Calculate safety scores for each deletion position
        position_scores = {}
        
        for i in range(len(tokens)):
            safety_score = 1.0
            
            # Check if this position is critical
            if i in critical_positions:
                importance = critical_positions[i]
                if importance > 0.9:  # Very critical - avoid deleting
                    safety_score = 0.05
                elif importance > 0.8:  # Highly critical
                    safety_score = 0.2
                elif importance > 0.7:  # Moderately critical
                    safety_score = 0.5
            
            position_scores[i] = safety_score
        
        # Choose position with highest safety score
        safe_positions = [pos for pos, score in position_scores.items() if score > 0.3]
        if safe_positions:
            return random.choice(safe_positions)
        
        # If no safe positions, choose least bad option
        best_pos = max(position_scores.items(), key=lambda x: x[1])[0]
        return best_pos
    
    def _is_wildcard_region(self, tokens: List[str], position: int) -> bool:
        """
        Check if a position falls within a wildcard region of any motif.
        
        Args:
            tokens: Sequence tokens
            position: Position to check
            
        Returns:
            True if position is in a wildcard region
        """
        if not tokens or position < 0 or position >= len(tokens):
            return False
        
        for motifs in self.critical_motifs.values():
            for motif in motifs:
                pattern = motif['pattern']
                motif_position = motif['position']
                
                # Handle negative positions
                if motif_position < 0:
                    start_pos = len(tokens) + motif_position
                else:
                    start_pos = motif_position
                
                # Check if position falls within this motif
                if start_pos <= position < start_pos + len(pattern):
                    pattern_index = position - start_pos
                    if 0 <= pattern_index < len(pattern):
                        # Check if this specific position is a wildcard
                        if pattern[pattern_index] == '*':
                            return True
        
        return False
    
    def _repair_chemical_issues(self, tokens: List[str]) -> List[str]:
        """Repair chemical plausibility issues."""
        repaired = tokens.copy()
        critical_positions = self._get_critical_positions(repaired)
        
        # Fix charge clustering
        consecutive_charged = self._count_consecutive_charged(repaired)
        if consecutive_charged > 3:
            # Find and break up charge clusters, avoiding critical positions
            charged_residues = set(['R', 'K', 'D', 'E', 'H'])
            for i in range(len(repaired) - 1):
                base_aa = self._get_base_amino_acid(repaired[i])
                if base_aa in charged_residues:
                    # Check if this position is critical
                    importance = critical_positions.get(i, 0.0)
                    if importance < 0.8 or self._is_wildcard_region(repaired, i):
                        # Safe to modify
                        neutral_options = ['A', 'G', 'S', 'T', 'N', 'Q']
                        repaired[i] = random.choice(neutral_options)
                        break
        
        return repaired
    
    def _repair_proteolysis_issues(self, tokens: List[str]) -> List[str]:
        """Repair proteolytic cleavage site issues."""
        repaired = tokens.copy()
        
        # Find and modify cleavage sites
        for i in range(len(repaired) - 1):
            # Check for trypsin sites (most problematic)
            if self._get_base_amino_acid(repaired[i]) in ['K', 'R']:
                # Replace with similar but non-cleavable residue
                if repaired[i] == 'K':
                    repaired[i] = 'Q'  # Similar but uncharged
                elif repaired[i] == 'R':
                    repaired[i] = 'Q'  # Similar but uncharged
                break
        
        return repaired
    
    def _repair_composition_issues(self, tokens: List[str]) -> List[str]:
        """Repair amino acid composition issues."""
        repaired = tokens.copy()
        
        # Identify over/under-represented amino acids
        aa_counts = {}
        for token in repaired:
            base_aa = self._get_base_amino_acid(token)
            aa_counts[base_aa] = aa_counts.get(base_aa, 0) + 1
        
        total_aa = len(repaired)
        
        # Find most over-represented amino acid
        max_deviation = 0
        worst_aa = None
        
        for aa in self.standard_aa:
            observed_freq = aa_counts.get(aa, 0) / total_aa
            expected_freq = self.natural_frequencies.get(aa, 0.05)
            deviation = observed_freq - expected_freq
            
            if deviation > max_deviation:
                max_deviation = deviation
                worst_aa = aa
        
        # Replace one instance of over-represented AA
        if worst_aa and max_deviation > 0.1:
            for i, token in enumerate(repaired):
                if self._get_base_amino_acid(token) == worst_aa:
                    # Replace with under-represented AA
                    under_represented = []
                    for aa in self.standard_aa:
                        obs_freq = aa_counts.get(aa, 0) / total_aa
                        exp_freq = self.natural_frequencies.get(aa, 0.05)
                        if obs_freq < exp_freq:
                            under_represented.append(aa)
                    
                    if under_represented:
                        repaired[i] = random.choice(under_represented)
                    break
        
        return repaired
    
    def get_weighted_random_residue(self) -> str:
        """Get a random residue based on type probabilities."""
        rand_val = random.random()
        
        if rand_val < self.mutation_weights['standard']:
            # Standard amino acid
            return random.choice(list(self.standard_aa))
        elif rand_val < self.mutation_weights['standard'] + self.mutation_weights['d_amino']:
            # D-amino acid
            d_amino_acids = [mod for mod, props in self.modification_properties.items() 
                           if props['type'] == 'd_amino']
            return random.choice(d_amino_acids)
        elif rand_val < (self.mutation_weights['standard'] + 
                        self.mutation_weights['d_amino'] + 
                        self.mutation_weights['unnatural']):
            # Unnatural amino acid
            unnatural_acids = [mod for mod, props in self.modification_properties.items() 
                             if props['type'] == 'unnatural']
            return random.choice(unnatural_acids)
        else:
            # Lipidation
            lipidations = [mod for mod, props in self.modification_properties.items() 
                         if props['type'] == 'lipidation']
            return random.choice(lipidations)
    
    def mutate_single_position(self, tokens: List[str], position: int) -> List[str]:
        """
        Mutate a single position in the sequence.
        
        Args:
            tokens: Current sequence tokens
            position: Position to mutate
            
        Returns:
            New sequence with mutation
        """
        if not tokens or position >= len(tokens):
            return tokens.copy()
        
        new_tokens = tokens.copy()
        
        # For lipidations, prefer to replace with standard amino acids
        if (tokens[position] in self.modification_properties and 
            self.modification_properties[tokens[position]]['type'] == 'lipidation'):
            # 70% chance to replace lipidation with standard AA
            if random.random() < 0.7:
                new_tokens[position] = random.choice(list(self.standard_aa))
            else:
                new_tokens[position] = self.get_weighted_random_residue()
        else:
            # Normal weighted mutation
            new_tokens[position] = self.get_weighted_random_residue()
        
        return new_tokens
    
    def insert_residue(self, tokens: List[str], position: int) -> List[str]:
        """
        Insert a residue at the specified position.
        
        Args:
            tokens: Current sequence tokens
            position: Position to insert at
            
        Returns:
            New sequence with insertion
        """
        new_tokens = tokens.copy()
        new_residue = self.get_weighted_random_residue()
        new_tokens.insert(position, new_residue)
        return new_tokens
    
    def delete_residue(self, tokens: List[str], position: int) -> List[str]:
        """
        Delete a residue at the specified position.
        
        Args:
            tokens: Current sequence tokens
            position: Position to delete
            
        Returns:
            New sequence with deletion
        """
        if not tokens or position >= len(tokens):
            return tokens.copy()
        
        new_tokens = tokens.copy()
        del new_tokens[position]
        return new_tokens
    
    def calculate_sequence_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate sequence similarity (0-1) using token-level comparison.
        
        Args:
            tokens1: First sequence tokens
            tokens2: Second sequence tokens
            
        Returns:
            Similarity score (0-1)
        """
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Use dynamic programming for sequence alignment
        len1, len2 = len(tokens1), len(tokens2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        # Calculate similarity
        max_len = max(len1, len2)
        edit_distance = dp[len1][len2]
        similarity = 1 - (edit_distance / max_len) if max_len > 0 else 1.0
        
        return similarity
    
    def generate_random_sequence(self, target_length: int) -> List[str]:
        """
        Generate a random valid peptide sequence.
        
        Args:
            target_length: Desired sequence length
            
        Returns:
            Random sequence tokens
        """
        tokens = []
        for _ in range(target_length):
            tokens.append(self.get_weighted_random_residue())
        
        # Ensure chemical plausibility
        max_attempts = 10
        attempts = 0
        while not self.is_chemically_plausible(tokens) and attempts < max_attempts:
            tokens = []
            for _ in range(target_length):
                tokens.append(self.get_weighted_random_residue())
            attempts += 1
        
        return tokens
    
    def evaluate_biological_plausibility(self, tokens: List[str]) -> Dict[str, float]:
        """
        Comprehensive biological plausibility evaluation with graduated scoring.
        
        Returns scores from 0.0 (implausible) to 1.0 (highly plausible)
        
        Args:
            tokens: Sequence tokens to evaluate
            
        Returns:
            Dictionary with plausibility scores and components
        """
        if not tokens:
            return {'overall': 0.0, 'chemical': 0.0, 'motifs': 0.0, 'proteolysis': 0.0, 'composition': 0.0}
        
        # Component scores
        chemical_score = self._evaluate_chemical_plausibility(tokens)
        motif_score = self._evaluate_motif_preservation(tokens)
        proteolysis_score = self._evaluate_proteolytic_stability(tokens)
        composition_score = self._evaluate_amino_acid_composition(tokens)
        
        # Weighted overall score
        overall_score = (
            chemical_score * 0.3 +      # Chemical constraints (30%)
            motif_score * 0.35 +        # Critical motifs (35%)
            proteolysis_score * 0.20 +  # Proteolytic stability (20%)
            composition_score * 0.15    # Natural composition (15%)
        )
        
        return {
            'overall': overall_score,
            'chemical': chemical_score,
            'motifs': motif_score,
            'proteolysis': proteolysis_score,
            'composition': composition_score
        }
    
    def _evaluate_chemical_plausibility(self, tokens: List[str]) -> float:
        """Enhanced chemical constraint evaluation."""
        if not tokens:
            return 0.0
        
        properties = self.calculate_properties(tokens)
        score = 1.0
        
        # Charge distribution penalty
        charge_density = abs(properties.charge) / properties.length
        if charge_density > 0.4:  # More than 40% charged
            score *= 0.3
        elif charge_density > 0.3:  # More than 30% charged
            score *= 0.6
        
        # Consecutive charge penalty
        consecutive_charged = self._count_consecutive_charged(tokens)
        if consecutive_charged > 4:
            score *= 0.2
        elif consecutive_charged > 3:
            score *= 0.5
        
        # Hydrophobic patch analysis
        hydrophobic_patches = self._analyze_hydrophobic_patches(tokens)
        if hydrophobic_patches['max_patch'] > 8:  # Too hydrophobic
            score *= 0.4
        elif hydrophobic_patches['max_patch'] > 6:
            score *= 0.7
        
        # Modification density
        mod_density = properties.modification_count / properties.length
        if mod_density > 0.4:  # More than 40% modified
            score *= 0.3
        elif mod_density > 0.3:  # More than 30% modified
            score *= 0.6
        
        # Lipidation compatibility
        lipidation_count = sum(1 for token in tokens 
                              if token in self.modification_properties and 
                              self.modification_properties[token]['type'] == 'lipidation')
        if lipidation_count > 2:
            score *= 0.01  # Very unfavorable
        elif lipidation_count > 1:
            score *= 0.3
        
        # Extreme hydrophobicity penalty
        if properties.hydrophobicity > 3.0:  # Very hydrophobic
            score *= 0.5
        elif properties.hydrophobicity < -2.0:  # Very hydrophilic
            score *= 0.6
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_motif_preservation(self, tokens: List[str]) -> float:
        """Evaluate preservation of critical binding motifs."""
        if not tokens:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Check each motif family
        for family_name, motifs in self.critical_motifs.items():
            family_score = self._check_motif_family(tokens, motifs)
            
            # Weight by family importance
            family_weight = 1.0
            if family_name == 'conserved_core':
                family_weight = 2.0  # Most important
            
            total_score += family_score * family_weight
            total_weight += family_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _check_motif_family(self, tokens: List[str], motifs: List[Dict]) -> float:
        """Check if sequence preserves motifs from a specific family."""
        if not motifs:
            return 1.0
        
        family_score = 0.0
        for motif in motifs:
            match_score = self._match_motif_pattern(tokens, motif)
            weighted_score = match_score * motif['importance']
            family_score += weighted_score
        
        return family_score / len(motifs)
    
    def _match_motif_pattern(self, tokens: List[str], motif: Dict) -> float:
        """Match a specific motif pattern in the sequence."""
        pattern = motif['pattern']
        position = motif['position']
        
        # Handle negative positions (from end)
        if position < 0:
            start_pos = len(tokens) + position
        else:
            start_pos = position
        
        # Check bounds
        if start_pos < 0 or start_pos + len(pattern) > len(tokens):
            return 0.0
        
        matches = 0
        for i, pattern_aa in enumerate(pattern):
            seq_pos = start_pos + i
            seq_token = tokens[seq_pos]
            
            # Convert modified amino acids to base form for comparison
            seq_aa = self._get_base_amino_acid(seq_token)
            
            if pattern_aa == '*':  # Wildcard
                matches += 1
            elif pattern_aa == seq_aa:
                matches += 1
            elif self._is_conservative_substitution(pattern_aa, seq_aa):
                matches += 0.7  # Partial credit for conservative substitutions
        
        return matches / len(pattern)
    
    def _evaluate_proteolytic_stability(self, tokens: List[str]) -> float:
        """Evaluate proteolytic stability by checking for cleavage sites."""
        if len(tokens) < 2:
            return 1.0
        
        penalty = 0.0
        
        # Check for each type of cleavage site
        for enzyme, patterns in self.cleavage_sites.items():
            for pattern in patterns:
                cleavage_count = self._count_cleavage_sites(tokens, pattern)
                
                # Penalty based on enzyme and site count
                if enzyme == 'trypsin':  # Most problematic
                    penalty += cleavage_count * 0.3
                elif enzyme == 'chymotrypsin':
                    penalty += cleavage_count * 0.2
                else:
                    penalty += cleavage_count * 0.1
        
        # Convert penalty to score (0-1)
        max_penalty = len(tokens) * 0.5  # Reasonable maximum
        stability_score = max(0.0, 1.0 - (penalty / max_penalty))
        
        return stability_score
    
    def _evaluate_amino_acid_composition(self, tokens: List[str]) -> float:
        """Evaluate amino acid composition against natural frequencies."""
        if not tokens:
            return 0.0
        
        # Count amino acids (convert modifications to base forms)
        aa_counts = {}
        for token in tokens:
            base_aa = self._get_base_amino_acid(token)
            aa_counts[base_aa] = aa_counts.get(base_aa, 0) + 1
        
        # Calculate frequency similarity to natural distribution
        total_aa = len(tokens)
        similarity_score = 0.0
        
        for aa in self.standard_aa:
            observed_freq = aa_counts.get(aa, 0) / total_aa
            expected_freq = self.natural_frequencies.get(aa, 0.05)
            
            # Calculate similarity (1 - normalized difference)
            freq_diff = abs(observed_freq - expected_freq)
            max_diff = max(observed_freq, expected_freq, 0.05)
            similarity = 1.0 - (freq_diff / max_diff)
            similarity_score += similarity
        
        return similarity_score / len(self.standard_aa)
    
    def _count_consecutive_charged(self, tokens: List[str]) -> int:
        """Count maximum consecutive charged residues."""
        charged_residues = set(['R', 'K', 'D', 'E', 'H'])
        
        max_consecutive = 0
        current_consecutive = 0
        
        for token in tokens:
            base_aa = self._get_base_amino_acid(token)
            
            # Check if residue is charged (including modifications)
            is_charged = base_aa in charged_residues
            if token in self.modification_properties:
                is_charged = is_charged or (self.modification_properties[token]['charge'] != 0)
            
            if is_charged:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _analyze_hydrophobic_patches(self, tokens: List[str]) -> Dict[str, int]:
        """Analyze hydrophobic patch distribution."""
        hydrophobic_residues = set(['F', 'I', 'L', 'M', 'V', 'W', 'Y'])
        
        max_patch = 0
        current_patch = 0
        patch_count = 0
        
        for token in tokens:
            base_aa = self._get_base_amino_acid(token)
            
            # Check hydrophobicity
            is_hydrophobic = base_aa in hydrophobic_residues
            if token in self.modification_properties:
                hydrophob = self.modification_properties[token]['hydrophobicity']
                is_hydrophobic = is_hydrophobic or (hydrophob > 2.0)
            
            if is_hydrophobic:
                current_patch += 1
                max_patch = max(max_patch, current_patch)
            else:
                if current_patch > 0:
                    patch_count += 1
                current_patch = 0
        
        if current_patch > 0:
            patch_count += 1
        
        return {'max_patch': max_patch, 'patch_count': patch_count}
    
    def _count_cleavage_sites(self, tokens: List[str], pattern: List[str]) -> int:
        """Count occurrences of a cleavage site pattern."""
        count = 0
        
        for i in range(len(tokens) - len(pattern) + 1):
            match = True
            for j, pattern_aa in enumerate(pattern):
                if pattern_aa != '*':
                    seq_aa = self._get_base_amino_acid(tokens[i + j])
                    if seq_aa != pattern_aa:
                        match = False
                        break
            
            if match:
                count += 1
        
        return count
    
    def _get_base_amino_acid(self, token: str) -> str:
        """Get the base amino acid from a token (handles modifications)."""
        if token in self.standard_aa:
            return token
        elif token in self.modification_properties:
            return self.modification_properties[token]['base_aa']
        else:
            return 'X'  # Unknown
    
    def _is_conservative_substitution(self, aa1: str, aa2: str) -> bool:
        """Check if amino acid substitution is conservative."""
        # Define conservative substitution groups
        conservative_groups = [
            set(['I', 'L', 'V']),        # Aliphatic
            set(['F', 'W', 'Y']),        # Aromatic
            set(['S', 'T']),             # Small polar
            set(['D', 'E']),             # Acidic
            set(['K', 'R']),             # Basic
            set(['N', 'Q']),             # Amide
        ]
        
        for group in conservative_groups:
            if aa1 in group and aa2 in group:
                return True
        
        return False