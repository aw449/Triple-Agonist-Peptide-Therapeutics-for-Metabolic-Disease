"""
Comprehensive Peptide Analysis Pipeline
Analyzes genetically generated peptide sequences for drug-likeness and conservation patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# RDKit for molecular property calculations
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Molecular properties will be estimated.")
    RDKIT_AVAILABLE = False

# BioPython for sequence analysis
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: BioPython not available. Using basic calculations.")
    BIOPYTHON_AVAILABLE = False


class PeptideAnalyzer:
    """Main class for peptide sequence analysis."""
    
    def __init__(self, csv_file: str, output_dir: str = "synthetic_peptide_analysis_results"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Native reference sequences
        self.native_sequences = {
            'GIP': 'YAEGTFISDYSIAMDKIHQQDFVNWLLAQKGKKNDWKHNITQ',
            'GLP1': 'HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR',
            'Glucagon': 'HSQGTFTSDYSKYLDSRRAQDFVQWLMNT'
        }
        
        # NSAA to standard AA mapping
        self.nsaa_mapping = {
            '[dAla]': 'A', '[dNle]': 'L', '[s]': 'S', '[a]': 'A',
            '[Aib]': 'A', '[Nle]': 'L', '[Sar]': 'G',  '[hSer]': 'S',
            '[nVal]':'V',
            '[Dap]': 'K', 
            '[MetO]': 'M',
            
            '[K(yE-C16)]': 'K', '[K(yE-yE-C16)]': 'K', '[K(yE-C18)]': 'K',
            '[K(OEG-OEG-yE-C18DA)]': 'K', '[K(OEG-OEG-yE-C20DA)]': 'K',
            '[K(eK-eK-yE-C20DA)]': 'K'
        }
        
        # NSAA to display character mapping
        self.nsaa_display = {
            '[dAla]': 'B', '[dNle]': 'J', '[s]': 'O', '[a]': 'U',
            '[Aib]': 'X', '[Nle]': 'Z', '[Sar]': 'S1',
            '[K(yE-C16)]': '1', '[K(yE-yE-C16)]': '2', '[K(yE-C18)]': '3',
            '[K(OEG-OEG-yE-C18DA)]': '4', '[K(OEG-OEG-yE-C20DA)]': '5',
            '[K(eK-eK-yE-C20DA)]': '6',
            '[hSer]': 'hS',
            '[nVal]':'nV',
            '[Dap]': 'K1', 
            '[MetO]': 'MO',
        }
        
        self.df = None
        self.results = {}
        
    def load_and_parse_data(self):
        """Load CSV and parse complex columns."""
        print("Loading and parsing data...")
        
        self.df = pd.read_csv(self.csv_file)
        
        # Parse dictionary columns
        for col in ['predictions', 'plausibility_scores']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._safe_literal_eval)
        
        # Parse list columns
        if 'generation_history' in self.df.columns:
            self.df['generation_history'] = self.df['generation_history'].apply(self._safe_literal_eval)
            
        print(f"Loaded {len(self.df)} peptide sequences")
        return self.df
    
    def _parse_sequence_with_nsaa(self, sequence):
        """Parse sequence handling NSAAs in brackets as single units."""
        tokens = []
        i = 0
        while i < len(sequence):
            if sequence[i] == '[':
                # Find closing bracket
                end = sequence.find(']', i)
                if end != -1:
                    tokens.append(sequence[i:end+1])
                    i = end + 1
                else:
                    tokens.append(sequence[i])
                    i += 1
            else:
                tokens.append(sequence[i])
                i += 1
        return tokens
    
    def _convert_nsaa_to_standard(self, sequence):
        """Convert NSAAs to standard amino acids for molecular calculations."""
        tokens = self._parse_sequence_with_nsaa(sequence)
        standard_seq = ""
        
        for token in tokens:
            if token.startswith('[') and token.endswith(']'):
                # Use mapping if available
                if token in self.nsaa_mapping:
                    standard_seq += self.nsaa_mapping[token]
                else:
                    # Default mapping for unknown NSAAs
                    standard_seq += 'A'
            else:
                standard_seq += token
        
        return standard_seq
    
    def _safe_literal_eval(self, val):
        """Safely evaluate string representations of Python literals."""
        if pd.isna(val) or val == '':
            return {}
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return {}
        """Safely evaluate string representations of Python literals."""
        if pd.isna(val) or val == '':
            return {}
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return {}
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\nGenerating descriptive statistics...")
        
        stats = {}
        
        # Basic sequence statistics
        stats['basic'] = {
            'total_sequences': len(self.df),
            'unique_sequences': self.df['sequence'].nunique(),
            'avg_length': self.df['length'].mean(),
            'length_std': self.df['length'].std(),
            'min_length': self.df['length'].min(),
            'max_length': self.df['length'].max(),
            'avg_fitness': self.df['fitness_score'].mean(),
            'fitness_std': self.df['fitness_score'].std(),
            'min_fitness': self.df['fitness_score'].min(),
            'max_fitness': self.df['fitness_score'].max()
        }
        
        # Persistence analysis
        if 'total_generations_present' in self.df.columns:
            persistent_seqs = self.df[self.df['total_generations_present'] > 1]
            stats['persistence'] = {
                'multi_generation_count': len(persistent_seqs),
                'avg_generations_present': self.df['total_generations_present'].mean(),
                'max_generations_present': self.df['total_generations_present'].max()
            }
        
        # Plausibility scores analysis
        if 'plausibility_scores' in self.df.columns:
            plausibility_data = []
            for scores in self.df['plausibility_scores']:
                if isinstance(scores, dict):
                    plausibility_data.append(scores)
            
            if plausibility_data:
                plausibility_df = pd.DataFrame(plausibility_data)
                stats['plausibility'] = {
                    'avg_overall': plausibility_df.get('overall', pd.Series()).mean(),
                    'avg_chemical': plausibility_df.get('chemical', pd.Series()).mean(),
                    'avg_motifs': plausibility_df.get('motifs', pd.Series()).mean(),
                    'avg_proteolysis': plausibility_df.get('proteolysis', pd.Series()).mean(),
                    'avg_composition': plausibility_df.get('composition', pd.Series()).mean()
                }
        
        # Receptor binding predictions
        if 'predictions' in self.df.columns:
            predictions_data = []
            for preds in self.df['predictions']:
                if isinstance(preds, dict):
                    predictions_data.append(preds)
            
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                stats['binding'] = {}
                for receptor in ['GCGR', 'GLP1R', 'GIPR']:
                    if receptor in predictions_df.columns:
                        stats['binding'][f'avg_{receptor}'] = predictions_df[receptor].mean()
                        stats['binding'][f'high_affinity_{receptor}'] = (predictions_df[receptor] > 0.5).sum()
        
        self.results['descriptive_stats'] = stats
        
        # Save statistics
        with open(self.output_dir / 'descriptive_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Print summary
        print(f"Dataset Summary:")
        print(f"  Total sequences: {stats['basic']['total_sequences']}")
        print(f"  Average length: {stats['basic']['avg_length']:.1f} ± {stats['basic']['length_std']:.1f}")
        print(f"  Average fitness: {stats['basic']['avg_fitness']:.3f} ± {stats['basic']['fitness_std']:.3f}")
        
        return stats
    
    def find_conserved_motifs(self, motif_length=4):
        """Search for known evolutionary and experimentally confirmed motifs."""
        print(f"\nSearching for known critical binding motifs...")
        
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
        
        # Get top 20 sequences + native sequences
        top_sequences = self.df.head(20)['sequence'].tolist()
        all_sequences = top_sequences + list(self.native_sequences.values())
        sequence_labels = [f'Rank_{i+1}' for i in range(20)] + list(self.native_sequences.keys())
        
        # Count matches for each motif
        motif_counts = {}
        
        print(f"Analyzing {len(all_sequences)} sequences for critical motifs...")
        
        # Check each motif
        for family_name, motifs in self.critical_motifs.items():
            for motif in motifs:
                motif_key = f"{family_name}_{motif['name']}_pos{motif['position']}"  # ← FIX: Add position to make unique
                matching_sequences = []
                
                # Check each sequence
                for seq_idx, (sequence, label) in enumerate(zip(all_sequences, sequence_labels)):
                    # Skip native sequences - only count GA-generated sequences
                    if label in ['GIP', 'Glucagon', 'GLP1']:
                        continue
                    # Parse sequence into tokens (handling NSAAs)
                    tokens = self._parse_sequence_with_nsaa(sequence)
                    
                    # Convert NSAAs to standard amino acids for motif matching
                    standard_tokens = []
                    for token in tokens:
                        if token.startswith('[') and token.endswith(']'):
                            if token in self.nsaa_mapping:
                                standard_tokens.append(self.nsaa_mapping[token])
                            else:
                                standard_tokens.append('A')  # Default mapping
                        else:
                            standard_tokens.append(token)
                    
                    # Check if this sequence matches the motif
                    if self._matches_motif_pattern(standard_tokens, motif):
                        matching_sequences.append(label)
                
                motif_counts[motif_key] = {
                    'motif_name': motif['name'],
                    'family': family_name,
                    'pattern': ''.join(motif['pattern']),
                    'importance': motif['importance'],
                    'position': motif['position'],
                    'count': len(matching_sequences),
                    'matching_sequences': matching_sequences
                }
                
        # Print results
        print(f"\nCritical Motif Match Counts:")
        print("="*60)
        
        # Sort by count descending
        sorted_motifs = sorted(motif_counts.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for motif_key, data in sorted_motifs:
            if data['count'] > 0:
                print(f"{data['motif_name']} ({data['family']}):")
                print(f"  Pattern: {data['pattern']} at position {data['position']}")
                ga_sequences_count = len(all_sequences) - 3  # Subtract 3 native sequences
                print(f"  Matches: {data['count']}/{ga_sequences_count} GA sequences")
                print(f"  Sequences: {', '.join(data['matching_sequences'][:5])}")
                if len(data['matching_sequences']) > 5:
                    print(f"    ... and {len(data['matching_sequences']) - 5} more")
                print()
        
        self.results['critical_motifs'] = motif_counts
        return motif_counts

    def _matches_motif_pattern(self, tokens, motif):
        """
        Check if sequence matches motif pattern exactly (with wildcards).
        
        Args:
            tokens: List of amino acid tokens (standard AAs)
            motif: Motif dictionary with pattern, position, etc.
            
        Returns:
            True if pattern matches, False otherwise
        """
        pattern = motif['pattern']
        position = motif['position']
        
        # Handle negative positions (from end)
        if position < 0:
            start_pos = len(tokens) + position
        else:
            start_pos = position
        
        # Check bounds
        if start_pos < 0 or start_pos + len(pattern) > len(tokens):
            return False
        
        # Check each position in pattern
        for i, pattern_aa in enumerate(pattern):
            seq_pos = start_pos + i
            seq_aa = tokens[seq_pos]
            
            if pattern_aa == '*':  # Wildcard matches anything
                continue
            elif pattern_aa == seq_aa:  # Exact match
                continue
            else:  # No match
                return False
        
        return True
    
    def create_sequence_alignment_visualization(self):
        """Create colored sequence alignment visualization with pairwise combination colors."""
        print("\nCreating sequence alignment visualization...")
        
        # Get sequences for alignment
        top_20_seqs = self.df.head(20)['sequence'].tolist()
        all_sequences = {
            **{f'Rank_{i+1}': seq for i, seq in enumerate(top_20_seqs)},
            **self.native_sequences
        }
        
        # Parse sequences into tokens (handling NSAAs)
        parsed_sequences = {}
        max_len = 0
        
        for label, seq in all_sequences.items():
            tokens = self._parse_sequence_with_nsaa(seq)
            parsed_sequences[label] = tokens
            max_len = max(max_len, len(tokens))
        
        # Create alignment matrix
        alignment_data = []
        sequence_labels = []
        
        for label, tokens in parsed_sequences.items():
            padded_tokens = tokens + ['-'] * (max_len - len(tokens))
            alignment_data.append(padded_tokens)
            sequence_labels.append(label)
        
        # Enhanced color mapping for amino acids with pairwise combinations
        def get_aa_color(token, position, seq_label):
            if token == '-':
                return 'white'
            
            # Convert NSAA to standard for comparison
            if token.startswith('[') and token.endswith(']'):
                if token.startswith('[K('):
                    aa = 'K'
                elif token in self.nsaa_mapping:
                    aa = self.nsaa_mapping[token]
                else:
                    aa = 'A'
            else:
                aa = token
            
            # Get amino acids at this position for each native sequence
            native_aas = {}
            native_order = ['GLP1', 'GIP', 'Glucagon']
            
            for native_label in native_order:
                if native_label in self.native_sequences:
                    native_tokens = self._parse_sequence_with_nsaa(self.native_sequences[native_label])
                    if position < len(native_tokens):
                        native_token = native_tokens[position]
                        # Convert NSAA to standard for comparison
                        if native_token.startswith('[') and native_token.endswith(']'):
                            if native_token.startswith('[K('):
                                native_aas[native_label] = 'K'
                            elif native_token in self.nsaa_mapping:
                                native_aas[native_label] = self.nsaa_mapping[native_token]
                            else:
                                native_aas[native_label] = 'A'
                        else:
                            native_aas[native_label] = native_token
                    else:
                        native_aas[native_label] = '-'
            
            # Only color designed sequences (Rank_X), not the native references
            if seq_label.startswith('Rank_'):
                # Check which natives match this amino acid
                matches = []
                for native_label, native_aa in native_aas.items():
                    if aa == native_aa:
                        matches.append(native_label)
                
                # Color based on match pattern
                if len(matches) == 3:  # All three natives
                    return 'red'
                elif len(matches) == 2:  # Pairwise combinations
                    if 'GLP1' in matches and 'Glucagon' in matches:
                        return 'orange'  # GLP1 + Glucagon
                    elif 'GIP' in matches and 'Glucagon' in matches:
                        return 'cyan'    # GIP + Glucagon
                    elif 'GLP1' in matches and 'GIP' in matches:
                        return 'magenta' # GLP1 + GIP
                elif len(matches) == 1:  # Single matches
                    if 'GLP1' in matches:
                        return 'blue'
                    elif 'GIP' in matches:
                        return 'green'
                    elif 'Glucagon' in matches:
                        return 'purple'
            
            return 'lightgray'  # Unique or native sequence
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(22, 12))
        
        # Plot each sequence
        for i, (label, tokens) in enumerate(zip(sequence_labels, alignment_data)):
            for j, token in enumerate(tokens):
                color = get_aa_color(token, j, label)
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.1)
                ax.add_patch(rect)
                
                if token != '-':
                    # Display mapped character for NSAAs, regular AA otherwise
                    if token.startswith('[') and token.endswith(']'):
                        display_token = self.nsaa_display.get(token, 'X')
                        fontsize = 7
                    else:
                        display_token = token
                        fontsize = 8
                    
                    ax.text(j+0.5, i+0.5, display_token, ha='center', va='center', 
                        fontsize=fontsize, weight='bold')
        
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, len(sequence_labels))
        ax.set_yticks([i + 0.5 for i in range(len(sequence_labels))])
        ax.set_yticklabels([label.replace('_', ' ') for label in sequence_labels], va='center')
        ax.set_xlabel('Position')
        ax.set_title('Sequence Alignment: Top 20 GA Peptides vs Native Sequences')
        
        # Enhanced legend with pairwise combinations
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='All 3 Natives (GLP1+GIP+Glucagon)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', label='GLP1 + Glucagon'),
            plt.Rectangle((0,0),1,1, facecolor='cyan', label='GIP + Glucagon'), 
            plt.Rectangle((0,0),1,1, facecolor='magenta', label='GLP1 + GIP'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='GLP1 Only'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='GIP Only'),
            plt.Rectangle((0,0),1,1, facecolor='purple', label='Glucagon Only'),
            plt.Rectangle((0,0),1,1, facecolor='lightgray', label='Unique')
        ]
        ax.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_alignment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced sequence alignment visualization saved with pairwise conservation analysis")
        
        # Print NSAA mapping
        if self.nsaa_display:
            print("\nNSAA Display Character Mapping:")
            for nsaa, char in self.nsaa_display.items():
                print(f"  {nsaa} → {char}")
        
        # Print conservation statistics
        print("\nConservation Pattern Statistics:")
        print("="*50)
        
        # Analyze conservation patterns
        pattern_counts = {
            'all_three': 0,
            'glp1_glucagon': 0,
            'gip_glucagon': 0, 
            'glp1_gip': 0,
            'glp1_only': 0,
            'gip_only': 0,
            'glucagon_only': 0,
            'unique': 0
        }
        
        top_20_seqs = self.df.head(20)['sequence'].tolist()
        
        for seq_idx, seq in enumerate(top_20_seqs):
            tokens = self._parse_sequence_with_nsaa(seq)
            
            for pos, token in enumerate(tokens):
                # Convert NSAA to standard for comparison
                if token.startswith('[') and token.endswith(']'):
                    if token.startswith('[K('):
                        aa = 'K'
                    elif token in self.nsaa_mapping:
                        aa = self.nsaa_mapping[token]
                    else:
                        aa = 'A'
                else:
                    aa = token
                
                # Get native AAs at this position
                native_aas = {}
                for native_label in ['GLP1', 'GIP', 'Glucagon']:
                    if native_label in self.native_sequences:
                        native_tokens = self._parse_sequence_with_nsaa(self.native_sequences[native_label])
                        if pos < len(native_tokens):
                            native_token = native_tokens[pos]
                            if native_token.startswith('[') and native_token.endswith(']'):
                                if native_token.startswith('[K('):
                                    native_aas[native_label] = 'K'
                                elif native_token in self.nsaa_mapping:
                                    native_aas[native_label] = self.nsaa_mapping[native_token]
                                else:
                                    native_aas[native_label] = 'A'
                            else:
                                native_aas[native_label] = native_token
                
                # Count matches
                matches = []
                for native_label, native_aa in native_aas.items():
                    if aa == native_aa:
                        matches.append(native_label)
                
                # Categorize pattern
                if len(matches) == 3:
                    pattern_counts['all_three'] += 1
                elif len(matches) == 2:
                    if 'GLP1' in matches and 'Glucagon' in matches:
                        pattern_counts['glp1_glucagon'] += 1
                    elif 'GIP' in matches and 'Glucagon' in matches:
                        pattern_counts['gip_glucagon'] += 1
                    elif 'GLP1' in matches and 'GIP' in matches:
                        pattern_counts['glp1_gip'] += 1
                elif len(matches) == 1:
                    if 'GLP1' in matches:
                        pattern_counts['glp1_only'] += 1
                    elif 'GIP' in matches:
                        pattern_counts['gip_only'] += 1
                    elif 'Glucagon' in matches:
                        pattern_counts['glucagon_only'] += 1
                else:
                    pattern_counts['unique'] += 1
        
        # Print statistics
        total_positions = sum(pattern_counts.values())
        
        print(f"Total analyzed positions: {total_positions}")
        print(f"All 3 natives:     {pattern_counts['all_three']:4d} ({100*pattern_counts['all_three']/total_positions:5.1f}%)")
        print(f"GLP1 + Glucagon:   {pattern_counts['glp1_glucagon']:4d} ({100*pattern_counts['glp1_glucagon']/total_positions:5.1f}%)")
        print(f"GIP + Glucagon:    {pattern_counts['gip_glucagon']:4d} ({100*pattern_counts['gip_glucagon']/total_positions:5.1f}%)")
        print(f"GLP1 + GIP:        {pattern_counts['glp1_gip']:4d} ({100*pattern_counts['glp1_gip']/total_positions:5.1f}%)")
        print(f"GLP1 only:         {pattern_counts['glp1_only']:4d} ({100*pattern_counts['glp1_only']/total_positions:5.1f}%)")
        print(f"GIP only:          {pattern_counts['gip_only']:4d} ({100*pattern_counts['gip_only']/total_positions:5.1f}%)")
        print(f"Glucagon only:     {pattern_counts['glucagon_only']:4d} ({100*pattern_counts['glucagon_only']/total_positions:5.1f}%)")
        print(f"Unique:            {pattern_counts['unique']:4d} ({100*pattern_counts['unique']/total_positions:5.1f}%)")
        
    def calculate_molecular_properties(self):
        """Calculate molecular properties for all sequences."""
        print("\nCalculating molecular properties...")
        
        properties = []
        
        # Calculate for all sequences
        for _, row in self.df.iterrows():
            props = self._calculate_peptide_properties(row['sequence'])
            props['sequence_type'] = 'GA_Generated'
            props['rank'] = row['rank']
            properties.append(props)
        
        # Calculate for native sequences
        for name, seq in self.native_sequences.items():
            props = self._calculate_peptide_properties(seq)
            props['sequence_type'] = 'Native'
            props['rank'] = name
            properties.append(props)
        
        properties_df = pd.DataFrame(properties)
        properties_df.to_csv(self.output_dir / 'molecular_properties.csv', index=False)
        
        self.results['molecular_properties'] = properties_df
        return properties_df
    
    def _calculate_peptide_properties(self, sequence):
        """Calculate molecular properties for a single peptide."""
        # Convert NSAAs to standard amino acids for calculations
        standard_sequence = self._convert_nsaa_to_standard(sequence)
        
        properties = {'sequence': sequence, 'standard_sequence': standard_sequence}
        
        if BIOPYTHON_AVAILABLE:
            try:
                analysis = ProteinAnalysis(standard_sequence)
                properties['molecular_weight'] = analysis.molecular_weight()
                properties['isoelectric_point'] = analysis.isoelectric_point()
                properties['instability_index'] = analysis.instability_index()
                properties['gravy'] = analysis.gravy()  # Hydrophobicity
            except:
                properties['molecular_weight'] = self._estimate_mw(standard_sequence)
                properties['isoelectric_point'] = 7.0
                properties['instability_index'] = 40.0
                properties['gravy'] = 0.0
        else:
            properties['molecular_weight'] = self._estimate_mw(standard_sequence)
            properties['isoelectric_point'] = 7.0
            properties['instability_index'] = 40.0
            properties['gravy'] = 0.0
        
        # Estimate LogP and PSA (simplified calculations)
        properties['estimated_logP'] = self._estimate_logP(standard_sequence)
        properties['estimated_PSA'] = self._estimate_PSA(standard_sequence)
        
        return properties
    
    def _estimate_mw(self, sequence):
        """Estimate molecular weight using average amino acid weights."""
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        weight = sum(aa_weights.get(aa, 110.0) for aa in sequence)
        weight -= (len(sequence) - 1) * 18.015  # Subtract water for peptide bonds
        return weight
    
    def _estimate_logP(self, sequence):
        """Estimate LogP using amino acid hydrophobicity values."""
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        total_hydrophobicity = sum(hydrophobicity.get(aa, 0) for aa in sequence)
        return total_hydrophobicity / len(sequence) if sequence else 0
    
    def _estimate_PSA(self, sequence):
        """Estimate polar surface area based on polar amino acids."""
        polar_aas = set('RNDQEHKST')
        polar_count = sum(1 for aa in sequence if aa in polar_aas)
        return polar_count * 50  # Rough estimate: 50 Ų per polar residue
    
    def create_property_visualizations(self):
        """Create boxplots and distributions of molecular properties."""
        print("\nCreating molecular property visualizations...")
        
        if 'molecular_properties' not in self.results:
            self.calculate_molecular_properties()
        
        df = self.results['molecular_properties']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        properties = ['molecular_weight', 'estimated_logP', 'estimated_PSA', 
                    'isoelectric_point', 'instability_index', 'gravy']
        
        # Subplot labels
        subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        
        # Colors for different peptide types (glucagon, GLP-1, GIP)
        peptide_colors = ['red', 'blue', 'green']
        
        for i, prop in enumerate(properties):
            if i < 6:
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                
                # Separate GA generated and native sequences
                ga_data = df[df['sequence_type'] == 'GA_Generated'][prop].dropna()
                native_data = df[df['sequence_type'] == 'Native'][prop].dropna()
                
                # Create boxplot
                box_data = [ga_data, native_data]
                labels = ['GA Generated', 'Native']
                
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                # Mark individual native sequences with different colors
                native_df = df[df['sequence_type'] == 'Native']
                for j, (_, row) in enumerate(native_df.iterrows()):
                    color = peptide_colors[j % len(peptide_colors)]
                    ax.scatter(2, row[prop], c=color, s=100, alpha=0.8, 
                            label=row['rank'] if i == 0 else "")
                
                # Add subplot label and title
                ax.set_title(f'{subplot_labels[i]} {prop.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                
                # Add legend for first subplot
                if i == 0:
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'molecular_properties_comparison.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
        plt.title('Molecular Properties Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'properties_correlation.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Molecular property visualizations saved")
    
    def generate_summary_report(self):
        """Generate comprehensive markdown summary report."""
        print("\nGenerating summary report...")
        
        report = []
        report.append("# Synthetic Peptide Analysis Report\n")
        report.append(f"Analysis of {len(self.df)} genetically generated peptide sequences\n")
        
        # Descriptive Statistics
        if 'descriptive_stats' in self.results:
            stats = self.results['descriptive_stats']
            report.append("## Descriptive Statistics\n")
            report.append("### Basic Properties")
            report.append(f"- Total sequences: {stats['basic']['total_sequences']}")
            report.append(f"- Unique sequences: {stats['basic']['unique_sequences']}")
            report.append(f"- Average length: {stats['basic']['avg_length']:.1f} ± {stats['basic']['length_std']:.1f}")
            report.append(f"- Length range: {stats['basic']['min_length']} - {stats['basic']['max_length']}")
            report.append(f"- Average fitness: {stats['basic']['avg_fitness']:.3f} ± {stats['basic']['fitness_std']:.3f}")
            report.append(f"- Fitness range: {stats['basic']['min_fitness']:.3f} - {stats['basic']['max_fitness']:.3f}\n")
            
            if 'persistence' in stats:
                report.append("### Sequence Persistence")
                report.append(f"- Multi-generation sequences: {stats['persistence']['multi_generation_count']}")
                report.append(f"- Average generations present: {stats['persistence']['avg_generations_present']:.1f}")
                report.append(f"- Maximum generations: {stats['persistence']['max_generations_present']}\n")
            
            if 'plausibility' in stats:
                report.append("### Plausibility Scores")
                for metric, value in stats['plausibility'].items():
                    if pd.notna(value):
                        report.append(f"- {metric.replace('avg_', '').title()}: {value:.3f}")
                report.append("")
        
        # Conserved Motifs
        if 'conserved_motifs' in self.results:
            motifs = self.results['conserved_motifs']['motifs']
            report.append("## Conserved Motifs\n")
            report.append(f"Found {len(motifs)} conserved motifs (length {self.results['conserved_motifs']['motif_length']}):\n")
            for motif, count in sorted(motifs.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"- {motif}: {count} sequences")
            report.append("")
        
        # Molecular Properties Summary
        if 'molecular_properties' in self.results:
            df = self.results['molecular_properties']
            ga_df = df[df['sequence_type'] == 'GA_Generated']
            native_df = df[df['sequence_type'] == 'Native']
            
            report.append("## Molecular Properties Summary\n")
            report.append("### GA Generated Sequences")
            report.append(f"- Average MW: {ga_df['molecular_weight'].mean():.1f} ± {ga_df['molecular_weight'].std():.1f} Da")
            report.append(f"- Average LogP: {ga_df['estimated_logP'].mean():.2f} ± {ga_df['estimated_logP'].std():.2f}")
            report.append(f"- Average PSA: {ga_df['estimated_PSA'].mean():.1f} ± {ga_df['estimated_PSA'].std():.1f} Ų")
            
            report.append("\n### Native Sequences Comparison")
            for _, row in native_df.iterrows():
                report.append(f"- {row['rank']}: MW={row['molecular_weight']:.1f} Da, "
                            f"LogP={row['estimated_logP']:.2f}, PSA={row['estimated_PSA']:.1f} Ų")
            report.append("")
        
        # Top sequences
        report.append("## Top 10 Sequences\n")
        for i, row in self.df.head(10).iterrows():
            report.append(f"{row['rank']}. {row['sequence']}")
            report.append(f"   - Fitness: {row['fitness_score']:.3f}, Length: {row['length']}")
            if isinstance(row['predictions'], dict):
                preds = row['predictions']
                pred_str = ", ".join([f"{k}={v:.3f}" for k, v in preds.items() if pd.notna(v)])
                report.append(f"   - Predictions: {pred_str}")
            report.append("")
        
        # Save report
        with open(self.output_dir / 'analysis_summary.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("Summary report saved to analysis_summary.md")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive peptide analysis...")
        
        # Load data
        self.load_and_parse_data()
        
        # Run all analyses
        self.descriptive_statistics()
        self.find_conserved_motifs()
        self.create_sequence_alignment_visualization()
        self.calculate_molecular_properties()
        self.create_property_visualizations()
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file_path in sorted(self.output_dir.glob('*')):
            print(f"  - {file_path.name}")


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = PeptideAnalyzer('top_20_overall_unique_peptides_20250810_122150.csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()