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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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
        """
        Create a publication-quality sequence alignment visualization
        using an upgraded version of the original code's logic.
        """
        print("\nCreating upgraded sequence alignment visualization...")

        # Define a new, more distinct color map for publication
        color_map = {
            'all_three': '#B30000',      # Darker Red
            'glp1_glucagon': '#FF8C00',  # Darker Orange
            'gip_glucagon': '#00BFFF',   # Deep Sky Blue
            'glp1_gip': '#DA70D6',       # Orchid
            'glp1_only': '#0000CD',      # Medium Blue
            'gip_only': '#008000',       # Dark Green
            'glucagon_only': '#4B0082',  # Indigo
            'unique': '#A9A9A9',         # Dark Gray
            'gap': 'white'
        }

        # Get sequences for alignment
        top_20_seqs = self.df.head(20)['sequence'].tolist()
        all_sequences = {
            **{f'Rank {i+1}': seq for i, seq in enumerate(top_20_seqs)},
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

        def get_color_key(token, position, seq_label):
            if token == '-':
                return 'gap'

            # Convert NSAA to standard AA for comparison (replicated logic)
            aa = token
            if token.startswith('[') and token.endswith(']'):
                if token.startswith('[K('):
                    aa = 'K'
                elif token in self.nsaa_mapping:
                    aa = self.nsaa_mapping[token]
                else:
                    aa = 'A'
            
            # Get native AAs at this position (replicated logic)
            native_aas = {}
            native_order = ['GLP1', 'GIP', 'Glucagon']
            for native_label in native_order:
                if native_label in self.native_sequences:
                    native_tokens = self._parse_sequence_with_nsaa(self.native_sequences[native_label])
                    if position < len(native_tokens):
                        native_token = native_tokens[position]
                        native_aa = native_token
                        if native_token.startswith('[') and native_token.endswith(']'):
                            if native_token.startswith('[K('):
                                native_aa = 'K'
                            elif native_token in self.nsaa_mapping:
                                native_aa = self.nsaa_mapping[native_token]
                            else:
                                native_aa = 'A'
                        native_aas[native_label] = native_aa
                    else:
                        native_aas[native_label] = '-'
            
            # Only color designed sequences (Rank X)
            if seq_label.startswith('Rank'):
                matches = [native_label for native_label, native_aa in native_aas.items() if aa == native_aa]

                if len(matches) == 3:
                    return 'all_three'
                elif len(matches) == 2:
                    if 'GLP1' in matches and 'Glucagon' in matches:
                        return 'glp1_glucagon'
                    elif 'GIP' in matches and 'Glucagon' in matches:
                        return 'gip_glucagon'
                    elif 'GLP1' in matches and 'GIP' in matches:
                        return 'glp1_gip'
                elif len(matches) == 1:
                    if 'GLP1' in matches:
                        return 'glp1_only'
                    elif 'GIP' in matches:
                        return 'gip_only'
                    elif 'Glucagon' in matches:
                        return 'glucagon_only'
                else:
                    return 'unique'
            return 'gap'

        # Create visualization
        fig, ax = plt.subplots(figsize=(18, 10))
        plt.style.use('seaborn-v0_8-whitegrid')

        for i, (label, tokens) in enumerate(zip(sequence_labels, alignment_data)):
            for j, token in enumerate(tokens):
                color_key = get_color_key(token, j, label)
                color = color_map.get(color_key, 'white')
                
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='#C0C0C0', linewidth=0.5)
                ax.add_patch(rect)
                
                if token != '-':
                    # Display mapped character for NSAAs, regular AA otherwise
                    if token.startswith('[') and token.endswith(']'):
                        display_token = self.nsaa_display.get(token, 'X')
                        fontsize = 7
                    else:
                        display_token = token
                        fontsize = 8
                    
                    # Use a color for the text that provides good contrast
                    text_color = 'black'
                    if color_key in ['all_three', 'glucagon_only']:
                        text_color = 'white'
                    
                    ax.text(j + 0.5, i + 0.5, display_token, ha='center', va='center',
                            fontsize=fontsize, weight='bold', color=text_color)

        # Set axes and labels
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, len(sequence_labels))
        ax.set_xticks(range(max_len))
        ax.set_xticklabels([str(pos + 1) for pos in range(max_len)], fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(len(sequence_labels))])
        ax.set_yticklabels([label.replace('_', ' ') for label in sequence_labels], va='center', fontsize=9)
        ax.tick_params(axis='both', which='both', length=0)

        ax.set_xlabel('Sequence Position', fontsize=12, weight='bold')
        ax.set_title('Alignment of Top 20 GA Peptides to Native Sequences', fontsize=16, weight='bold')

        # Enhanced legend with patches
        legend_elements = [
            mpatches.Patch(color=color_map['all_three'], label='All 3 Natives (GLP-1, GIP, Glucagon)'),
            mpatches.Patch(color=color_map['glp1_gip'], label='GLP-1 & GIP'),
            mpatches.Patch(color=color_map['glp1_glucagon'], label='GLP-1 & Glucagon'),
            mpatches.Patch(color=color_map['gip_glucagon'], label='GIP & Glucagon'),
            mpatches.Patch(color=color_map['glp1_only'], label='GLP-1 Only'),
            mpatches.Patch(color=color_map['gip_only'], label='GIP Only'),
            mpatches.Patch(color=color_map['glucagon_only'], label='Glucagon Only'),
            mpatches.Patch(color=color_map['unique'], label='Unique')
        ]
        ax.legend(handles=legend_elements, title='Conservation Pattern', 
                bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=12)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(self.output_dir / 'sequence_alignment_publication.png', dpi=600, bbox_inches='tight')
        plt.close()

        print("Upgraded sequence alignment visualization saved successfully.")

    def analyze_sequence_similarity_to_training(self):
        """Analyze highest sequence similarity for each top 20 candidate to training sequences."""
        print("\nAnalyzing sequence similarity to training dataset...")
        
        try:
            # Load training data
            train_df = pd.read_csv('all_sequences_aligned.csv')
            train_sequences = train_df['sequence'].tolist()
            print(f"Loaded {len(train_sequences)} training sequences")
            
            # Get our candidate sequences
            candidate_sequences = self.df['sequence'].tolist()
            
            # Similarity analyzer with flexible edit distance calculation
            class FlexibleSimilarityAnalyzer:
                def tokenize_sequence(self, sequence):
                    """Tokenize sequence preserving NSAA modifications."""
                    import re
                    pattern = r'\[[^\]]+\]|.'
                    return re.findall(pattern, sequence)
                
                def calculate_sequence_similarity(self, seq1, seq2):
                    """Calculate sequence similarity using edit distance (handles different lengths)."""
                    tokens1 = self.tokenize_sequence(seq1)
                    tokens2 = self.tokenize_sequence(seq2)
                    
                    if not tokens1 and not tokens2:
                        return 1.0
                    if not tokens1 or not tokens2:
                        return 0.0
                    
                    # Calculate edit distance using dynamic programming
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
                    
                    # Convert edit distance to similarity percentage
                    max_len = max(len1, len2)
                    edit_distance = dp[len1][len2]
                    similarity = 1 - (edit_distance / max_len) if max_len > 0 else 1.0
                    
                    return similarity
            
            analyzer = FlexibleSimilarityAnalyzer()
            
            # Calculate highest similarity for each candidate
            similarity_results = []
            
            for i, candidate_seq in enumerate(candidate_sequences):
                print(f"Processing candidate {i+1}/{len(candidate_sequences)}")
                
                max_similarity = 0.0
                best_match_id = ""
                best_match_seq = ""
                
                # Find maximum similarity to any training sequence for this candidate
                for j, train_seq in enumerate(train_sequences):
                    similarity = analyzer.calculate_sequence_similarity(candidate_seq, train_seq)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_seq = train_seq
                        if 'pep_ID' in train_df.columns:
                            best_match_id = train_df.iloc[j]['pep_ID']
                        else:
                            best_match_id = f"train_seq_{j}"
                
                similarity_results.append({
                    'peptide_rank': i + 1,
                    'sequence': candidate_seq,
                    'highest_similarity_percent': max_similarity * 100,
                    'best_match_id': best_match_id,
                    'best_match_sequence': best_match_seq,
                    'fitness_score': self.df.iloc[i]['fitness_score']
                })
            
            # Store results for summary report
            self.results['similarity_analysis'] = {
                'per_peptide_similarities': similarity_results
            }
            
            # Save detailed results to CSV
            similarity_df = pd.DataFrame(similarity_results)
            similarity_df.to_csv(self.output_dir / 'per_peptide_similarity_analysis.csv', index=False)
            
            print(f"Sequence similarity analysis completed!")
            print(f"Per-peptide similarity results:")
            for result in similarity_results:
                print(f"  Peptide {result['peptide_rank']}: {result['highest_similarity_percent']:.1f}% similarity")
            
        except Exception as e:
            print(f"Error in similarity analysis: {e}")
            print("Continuing with other analyses...")

    def generate_hybrid_visualization(self):
        """
        Generate a hybrid sequence alignment visualization with a detailed grid
        and a conservation bar chart below it. This provides both a qualitative
        and quantitative view of peptide conservation.
        """
        print("\nCreating hybrid sequence alignment visualization...")

        # Define the color map for the alignment grid
        color_map = {
            'all_three': '#B30000',      # Darker Red
            'glp1_glucagon': '#FF8C00',  # Darker Orange
            'gip_glucagon': '#00BFFF',   # Deep Sky Blue
            'glp1_gip': '#DA70D6',       # Orchid
            'glp1_only': '#0000CD',      # Medium Blue
            'gip_only': '#008000',       # Dark Green
            'glucagon_only': '#4B0082',  # Indigo
            'unique': '#A9A9A9',         # Dark Gray
            'gap': 'white'
        }

        # Get the sequences to be plotted (top 20 + 3 natives)
        top_peptides_count = 20
        top_peptides = self.df.head(top_peptides_count)['sequence'].tolist()
        sequences_to_plot = {
            **{f'Rank {i+1}': seq for i, seq in enumerate(top_peptides)},
            **self.native_sequences
        }
        
        # Parse sequences into tokens and find max length
        parsed_sequences = {}
        max_len = 0
        for label, seq in sequences_to_plot.items():
            tokens = self._parse_sequence_with_nsaa(seq)
            parsed_sequences[label] = tokens
            max_len = max(max_len, len(tokens))

        # Pad sequences to the same length for alignment
        alignment_data = []
        sequence_labels = []
        for label, tokens in parsed_sequences.items():
            padded_tokens = tokens + ['-'] * (max_len - len(tokens))
            alignment_data.append(padded_tokens)
            sequence_labels.append(label)

        # --- Conservation Score Calculation ---
        # This loop is specifically for the bar chart.
        # It counts how many of the top 20 peptides match at least one native sequence at each position.
        conservation_scores = [0] * max_len
        top_peptide_labels = [f'Rank {i+1}' for i in range(top_peptides_count)]
        native_labels = ['GLP1', 'GIP', 'Glucagon']

        for j in range(max_len):
            match_count = 0
            for peptide_label in top_peptide_labels:
                if peptide_label not in parsed_sequences:
                    continue

                peptide_token = parsed_sequences[peptide_label][j] if j < len(parsed_sequences[peptide_label]) else '-'
                
                # Convert NSAA to standard AA
                peptide_aa = peptide_token
                if peptide_token.startswith('[') and peptide_token.endswith(']'):
                    peptide_aa = self.nsaa_mapping.get(peptide_token, 'X')
                
                is_conserved = False
                for native_label in native_labels:
                    if native_label not in parsed_sequences:
                        continue

                    native_token = parsed_sequences[native_label][j] if j < len(parsed_sequences[native_label]) else '-'
                    
                    # Convert native NSAA to standard AA
                    native_aa = native_token
                    if native_token.startswith('[') and native_token.endswith(']'):
                        native_aa = self.nsaa_mapping.get(native_token, 'X')

                    if peptide_aa == native_aa and peptide_aa != '-':
                        is_conserved = True
                        break
                
                if is_conserved:
                    match_count += 1
            
            conservation_scores[j] = match_count

        # --- Plotting the figure with two subplots ---
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)
        
        # Top subplot for the alignment grid
        ax_align = fig.add_subplot(gs[0])
        # Bottom subplot for the bar chart, sharing the x-axis with the top plot
        ax_bar = fig.add_subplot(gs[1], sharex=ax_align)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot the alignment grid on the top subplot
        for i, (label, tokens) in enumerate(zip(sequence_labels, alignment_data)):
            for j, token in enumerate(tokens):
                if label.startswith('Rank'):
                    # For peptides, determine color based on conservation to natives
                    matches = 0
                    for native_label in native_labels:
                        # Check if the native sequence exists at this position
                        if j < len(parsed_sequences[native_label]):
                            native_token = parsed_sequences[native_label][j]
                        else:
                            native_token = '-'

                        peptide_aa = token
                        if token.startswith('[') and token.endswith(']'):
                            peptide_aa = self.nsaa_mapping.get(token, 'X')
                        
                        native_aa = native_token
                        if native_token.startswith('[') and native_token.endswith(']'):
                            native_aa = self.nsaa_mapping.get(native_token, 'X')

                        if peptide_aa == native_aa and peptide_aa != '-':
                            matches += 1

                    # Re-run the color key logic for the top 20 peptides
                    color_key = 'unique'
                    if matches == 3: color_key = 'all_three'
                    elif matches == 2:
                        is_glp1_glucagon = (j < len(parsed_sequences['GLP1']) and parsed_sequences['GLP1'][j] == peptide_aa and 
                                        j < len(parsed_sequences['Glucagon']) and parsed_sequences['Glucagon'][j] == peptide_aa)
                        is_gip_glucagon = (j < len(parsed_sequences['GIP']) and parsed_sequences['GIP'][j] == peptide_aa and 
                                        j < len(parsed_sequences['Glucagon']) and parsed_sequences['Glucagon'][j] == peptide_aa)
                        is_glp1_gip = (j < len(parsed_sequences['GLP1']) and parsed_sequences['GLP1'][j] == peptide_aa and 
                                    j < len(parsed_sequences['GIP']) and parsed_sequences['GIP'][j] == peptide_aa)
                        if is_glp1_glucagon: color_key = 'glp1_glucagon'
                        elif is_gip_glucagon: color_key = 'gip_glucagon'
                        elif is_glp1_gip: color_key = 'glp1_gip'
                    elif matches == 1:
                        is_glp1 = j < len(parsed_sequences['GLP1']) and parsed_sequences['GLP1'][j] == peptide_aa
                        is_gip = j < len(parsed_sequences['GIP']) and parsed_sequences['GIP'][j] == peptide_aa
                        is_glucagon = j < len(parsed_sequences['Glucagon']) and parsed_sequences['Glucagon'][j] == peptide_aa
                        if is_glp1: color_key = 'glp1_only'
                        elif is_gip: color_key = 'gip_only'
                        elif is_glucagon: color_key = 'glucagon_only'
                else:
                    # For native sequences, use a simple gap color
                    color_key = 'gap'
                
                color = color_map.get(color_key, 'white')
                
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='#C0C0C0', linewidth=0.5)
                ax_align.add_patch(rect)
                
                if token != '-':
                    display_token = self.nsaa_display.get(token, token)
                    text_color = 'black'
                    if color_key in ['all_three', 'glucagon_only']:
                        text_color = 'white'
                    ax_align.text(j + 0.5, i + 0.5, display_token, ha='center', va='center',
                                    fontsize=8, weight='bold', color=text_color)
        
        ax_align.set_xlim(0, max_len)
        ax_align.set_ylim(0, len(sequence_labels))
        ax_align.set_aspect('equal', adjustable='box')
        ax_align.set_yticks([i + 0.5 for i in range(len(sequence_labels))])
        ax_align.set_yticklabels([label.replace('_', ' ') for label in sequence_labels], va='center', fontsize=9)
        ax_align.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_align.set_title(f'Alignment of Top {top_peptides_count} Peptides to Native Sequences', fontsize=16, weight='bold', pad=20)
        
        # Plot the bar chart on the bottom subplot
        ax_bar.bar(range(max_len), conservation_scores, color='#006400', edgecolor='none')
        ax_bar.set_xlim(0, max_len)
        ax_bar.set_xticks(range(max_len))
        ax_bar.set_xticklabels([str(pos + 1) for pos in range(max_len)], fontsize=8)
        ax_bar.set_xlabel('Sequence Position', fontsize=12, weight='bold')
        ax_bar.set_ylabel('Num. Peptides Conserved', fontsize=10, weight='bold')
        ax_bar.set_ylim(0, top_peptides_count) # Y-axis scale up to 20 for the number of peptides
        ax_bar.set_title('Conservation Score per Position', fontsize=12, pad=10)

        # Add a single legend for the entire figure
        legend_elements = [
            mpatches.Patch(color=color_map['all_three'], label='All 3 Natives'),
            mpatches.Patch(color=color_map['glp1_gip'], label='GLP-1 & GIP'),
            mpatches.Patch(color=color_map['glp1_glucagon'], label='GLP-1 & Glucagon'),
            mpatches.Patch(color=color_map['gip_glucagon'], label='GIP & Glucagon'),
            mpatches.Patch(color=color_map['glp1_only'], label='GLP-1 Only'),
            mpatches.Patch(color=color_map['gip_only'], label='GIP Only'),
            mpatches.Patch(color=color_map['glucagon_only'], label='Glucagon Only'),
            mpatches.Patch(color=color_map['unique'], label='Unique')
        ]
        
        # Place the legend outside the plots
        fig.legend(handles=legend_elements, title='Conservation Pattern', 
                bbox_to_anchor=(0.85, 0.95), loc='upper left', fontsize=10, title_fontsize=12, frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(self.output_dir / 'hybrid_visualization.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print("Hybrid visualization saved successfully.")



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
        
            if 'similarity_analysis' in self.results:
                sim_data = self.results['similarity_analysis']['per_peptide_similarities']
                report.append("## Sequence Similarity to Training Data\n")
                report.append("### Per-Peptide Highest Similarity Matches")
                for result in sim_data:
                    report.append(f"- Peptide {result['peptide_rank']}: {result['highest_similarity_percent']:.1f}% similarity to {result['best_match_id']}")
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
        self.generate_hybrid_visualization()
        self.calculate_molecular_properties()
        self.create_property_visualizations()
        self.analyze_sequence_similarity_to_training()
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