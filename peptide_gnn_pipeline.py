import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math

#this is the "encoding" part
class PeptideSequenceParser:
    """Parse peptide sequences with support for non-standard amino acids and modifications."""
    
    def __init__(self):
        # Standard amino acids
        self.standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        # Define patterns for different modifications
        self.modification_patterns = {
            'd_amino': r'\[d\w+\]',  # D-amino acids like [dNle]
            'single_letter_d': r'\[[a-z]\]',  # Single letter D-amino acids like [s], [a]
            'unnatural': r'\[Aib\]|\[Nle\]',  # Unnatural residues
            'lipidation': r'\[K\([^)]+\)\]',  # Lipidations like [K(yE-C16)]
        }
    
    def tokenize_sequence(self, sequence: str) -> List[str]:
        """
        Tokenize a peptide sequence into individual residues/modifications.
        
        Args:
            sequence: Peptide sequence string (may contain modifications in brackets)
            
        Returns:
            List of tokens (standard AAs or modifications)
        """
        if pd.isna(sequence) or not sequence.strip():
            return []
        
        sequence = sequence.strip()
        tokens = []
        i = 0
        
        while i < len(sequence):
            if sequence[i] == '[':
                # Find the closing bracket
                close_idx = sequence.find(']', i)
                if close_idx == -1:
                    # Malformed bracket, treat as regular character
                    tokens.append(sequence[i])
                    i += 1
                else:
                    # Extract the modification
                    modification = sequence[i:close_idx+1]
                    tokens.append(modification)
                    i = close_idx + 1
            elif sequence[i] in self.standard_aa:
                # Standard amino acid
                tokens.append(sequence[i])
                i += 1
            elif sequence[i] == '-':
                # Gap character, skip
                i += 1
            else:
                # Unknown character, skip or handle as needed
                i += 1
                
        return tokens
    
    def classify_modification(self, token: str) -> str:
        """
        Classify a modification token.
        
        Args:
            token: Token to classify
            
        Returns:
            Classification string ('standard', 'd_amino', 'unnatural', 'lipidation', 'unknown')
        """
        if token in self.standard_aa:
            return 'standard'
        
        for mod_type, pattern in self.modification_patterns.items():
            if re.match(pattern, token):
                return mod_type
                
        return 'unknown'


class PeptideFeatureEncoder:
    """Encode peptide residues and modifications into numerical features."""
    
    def __init__(self):
        # Standard amino acid properties
        self.aa_properties = {
            # Hydrophobicity (Kyte-Doolittle scale)
            'hydrophobicity': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            # Net charge at pH 7
            'charge': {
                'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
                'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
                'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
            },
            # Molecular weight (approximate)
            'molecular_weight': {
                'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
                'Q': 146, 'E': 147, 'G': 75, 'H': 155, 'I': 131,
                'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
                'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117
            }
        }
        
        # Non-standard amino acid properties (estimated based on structure)
        self.nsaa_properties = {
            '[dNle]': {'hydrophobicity': 3.8, 'charge': 0, 'molecular_weight': 131, 'is_d_amino': True, 'is_lipidated': False},
            '[s]': {'hydrophobicity': -0.8, 'charge': 0, 'molecular_weight': 105, 'is_d_amino': True, 'is_lipidated': False},
            '[a]': {'hydrophobicity': 1.8, 'charge': 0, 'molecular_weight': 89, 'is_d_amino': True, 'is_lipidated': False},
            '[Aib]': {'hydrophobicity': 1.8, 'charge': 0, 'molecular_weight': 103, 'is_d_amino': False, 'is_lipidated': False},
            '[Nle]': {'hydrophobicity': 3.8, 'charge': 0, 'molecular_weight': 131, 'is_d_amino': False, 'is_lipidated': False},
            '[hSer]': { 
            'hydrophobicity': -0.4,
            'charge': 0,
            'molecular_weight': 119,  
            'is_d_amino': False, 
            'is_lipidated': False
            },
            '[nVal]': { 
                'hydrophobicity': 3.8, 
                'charge': 0,
                'molecular_weight': 117,  
                'is_d_amino': False,
                'is_lipidated': False
            },
            '[Dap]': { 
                'hydrophobicity': -2.5,  
                'charge': 1,
                'molecular_weight': 104, 
                'is_d_amino': False,
                'is_lipidated': False
            },
            '[MetO]': {  
                'hydrophobicity': 0.3,   
                'charge': 0,
                'molecular_weight': 165, 
                'is_d_amino': False,
                'is_lipidated': False
            }
        }
        
        # Lipidation properties (base lysine + lipid chain effects)
        self.lipidation_properties = {
            '[K(yE-C16)]': {'hydrophobicity': 0.6, 'charge': -1.23, 'molecular_weight': 513, 'is_d_amino': False, 'is_lipidated': True},
            '[K(yE-yE-C16)]': {'hydrophobicity': -2.9, 'charge': -2.22, 'molecular_weight': 642, 'is_d_amino': False, 'is_lipidated': True},
            '[K(yE-C18)]': {'hydrophobicity': 1.6, 'charge': -1.23, 'molecular_weight': 541, 'is_d_amino': False, 'is_lipidated': True},
            '[K(OEG-OEG-yE-C18DA)]': {'hydrophobicity': -3.4, 'charge': -2.22, 'molecular_weight': 741.599, 'is_d_amino': False, 'is_lipidated': True},
            '[K(OEG-OEG-yE-C20DA)]': {'hydrophobicity': -2.4, 'charge': -2.22, 'molecular_weight': 769.63, 'is_d_amino': False, 'is_lipidated': True},
            '[K(eK-eK-yE-C20DA)]': {'hydrophobicity': -8.2, 'charge': -0.22, 'molecular_weight': 855.606, 'is_d_amino': False, 'is_lipidated': True},
        }

        self.feature_stats = None  # Will be computed from data

    def _get_raw_features(self, token: str) -> List[float]:
        """
        Extract raw (unnormalized) features from a token.
        
        Returns:
            [hydrophobicity, charge, molecular_weight] (first 3 features only)
        """
        # Standard amino acids
        if token in self.aa_properties['hydrophobicity']:
            return [
                self.aa_properties['hydrophobicity'][token],
                self.aa_properties['charge'][token],
                self.aa_properties['molecular_weight'][token]
            ]
        
        # Non-standard amino acids
        elif token in self.nsaa_properties:
            props = self.nsaa_properties[token]
            return [
                props['hydrophobicity'],
                props['charge'],
                props['molecular_weight']
            ]
        
        # Lipidations
        elif token in self.lipidation_properties:
            props = self.lipidation_properties[token]
            return [
                props['hydrophobicity'],
                props['charge'],
                props['molecular_weight']
            ]
        
        # D-amino acids pattern matching
        elif re.match(r'\[[a-z]\]', token):
            aa = token[1:-1].upper()
            if aa in self.aa_properties['hydrophobicity']:
                return [
                    self.aa_properties['hydrophobicity'][aa],
                    self.aa_properties['charge'][aa],
                    self.aa_properties['molecular_weight'][aa]
                ]
            else:
                return [0.0, 0.0, 100.0]  # Default for unknown D-amino acids
        
        # Other lipidation patterns
        elif re.match(r'\[K\([^)]+\)\]', token):
            return [8.5, 1.0, 450.0]  # Estimated values for unknown lipidations
        
        # Unknown tokens
        else:
            return [0.0, 0.0, 100.0]
    
    def _compute_normalization_stats(self, all_tokens: List[List[str]]):
        """Compute normalization stats from actual dataset."""
        all_features = {
            'hydrophobicity': [],
            'charge': [],
            'molecular_weight': []
        }
        
        for tokens in all_tokens:
            for token in tokens:
                raw_features = self._get_raw_features(token)
                all_features['hydrophobicity'].append(raw_features[0])
                all_features['charge'].append(raw_features[1])
                all_features['molecular_weight'].append(raw_features[2])
        
        self.feature_stats = {}
        for feature_name, values in all_features.items():
            self.feature_stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
        print(f"Computed normalization stats:")
        for name, stats in self.feature_stats.items():
            print(f"  {name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    def encode_token(self, token: str, position: int = 0, sequence_length: int = 1) -> List[float]:
        """Encode token with normalization of first 3 features only."""
        
        # Get base features including binary values
        if token in self.aa_properties['hydrophobicity']:
            base_features = [
                self.aa_properties['hydrophobicity'][token],
                self.aa_properties['charge'][token],
                self.aa_properties['molecular_weight'][token],
                0.0,  # is_d_amino
                0.0   # is_lipidated
            ]
        elif token in self.nsaa_properties:
            props = self.nsaa_properties[token]
            base_features = [
                props['hydrophobicity'],
                props['charge'],
                props['molecular_weight'],
                float(props['is_d_amino']),
                float(props['is_lipidated'])
            ]
        elif token in self.lipidation_properties:
            props = self.lipidation_properties[token]
            base_features = [
                props['hydrophobicity'],
                props['charge'],
                props['molecular_weight'],
                float(props['is_d_amino']),
                float(props['is_lipidated'])
            ]
        elif re.match(r'\[[a-z]\]', token):  # D-amino acids
            aa = token[1:-1].upper()
            if aa in self.aa_properties['hydrophobicity']:
                base_features = [
                    self.aa_properties['hydrophobicity'][aa],
                    self.aa_properties['charge'][aa],
                    self.aa_properties['molecular_weight'][aa],
                    1.0,  # is_d_amino
                    0.0   # is_lipidated
                ]
            else:
                base_features = [0.0, 0.0, 100.0, 1.0, 0.0]
        else:
            # Unknown token
            base_features = [0.0, 0.0, 100.0, 0.0, 0.0]
        
        # Normalize only first 3 features
        feature_names = ['hydrophobicity', 'charge', 'molecular_weight']
        normalized_features = base_features.copy()
        
        for i, name in enumerate(feature_names):
            normalized_features[i] = self._normalize_feature(base_features[i], name)
        
        # Add positional encoding
        pos = position / max(sequence_length - 1, 1)
        normalized_features.extend([math.sin(pos * math.pi), math.cos(pos * math.pi)])
        
        return normalized_features
    
    def _normalize_feature(self, value, feature_name):
        """Normalize using computed stats from actual data."""
        if self.feature_stats and feature_name in self.feature_stats:
            stats = self.feature_stats[feature_name]
            return (value - stats['mean']) / stats['std']
        return value  # Fallback if stats not computed yet

    
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        return ['hydrophobicity', 'charge', 'molecular_weight', 'is_d_amino', 'is_lipidated']


class PeptideGraphConstructor:
    """Construct graph representations of peptide sequences."""
    
    def __init__(self, feature_encoder: PeptideFeatureEncoder):
        self.feature_encoder = feature_encoder
    
    def sequence_to_graph(self, tokens: List[str]) -> Data:
        """
        Convert a tokenized peptide sequence to a PyTorch Geometric Data object.
        
        Args:
            tokens: List of amino acid/modification tokens
            
        Returns:
            PyTorch Geometric Data object representing the peptide graph
        """
        if not tokens:
            # Empty sequence, return empty graph
            return Data(
                x=torch.zeros((1, 5), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=1
            )
        
        # Encode node features
        node_features = []
        sequence_length = len(tokens)
        for i, token in enumerate(tokens):
            features = self.feature_encoder.encode_token(token, position=i, sequence_length=sequence_length)
            node_features.append(features)
            
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create linear chain edges (peptide bonds)
        edge_list = []
        for i in range(len(tokens) - 1):
            # Add bidirectional edges for peptide bonds
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Single node, no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, num_nodes=len(tokens))


class PeptideDataset(Dataset):
    """PyTorch Geometric Dataset for peptide sequences and receptor binding data."""
    
    def __init__(self, csv_file: str, high_affinity_cutoff_pM: float = 1000.0):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with peptide data
            high_affinity_cutoff_pM: EC50 cutoff for high affinity (pM)
        """
        super().__init__()
        
        self.csv_file = csv_file
        self.high_affinity_cutoff = high_affinity_cutoff_pM
        
        # Initialize components
        self.parser = PeptideSequenceParser()
        self.feature_encoder = PeptideFeatureEncoder()
        self.graph_constructor = PeptideGraphConstructor(self.feature_encoder)

        # Load and process data
        self.df = pd.read_csv(csv_file)

        # Compute normalization stats from actual data
        all_tokens = [self.parser.tokenize_sequence(row['sequence']) 
                    for _, row in self.df.iterrows()]
        self.feature_encoder._compute_normalization_stats(all_tokens)

        self.data_list = self._process_data()
    
    def _process_data(self) -> List[Data]:
        """Process the CSV data into graph objects."""
        data_list = []
        
        print(f"Processing {len(self.df)} peptide sequences...")
        
        for idx, row in self.df.iterrows():
            try:
                # Parse sequence
                sequence = row.get('sequence', '')
                tokens = self.parser.tokenize_sequence(sequence)
                
                if not tokens:
                    print(f"Warning: Empty sequence at index {idx}, skipping")
                    continue
                
                # Create graph
                graph = self.graph_constructor.sequence_to_graph(tokens)
                
                # Process targets
                targets = self._process_targets(row)
                
                # Add metadata
                graph.sequence_id = row.get('pep_ID', f'seq_{idx}')
                graph.original_sequence = sequence
                graph.tokens = tokens
                graph.y = targets
                
                data_list.append(graph)
                
            except Exception as e:
                print(f"Error processing sequence at index {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(data_list)} sequences")
        return data_list
    
    def _process_targets(self, row: pd.Series) -> torch.Tensor:
        """
        Process target values (EC50) into binary affinity labels.
        
        Args:
            row: DataFrame row containing target data
            
        Returns:
            Tensor with binary labels [GCGR, GLP1R, GIPR]
        """
        # Map columns to targets
        target_mapping = {
            'EC50_T1': 'GCGR',   # T1 -> GCGR  
            'EC50_T2': 'GLP1R',  # T2 -> GLP1R
            'EC50_T3': 'GIPR'    # T3 -> GIPR
        }
        
        targets = []
        for col, target_name in target_mapping.items():
            ec50_value = row.get(col, np.nan)
            
            if pd.isna(ec50_value):
                label = -1
            elif ec50_value < self.high_affinity_cutoff:
                label = 1
            elif ec50_value <= 50000:
                label = 0
            else:
                label = -1
            
            targets.append(label)
        return torch.tensor(targets, dtype=torch.long)
    
    def get_sequence_info(self, idx: int) -> Dict:
        """Get detailed information about a specific sequence."""
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range")
        
        data = self.data_list[idx]
        return {
            'sequence_id': data.sequence_id,
            'original_sequence': data.original_sequence,
            'tokens': data.tokens,
            'targets': data.y.numpy(),
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'node_features_shape': data.x.shape
        }
    
    def get_sequences_by_target_pattern(self, target_pattern: List[int]) -> List[int]:
        """
        Get sequence indices that match a specific target pattern.
        
        Args:
            target_pattern: List of target values [GCGR, GLP1R, GIPR] 
                          (-1 for missing, 0 for low, 1 for high)
        
        Returns:
            List of matching sequence indices
        """
        matching_indices = []
        
        for i, data in enumerate(self.data_list):
            targets = data.y.numpy()
            if np.array_equal(targets, target_pattern):
                matching_indices.append(i)
        
        return matching_indices
    
    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get a single sample by index."""
        return self.data_list[idx]
    
    def get_feature_info(self) -> Dict:
        """Return information about the features."""
        return {
            'feature_names': self.feature_encoder.get_feature_names(),
            'feature_dim': len(self.feature_encoder.get_feature_names()),
            'num_samples': len(self.data_list),
            'target_names': ['GCGR', 'GLP1R', 'GIPR'],
            'high_affinity_cutoff_pM': self.high_affinity_cutoff
        }
    
    def get_target_statistics(self) -> Dict:
        """Compute statistics about the target distribution."""
        if not self.data_list:
            return {}
        
        # Collect all targets
        all_targets = torch.stack([data.y for data in self.data_list])
        target_names = ['GCGR', 'GLP1R', 'GIPR']
        
        stats = {}
        for i, target_name in enumerate(target_names):
            target_col = all_targets[:, i]
            
            # Count valid (non -1) samples
            valid_mask = target_col != -1
            valid_targets = target_col[valid_mask]
            
            if len(valid_targets) > 0:
                high_affinity_count = (valid_targets == 1).sum().item()
                low_affinity_count = (valid_targets == 0).sum().item()
                total_valid = len(valid_targets)
                
                stats[target_name] = {
                    'total_samples': len(target_col),
                    'valid_samples': total_valid,
                    'missing_samples': len(target_col) - total_valid,
                    'high_affinity': high_affinity_count,
                    'low_affinity': low_affinity_count,
                    'high_affinity_percentage': (high_affinity_count / total_valid) * 100 if total_valid > 0 else 0
                }
            else:
                stats[target_name] = {
                    'total_samples': len(target_col),
                    'valid_samples': 0,
                    'missing_samples': len(target_col),
                    'high_affinity': 0,
                    'low_affinity': 0,
                    'high_affinity_percentage': 0
                }
        
        return stats


def load_peptide_dataset(csv_file: str, high_affinity_cutoff_pM: float = 1000.0) -> PeptideDataset:
    """
    Convenience function to load and create a peptide dataset.
    
    Args:
        csv_file: Path to CSV file
        high_affinity_cutoff_pM: EC50 cutoff for high affinity classification
        
    Returns:
        PeptideDataset object ready for GNN training
    """
    dataset = PeptideDataset(csv_file, high_affinity_cutoff_pM)
    
    # Print dataset information
    info = dataset.get_feature_info()
    stats = dataset.get_target_statistics()
    
    print(f"\n{'='*60}")
    print("PEPTIDE DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Number of sequences: {info['num_samples']}")
    print(f"Feature dimension: {info['feature_dim']}")
    print(f"Features: {', '.join(info['feature_names'])}")
    print(f"High affinity cutoff: {info['high_affinity_cutoff_pM']} pM")
    print(f"Targets: {', '.join(info['target_names'])}")
    
    print(f"\nTarget Distribution:")
    for target_name, target_stats in stats.items():
        print(f"  {target_name}:")
        print(f"    Valid samples: {target_stats['valid_samples']}/{target_stats['total_samples']}")
        print(f"    High affinity: {target_stats['high_affinity']} ({target_stats['high_affinity_percentage']:.1f}%)")
        print(f"    Low affinity: {target_stats['low_affinity']}")
    
    return dataset


def demonstrate_dataset_usage():
    """Demonstrate how to use the peptide dataset."""
    print("🧬 PEPTIDE GRAPH NEURAL NETWORK DATASET")
    print("="*60)
    
    # Load the dataset
    try:
        dataset = load_peptide_dataset('all_sequences_aligned.csv')
        
        # Show some examples
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Show first sample
            sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Sequence ID: {sample.sequence_id}")
            print(f"  Original sequence: {sample.original_sequence[:50]}...")
            print(f"  Tokens: {sample.tokens[:10]}...")
            print(f"  Graph shape: {sample.x.shape} nodes, {sample.edge_index.shape[1]} edges")
            print(f"  Targets (GCGR, GLP1R, GIPR): {sample.y}")
            
            # Show sequence parsing examples
            parser = PeptideSequenceParser()
            encoder = PeptideFeatureEncoder()
            
            print(f"\n{'='*60}")
            print("SEQUENCE PARSING EXAMPLES")
            print(f"{'='*60}")
            
            example_sequences = [
                "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                "H[s]EGTFTSDVSSYLEGQAAKEFIAWLVKGR[K(yE-C16)]",
                "H[dNle]EGTFT[Aib]DVSSYLEGQAAKEFIA[K(yE-yE-C16)]",
            ]
            
            for seq in example_sequences:
                tokens = parser.tokenize_sequence(seq)
                print(f"\nSequence: {seq}")
                print(f"Tokens: {tokens}")
                
                if tokens:
                    # Show feature encoding for first few tokens
                    print("Features for first 3 tokens:")
                    feature_names = encoder.get_feature_names()
                    for i, token in enumerate(tokens[:3]):
                        features = encoder.encode_token(token)
                        print(f"  {token}: {dict(zip(feature_names, features))}")
        
        print(f"\n✅ Dataset is ready for GNN training!")
        print(f"Use with PyTorch Geometric DataLoader for batching.")
        
        return dataset
        
    except FileNotFoundError:
        print("❌ File 'all_sequences_aligned.csv' not found.")
        print("Please ensure the data file is in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


if __name__ == "__main__":
    # Demonstrate the dataset
    dataset = demonstrate_dataset_usage()
