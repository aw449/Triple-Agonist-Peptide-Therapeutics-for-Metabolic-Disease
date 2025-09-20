"""
GAT Model Validation Script with Similarity-Based Filtering
Validates trained GAT models on independent validation sequences with two evaluation modes:
1. Novel sequences only (≤80% similarity to training)
2. Complete validation set
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import re
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, roc_curve, precision_recall_curve
)
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Import existing components
from peptide_gnn_pipeline import PeptideSequenceParser, PeptideFeatureEncoder, PeptideGraphConstructor
from multitask_gnn_trainer import TransferLearningGNN
from torch_geometric.data import Data, Batch


class NSAAMapper:
    """Maps non-standard amino acids to standard amino acids."""
    
    def __init__(self):
        # Mapping of NSAA to closest standard AA
        self.nsaa_mapping = {
            '[hSer]': 'S',      # Homoserine -> Serine
            '[nVal]': 'V',      # Norvaline -> Valine  
            '[Dap]': 'K',       # Diaminopropionic acid -> Lysine
            '[MetO]': 'M',      # Methionine sulfoxide -> Methionine
            '[dNle]': 'L',      # D-Norleucine -> Leucine
            '[s]': 'S',         # D-Serine -> Serine
            '[a]': 'A',         # D-Alanine -> Alanine
            '[Aib]': 'A',       # α-Aminoisobutyric acid -> Alanine
            '[Nle]': 'L',       # Norleucine -> Leucine
        }
        
        # Keep known lipidations as-is (they're handled by the feature encoder)
        self.known_lipidations = {
            '[K(yE-C16)]', '[K(yE-yE-C16)]', '[K(yE-C18)]',
            '[K(OEG-OEG-yE-C18DA)]', '[K(OEG-OEG-yE-C20DA)]',
            '[K(eK-eK-yE-C20DA)]'
        }
    
    def map_sequence(self, sequence: str) -> str:
        """Map NSAA in sequence to standard amino acids."""
        mapped_sequence = sequence
        
        # Find all bracketed modifications
        pattern = r'\[[^\]]+\]'
        modifications = re.findall(pattern, sequence)
        
        for mod in modifications:
            if mod in self.nsaa_mapping:
                # Replace with standard AA
                mapped_sequence = mapped_sequence.replace(mod, self.nsaa_mapping[mod])
            elif mod in self.known_lipidations:
                # Keep known lipidations
                continue
            else:
                # Unknown modification - map to closest standard AA based on content
                if 'K(' in mod:  # Lysine modification
                    mapped_sequence = mapped_sequence.replace(mod, 'K')
                elif any(aa in mod.upper() for aa in 'ARNDCQEGHILKMFPSTWYV'):
                    # Extract first standard AA found
                    for aa in 'ARNDCQEGHILKMFPSTWYV':
                        if aa in mod.upper():
                            mapped_sequence = mapped_sequence.replace(mod, aa)
                            break
                else:
                    # Default to Alanine
                    mapped_sequence = mapped_sequence.replace(mod, 'A')
        
        return mapped_sequence


class SequenceSimilarityAnalyzer:
    """Analyzes sequence similarity between validation and training sets."""
    
    def __init__(self):
        self.parser = PeptideSequenceParser()
        self.nsaa_mapper = NSAAMapper()
    
    def calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity percentage."""
        # Tokenize sequences BEFORE mapping (preserve NSAA differences)
        tokens1 = self.parser.tokenize_sequence(seq1)
        tokens2 = self.parser.tokenize_sequence(seq2)
        
        if len(tokens1) != len(tokens2):
            return 0.0
        
        matches = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2)
        return matches / len(tokens1) if tokens1 else 0.0
    
    def calculate_similarity_matrix(self, val_sequences: List[str], 
                                  train_sequences: List[str]) -> np.ndarray:
        """Calculate similarity matrix between validation and training sequences."""
        n_val = len(val_sequences)
        n_train = len(train_sequences)
        similarity_matrix = np.zeros((n_val, n_train))
        
        for i, val_seq in enumerate(val_sequences):
            for j, train_seq in enumerate(train_sequences):
                similarity_matrix[i, j] = self.calculate_sequence_identity(val_seq, train_seq)
        
        return similarity_matrix
    
    def find_exact_matches(self, val_sequences: List[str], 
                          train_sequences: List[str]) -> List[int]:
        """Find indices of validation sequences that exactly match training sequences."""
        exact_match_indices = []
        
        for i, val_seq in enumerate(val_sequences):
            # Tokenize BEFORE mapping (preserve NSAA differences)
            val_tokens = self.parser.tokenize_sequence(val_seq)
            
            for train_seq in train_sequences:
                train_tokens = self.parser.tokenize_sequence(train_seq)
                
                if val_tokens == train_tokens:
                    exact_match_indices.append(i)
                    break
        
        return exact_match_indices
    
    def calculate_max_similarities(self, val_sequences: List[str], 
                                 train_sequences: List[str]) -> List[float]:
        """Calculate maximum similarity for each validation sequence to training set."""
        max_similarities = []
        
        print(f"Computing similarities for {len(val_sequences)} validation sequences...")
        
        for i, val_seq in enumerate(val_sequences):
            if i % 10 == 0:
                print(f"  Processing sequence {i+1}/{len(val_sequences)}")
            
            # Find maximum similarity to any training sequence
            max_sim = 0.0
            for train_seq in train_sequences:
                sim = self.calculate_sequence_identity(val_seq, train_seq)
                max_sim = max(max_sim, sim)
            max_similarities.append(max_sim)
        
        return max_similarities


class GATEnsemblePredictor:
    """Ensemble predictor using multiple GAT model folds."""
    
    def __init__(self, model_dir: str, training_data_file: str = 'all_sequences_aligned.csv'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.parser = PeptideSequenceParser()
        self.feature_encoder = PeptideFeatureEncoder()
        self.graph_constructor = PeptideGraphConstructor(self.feature_encoder)
        self.nsaa_mapper = NSAAMapper()
        
        # CRITICAL: Compute normalization stats from training data
        self._setup_feature_normalization(training_data_file)
        self._load_models()
    
    def _setup_feature_normalization(self, training_data_file: str):
        """Setup feature normalization using the same stats as training."""
        print("Computing feature normalization stats from training data...")
        
        try:
            # Load training data
            train_df = pd.read_csv(training_data_file)
            
            # Get all training sequences and compute normalization stats
            all_tokens = []
            for _, row in train_df.iterrows():
                sequence = row.get('sequence', '')
                if sequence and isinstance(sequence, str):
                    # Map NSAA just like we do for validation
                    mapped_sequence = self.nsaa_mapper.map_sequence(sequence)
                    tokens = self.parser.tokenize_sequence(mapped_sequence)
                    all_tokens.append(tokens)
            
            # Compute normalization stats (same as training)
            self.feature_encoder._compute_normalization_stats(all_tokens)
            print("✅ Feature normalization stats computed successfully")
            
        except Exception as e:
            print(f"❌ Warning: Could not compute normalization stats: {e}")
            print("Features will not be properly normalized!")
    
    def _load_models(self):
        """Load all fold models."""
        model_files = list(self.model_dir.glob("fold_*_final_model.pth"))
        model_files.sort()
        
        print(f"Loading {len(model_files)} GAT models...")
        
        for model_path in model_files:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Get model config
                model_config = checkpoint.get('model_config', {
                    'num_node_features': 7,
                    'hidden_dim': 96,
                    'num_layers': 4,
                    'dropout': 0.2,
                    'use_attention': True
                })
                
                # Initialize and load model
                model = TransferLearningGNN(**model_config).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Activate all heads
                model.activate_head('gcgr')
                model.activate_head('glp1r')
                model.activate_head('gipr')
                model.eval()
                
                self.models.append(model)
                print(f"✅ Loaded {model_path.name}")
                
            except Exception as e:
                print(f"❌ Failed to load {model_path.name}: {e}")
        
        if not self.models:
            raise RuntimeError("Failed to load any models")
        
        print(f"Loaded {len(self.models)} models successfully")
    
    def sequence_to_graph(self, sequence: str) -> Data:
        """Convert sequence to graph, mapping NSAA to standard AAs."""
        # Map NSAA to standard AAs
        mapped_sequence = self.nsaa_mapper.map_sequence(sequence)
        
        # Tokenize and create graph
        tokens = self.parser.tokenize_sequence(mapped_sequence)
        return self.graph_constructor.sequence_to_graph(tokens)
    
    def predict_sequences(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """Predict binding probabilities for sequences using ensemble."""
        all_predictions = {'GCGR': [], 'GLP1R': [], 'GIPR': []}
        
        for sequence in sequences:
            # Convert to graph
            graph = self.sequence_to_graph(sequence)
            batch = Batch.from_data_list([graph]).to(self.device)
            
            # Collect predictions from all models
            seq_predictions = {'GCGR': [], 'GLP1R': [], 'GIPR': []}
            
            with torch.no_grad():
                for model in self.models:
                    outputs = model(batch.x, batch.edge_index, batch.batch, 
                                  ['GCGR', 'GLP1R', 'GIPR'])
                    
                    for receptor, logits in outputs.items():
                        probability = torch.sigmoid(logits).item()
                        seq_predictions[receptor].append(probability)
            
            # Average across ensemble
            for receptor in all_predictions.keys():
                if seq_predictions[receptor]:
                    avg_prob = np.mean(seq_predictions[receptor])
                    all_predictions[receptor].append(avg_prob)
                else:
                    all_predictions[receptor].append(0.0)
        
        # Convert to numpy arrays
        for receptor in all_predictions.keys():
            all_predictions[receptor] = np.array(all_predictions[receptor])
        
        return all_predictions


def load_and_process_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[float]]:
    """Load validation and training data, find exact matches and similarities."""
    print("Loading validation and training data...")
    
    # Load validation data
    val_df = pd.read_csv('validation_sequences.csv')
    print(f"Loaded {len(val_df)} validation sequences")
    
    # Load training data
    train_df = pd.read_csv('all_sequences_aligned.csv')
    print(f"Loaded {len(train_df)} training sequences")
    
    # Find exact matches and similarities
    analyzer = SequenceSimilarityAnalyzer()
    exact_matches = analyzer.find_exact_matches(
        val_df['sequence'].tolist(),
        train_df['sequence'].tolist()
    )
    
    print(f"Found {len(exact_matches)} exact matches to remove")
    
    # Remove exact matches
    val_df_filtered = val_df.drop(exact_matches).reset_index(drop=True)
    print(f"Validation set after removing exact matches: {len(val_df_filtered)} sequences")
    
    # Calculate similarities for remaining sequences
    max_similarities = analyzer.calculate_max_similarities(
        val_df_filtered['sequence'].tolist(),
        train_df['sequence'].tolist()
    )
    
    return val_df_filtered, train_df, exact_matches, max_similarities


def create_similarity_analysis_plot(max_similarities: List[float], 
                                  similarity_threshold: float = 0.8,
                                  save_prefix: str = ""):
    """Create sequence similarity visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Convert to percentages
    similarities_percent = [sim * 100 for sim in max_similarities]
    threshold_percent = similarity_threshold * 100
    
    ax.hist(similarities_percent, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax.axvline(x=threshold_percent, color='red', linestyle='--', linewidth=2, 
               label=f'{threshold_percent:.0f}% similarity threshold')
    ax.axvline(x=90, color='orange', linestyle='--', linewidth=2, label='90% similarity')
    ax.axvline(x=100, color='green', linestyle='--', linewidth=2, label='100% similarity (exact)')
    
    ax.set_title('Sequence Similarity Distribution\n(Max similarity of each validation sequence to training set)')
    ax.set_xlabel('Maximum Sequence Similarity (%)')
    ax.set_ylabel('Number of Validation Sequences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_sim = np.mean(similarities_percent)
    median_sim = np.median(similarities_percent)
    max_sim = np.max(similarities_percent)
    below_threshold = sum(1 for s in similarities_percent if s <= threshold_percent)
    
    stats_text = f'Mean: {mean_sim:.1f}%\nMedian: {median_sim:.1f}%\nMax: {max_sim:.1f}%\n≤{threshold_percent:.0f}%: {below_threshold}/{len(similarities_percent)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    filename = f'{save_prefix}sequence_similarity_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved similarity analysis: {filename}")
    plt.show()
    
    print(f"Similarity statistics:")
    print(f"  Mean similarity: {mean_sim:.1f}%")
    print(f"  Sequences ≤{threshold_percent:.0f}% similar: {below_threshold}/{len(similarities_percent)}")
    print(f"  Sequences >90% similar: {sum(1 for s in similarities_percent if s > 90)}")
    
    return similarities_percent


def filter_sequences_by_similarity(val_df: pd.DataFrame, 
                                 max_similarities: List[float],
                                 similarity_threshold: float = 0.8) -> Tuple[pd.DataFrame, List[int]]:
    """Filter validation sequences by similarity threshold."""
    # Find indices of sequences below threshold
    novel_indices = [i for i, sim in enumerate(max_similarities) if sim <= similarity_threshold]
    
    # Filter dataframe
    novel_df = val_df.iloc[novel_indices].reset_index(drop=True)
    
    print(f"Filtered to {len(novel_df)} sequences with ≤{similarity_threshold*100:.0f}% similarity")
    
    return novel_df, novel_indices


def prepare_validation_targets(val_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Prepare validation targets with missing value handling."""
    targets = {}
    
    # EC50 threshold: < 1000 pM = high affinity (1), >= 1000 pM = low affinity (0)
    threshold = 1000.0
    
    for i, receptor in enumerate(['GCGR', 'GLP1R', 'GIPR']):
        ec50_col = f'EC50_T{i+1}'
        
        if ec50_col in val_df.columns:
            ec50_values = val_df[ec50_col].values
            # Convert to binary: 1 if < 1000 pM, 0 if >= 1000 pM, -1 if missing
            binary_targets = np.where(
                pd.isna(ec50_values), -1,  # Missing values
                np.where(ec50_values < threshold, 1, 0)  # Binary classification
            )
            targets[receptor] = binary_targets
        else:
            # All missing if column doesn't exist
            targets[receptor] = np.full(len(val_df), -1)
    
    return targets


def evaluate_predictions(predictions: Dict[str, np.ndarray], 
                        targets: Dict[str, np.ndarray],
                        analysis_name: str = "") -> Dict[str, Dict]:
    """Evaluate model predictions against validation targets."""
    results = {}
    
    print(f"\nEvaluating predictions for {analysis_name}...")
    
    for receptor in ['GCGR', 'GLP1R', 'GIPR']:
        if receptor in predictions and receptor in targets:
            pred_probs = predictions[receptor]
            true_labels = targets[receptor]
            
            # Filter out missing values (-1)
            valid_mask = true_labels != -1
            
            if valid_mask.sum() == 0:
                print(f"No valid targets for {receptor}")
                continue
            
            valid_probs = pred_probs[valid_mask]
            valid_labels = true_labels[valid_mask]
            
            # Binary predictions (threshold = 0.5)
            binary_preds = (valid_probs > 0.5).astype(int)
            
            # Calculate metrics
            if len(np.unique(valid_labels)) > 1:  # Need both classes for AUC
                auc_roc = roc_auc_score(valid_labels, valid_probs)
                auc_pr = average_precision_score(valid_labels, valid_probs)
                
                # Classification report
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                
                results[receptor] = {
                    'n_samples': len(valid_labels),
                    'n_positive': int(valid_labels.sum()),
                    'n_negative': int((valid_labels == 0).sum()),
                    'accuracy': accuracy_score(valid_labels, binary_preds),
                    'f1_score': f1_score(valid_labels, binary_preds, zero_division=0),
                    'precision': precision_score(valid_labels, binary_preds, zero_division=0),
                    'recall': recall_score(valid_labels, binary_preds, zero_division=0),
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'predictions': valid_probs,
                    'true_labels': valid_labels,
                    'mean_prediction': np.mean(valid_probs),
                    'std_prediction': np.std(valid_probs),
                    'class_distribution': 'both_classes'
                }
            else:
                # Only one class present - use regression-like metrics
                single_class = int(valid_labels[0])
                class_name = 'high_affinity' if single_class == 1 else 'low_affinity'
                
                results[receptor] = {
                    'n_samples': len(valid_labels),
                    'n_positive': int(valid_labels.sum()),
                    'n_negative': int((valid_labels == 0).sum()),
                    'accuracy': np.nan,  # Not meaningful with one class
                    'f1_score': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'auc_roc': np.nan,
                    'auc_pr': np.nan,
                    'predictions': valid_probs,
                    'true_labels': valid_labels,
                    'mean_prediction': np.mean(valid_probs),
                    'std_prediction': np.std(valid_probs),
                    'class_distribution': f'single_class_{class_name}'
                }
                
                print(f"⚠️  {receptor}: Only {class_name} samples present - limited metrics available")
                print(f"    Mean prediction: {np.mean(valid_probs):.3f} ± {np.std(valid_probs):.3f}")
            
            print(f"\n{receptor} Results ({analysis_name}):")
            print(f"  Samples: {results[receptor]['n_samples']} "
                  f"(Pos: {results[receptor]['n_positive']}, "
                  f"Neg: {results[receptor]['n_negative']})")
            if not np.isnan(results[receptor]['accuracy']):
                print(f"  Accuracy: {results[receptor]['accuracy']:.3f}")
                print(f"  F1-Score: {results[receptor]['f1_score']:.3f}")
                print(f"  AUC-ROC: {results[receptor]['auc_roc']:.3f}")
            else:
                print(f"  Mean prediction: {results[receptor]['mean_prediction']:.3f}")
                print(f"  Prediction range: {valid_probs.min():.3f} - {valid_probs.max():.3f}")
                print(f"  Classification metrics: N/A (single class)")
    
    return results


def plot_comparative_performance(novel_results: Dict[str, Dict], 
                               complete_results: Dict[str, Dict]):
    """Create comparative visualization of novel vs complete validation performance."""
    
    receptors = [r for r in ['GCGR', 'GLP1R', 'GIPR'] if r in novel_results and r in complete_results]
    
    # Main figure: AUC-PR and F1 Score
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
    
    main_metrics = ['f1_score', 'auc_pr']
    main_labels = ['F1-Score', 'AUC-PR']
    main_subplot_labels = ['a)', 'b)']
    
    for i, (metric, label) in enumerate(zip(main_metrics, main_labels)):
        ax = axes1[i]
        
        # Add subplot label
        ax.text(0.1, 1.01, main_subplot_labels[i], transform=ax.transAxes, 
                fontsize=14, va='bottom', ha='right')
        
        novel_values = []
        complete_values = []
        valid_receptors = []
        
        for receptor in receptors:
            novel_val = novel_results[receptor].get(metric, np.nan)
            complete_val = complete_results[receptor].get(metric, np.nan)
            
            # Only include if both have valid values
            if not np.isnan(novel_val) and not np.isnan(complete_val):
                novel_values.append(novel_val)
                complete_values.append(complete_val)
                valid_receptors.append(receptor)
        
        if valid_receptors:
            x = np.arange(len(valid_receptors))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, novel_values, width, label='Novel (≤80%)', 
                          alpha=0.7, color='red')
            bars2 = ax.bar(x + width/2, complete_values, width, label='Complete Set', 
                          alpha=0.7, color='blue')
            
            # Add value labels
            for bars, values in [(bars1, novel_values), (bars2, complete_values)]:
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='center', fontsize=8)
            
            ax.set_xlabel('Receptor')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_receptors)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1 if metric != 'auc_pr' else max(max(novel_values), max(complete_values)) * 1.1)
        else:
            ax.text(0.5, 0.5, f'No valid {label} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Comparison')
    
    plt.suptitle('Performance Comparison: Novel Sequences vs Complete Validation Set', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparative_performance_main_new.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Second figure: Remaining metrics
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    other_metrics = ['accuracy', 'precision', 'recall', 'auc_roc']
    other_labels = ['Accuracy', 'Precision', 'Recall', 'AUC-ROC']
    other_subplot_labels = ['a)', 'b)', 'c)', 'd)']
    
    for i, (metric, label) in enumerate(zip(other_metrics, other_labels)):
        row = i // 2
        col = i % 2
        ax = axes2[row, col]
        
        # Add subplot label
        ax.text(0.1, 1.01, other_subplot_labels[i], transform=ax.transAxes, 
                fontsize=14, va='bottom', ha='right')
        
        novel_values = []
        complete_values = []
        valid_receptors = []
        
        for receptor in receptors:
            novel_val = novel_results[receptor].get(metric, np.nan)
            complete_val = complete_results[receptor].get(metric, np.nan)
            
            # Only include if both have valid values
            if not np.isnan(novel_val) and not np.isnan(complete_val):
                novel_values.append(novel_val)
                complete_values.append(complete_val)
                valid_receptors.append(receptor)
        
        if valid_receptors:
            x = np.arange(len(valid_receptors))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, novel_values, width, label='Novel (≤80%)', 
                          alpha=0.7, color='red')
            bars2 = ax.bar(x + width/2, complete_values, width, label='Complete Set', 
                          alpha=0.7, color='blue')
            
            # Add value labels
            for bars, values in [(bars1, novel_values), (bars2, complete_values)]:
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Receptor')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(valid_receptors)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            if metric == 'auc_roc':
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f'No valid {label} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Comparison')
    
    plt.suptitle('Novel Sequences vs Complete Validation Set', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparative_performance_additional_new.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_sample_distribution_comparison(novel_results: Dict[str, Dict], 
                                        complete_results: Dict[str, Dict]):
    """Create sample distribution comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Add subplot labels
    ax1.text(0.1, 1.01, 'a)', transform=ax1.transAxes, 
             fontsize=14, va='bottom', ha='right')
    ax2.text(0.1, 1.01, 'b)', transform=ax2.transAxes, 
             fontsize=14, va='bottom', ha='right')
    
    receptors = [r for r in ['GCGR', 'GLP1R', 'GIPR'] if r in novel_results or r in complete_results]
    
    # Novel sequences distribution
    novel_pos = [novel_results[r]['n_positive'] if r in novel_results else 0 for r in receptors]
    novel_neg = [novel_results[r]['n_negative'] if r in novel_results else 0 for r in receptors]
    novel_total = [novel_pos[i] + novel_neg[i] for i in range(len(receptors))]
    
    # Complete set distribution
    complete_pos = [complete_results[r]['n_positive'] if r in complete_results else 0 for r in receptors]
    complete_neg = [complete_results[r]['n_negative'] if r in complete_results else 0 for r in receptors]
    complete_total = [complete_pos[i] + complete_neg[i] for i in range(len(receptors))]
    
    x = np.arange(len(receptors))
    
    # Novel sequences
    ax1.bar(x, novel_neg, label='Low Affinity (≥1000 pM)', alpha=0.7, color='red')
    ax1.bar(x, novel_pos, bottom=novel_neg, label='High Affinity (<1000 pM)', alpha=0.7, color='blue')
    ax1.set_title('Novel Sequences (≤80% similarity)')
    ax1.set_xlabel('Receptor')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xticks(x)
    ax1.set_xticklabels(receptors)
    ax1.legend()
    
    # Add total labels
    for i, total in enumerate(novel_total):
        if total > 0:
            ax1.text(i, total + 0.1, f'n={total}', ha='center', va='bottom', fontweight='bold')
    
    # Complete set
    ax2.bar(x, complete_neg, label='Low Affinity (≥1000 pM)', alpha=0.7, color='red')
    ax2.bar(x, complete_pos, bottom=complete_neg, label='High Affinity (<1000 pM)', alpha=0.7, color='blue')
    ax2.set_title('Complete Validation Set')
    ax2.set_xlabel('Receptor')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks(x)
    ax2.set_xticklabels(receptors)
    ax2.legend()
    
    # Add total labels
    for i, total in enumerate(complete_total):
        if total > 0:
            ax2.text(i, total + 0.1, f'n={total}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sample_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_summary(novel_results: Dict[str, Dict], 
                        complete_results: Dict[str, Dict],
                        similarity_threshold: float = 0.8):
    """Save comprehensive results summary to CSV."""
    summary_data = []
    
    for receptor in ['GCGR', 'GLP1R', 'GIPR']:
        # Novel sequences results
        if receptor in novel_results:
            novel_data = novel_results[receptor]
            summary_data.append({
                'receptor': receptor,
                'validation_set': f'novel_≤{similarity_threshold*100:.0f}%',
                'n_samples': novel_data['n_samples'],
                'n_positive': novel_data['n_positive'],
                'n_negative': novel_data['n_negative'],
                'accuracy': novel_data.get('accuracy', np.nan),
                'f1_score': novel_data.get('f1_score', np.nan),
                'precision': novel_data.get('precision', np.nan),
                'recall': novel_data.get('recall', np.nan),
                'auc_roc': novel_data.get('auc_roc', np.nan),
                'auc_pr': novel_data.get('auc_pr', np.nan),
                'mean_prediction': novel_data['mean_prediction'],
                'std_prediction': novel_data['std_prediction']
            })
        
        # Complete set results
        if receptor in complete_results:
            complete_data = complete_results[receptor]
            summary_data.append({
                'receptor': receptor,
                'validation_set': 'complete',
                'n_samples': complete_data['n_samples'],
                'n_positive': complete_data['n_positive'],
                'n_negative': complete_data['n_negative'],
                'accuracy': complete_data.get('accuracy', np.nan),
                'f1_score': complete_data.get('f1_score', np.nan),
                'precision': complete_data.get('precision', np.nan),
                'recall': complete_data.get('recall', np.nan),
                'auc_roc': complete_data.get('auc_roc', np.nan),
                'auc_pr': complete_data.get('auc_pr', np.nan),
                'mean_prediction': complete_data['mean_prediction'],
                'std_prediction': complete_data['std_prediction']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('validation_results_summary.csv', index=False)
    print("✅ Results summary saved: validation_results_summary.csv")


def main():
    """Main validation pipeline with similarity-based filtering."""
    print("🧬 GAT Model Validation Pipeline with Similarity Filtering")
    print("="*70)
    
    # Configuration
    similarity_threshold = 0.8  # 80% similarity threshold
    
    # 1. Load and process data
    val_df, train_df, exact_matches, max_similarities = load_and_process_data()
    
    if len(exact_matches) > 0:
        print(f"Removed exact matches at indices: {exact_matches}")
    
    # 2. Create similarity analysis
    similarities_percent = create_similarity_analysis_plot(max_similarities, similarity_threshold)
    
    # 3. Load GAT models
    print("\nLoading GAT ensemble...")
    predictor = GATEnsemblePredictor('kfold_transfer_learning_results_GAT', 'all_sequences_aligned.csv')
    
    # ===== VALIDATION 1: NOVEL SEQUENCES ONLY (≤80% similarity) =====
    print(f"\n{'='*70}")
    print(f"VALIDATION 1: NOVEL SEQUENCES (≤{similarity_threshold*100:.0f}% SIMILARITY)")
    print(f"{'='*70}")
    
    # Filter to novel sequences
    novel_val_df, novel_indices = filter_sequences_by_similarity(val_df, max_similarities, similarity_threshold)
    
    if len(novel_val_df) > 0:
        # Make predictions for novel sequences
        novel_sequences = novel_val_df['sequence'].tolist()
        novel_predictions = predictor.predict_sequences(novel_sequences)
        
        # Prepare targets for novel sequences
        novel_targets = prepare_validation_targets(novel_val_df)
        
        # Evaluate novel sequences
        novel_results = evaluate_predictions(novel_predictions, novel_targets, "Novel Sequences")
    else:
        print("❌ No novel sequences found for validation")
        novel_results = {}
    
    # ===== VALIDATION 2: COMPLETE VALIDATION SET =====
    print(f"\n{'='*70}")
    print("VALIDATION 2: COMPLETE VALIDATION SET")
    print(f"{'='*70}")
    
    # Make predictions for complete set
    complete_sequences = val_df['sequence'].tolist()
    complete_predictions = predictor.predict_sequences(complete_sequences)
    
    # Prepare targets for complete set
    complete_targets = prepare_validation_targets(val_df)
    
    # Evaluate complete set
    complete_results = evaluate_predictions(complete_predictions, complete_targets, "Complete Set")
    
    # ===== COMPARATIVE ANALYSIS =====
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}")
    
    if novel_results and complete_results:
        # Create comparative visualizations
        plot_comparative_performance(novel_results, complete_results)
        create_sample_distribution_comparison(novel_results, complete_results)
        
        # Save comprehensive results
        save_results_summary(novel_results, complete_results, similarity_threshold)
        
        # Print comparative summary
        print(f"\n📊 COMPARATIVE VALIDATION SUMMARY:")
        print("="*50)
        print(f"Novel Sequences (≤{similarity_threshold*100:.0f}% similarity):")
        
        for receptor, metrics in novel_results.items():
            print(f"  {receptor}: n={metrics['n_samples']}, "
                  f"AUC={metrics.get('auc_roc', 'N/A'):.3f}, "
                  f"F1={metrics.get('f1_score', 'N/A'):.3f}")
        
        print(f"\nComplete Validation Set:")
        for receptor, metrics in complete_results.items():
            print(f"  {receptor}: n={metrics['n_samples']}, "
                  f"AUC={metrics.get('auc_roc', 'N/A'):.3f}, "
                  f"F1={metrics.get('f1_score', 'N/A'):.3f}")
        
        # Analysis insights
        print(f"\n🔍 KEY INSIGHTS:")
        for receptor in ['GCGR', 'GLP1R', 'GIPR']:
            if receptor in novel_results and receptor in complete_results:
                novel_auc = novel_results[receptor].get('auc_roc', np.nan)
                complete_auc = complete_results[receptor].get('auc_roc', np.nan)
                
                if not np.isnan(novel_auc) and not np.isnan(complete_auc):
                    diff = complete_auc - novel_auc
                    print(f"• {receptor}: Complete set AUC {diff:+.3f} vs novel sequences")
                    
                    if diff > 0.05:
                        print(f"  → Model performs better on more similar sequences")
                    elif diff < -0.05:
                        print(f"  → Model performs better on novel sequences")
                    else:
                        print(f"  → Similar performance across similarity ranges")
    
    else:
        print("❌ Insufficient data for comparative analysis")
    
    print(f"\n✅ Validation completed!")
    print(f"📁 Results saved: comparative_performance_analysis.png, sample_distribution_comparison.png")


if __name__ == "__main__":
    main()