"""
Transfer Learning GNN for Multi-Receptor EC50 Prediction

This module implements a comprehensive GNN-based approach for predicting 
EC50 values across GCGR, GLP1R, and GIPR receptors using transfer learning
with k-fold cross-validation.

Stages:
1. Initial training on GCGR + GLP1R
2. Transfer learning to GIPR
3. Unified multi-receptor classification

Features:
- K-fold cross-validation for robust evaluation
- Comprehensive metrics and visualizations
- Transfer learning with encoder freezing/unfreezing
- Reproducible results with fixed seeds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef
)
import numpy as np
from torch_geometric.data import Batch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime
from pathlib import Path  # <-- ADD THIS LINE
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.nn import Set2Set
import warnings
warnings.filterwarnings('ignore')

# Import the peptide pipeline
from peptide_gnn_pipeline import load_peptide_dataset


class SharedGNNEncoder(nn.Module):
    """Shared GNN encoder for learning peptide representations."""
    
    def __init__(self, 
                 num_node_features: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if use_attention:
            self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=6, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=6, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final representation layer
        self.pooling = Set2Set(hidden_dim, processing_steps=3)
        self.representation_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Set2Set outputs 2 * hidden_dim
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index, batch, return_attention_weights=False):
        """Forward pass with optional attention weight return."""
        attention_weights = [] if return_attention_weights else None
        
        # Graph convolutions with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Store input for residual (after first layer to handle dimension mismatch)
            if i > 0:
                residual = x
            
            # Apply GAT layer - pass return_attention_weights to forward method
            if self.use_attention and return_attention_weights:
                result = conv(x, edge_index, return_attention_weights=True)
                x = result[0]  # Node features
                edge_index_att, alpha = result[1]  # Attention weights tuple
                attention_weights.append((edge_index_att, alpha))
            else:
                x = conv(x, edge_index)
            
            x = bn(x)
            x = F.relu(x)
            x = nn.LayerNorm(x.size(1)).to(x.device)(x)
            
            # Add residual connection every 2 layers
            if i > 0 and i % 2 == 1:  # Layers 1, 3, 5, etc.
                x = x + residual
                
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling and final representation
        x = self.pooling(x, batch)
        x = self.representation_layer(x)
        
        if return_attention_weights:
            return x, attention_weights
        return x


class TaskSpecificHead(nn.Module):
    """Task-specific prediction head for individual receptors."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.head(x)


class TransferLearningGNN(nn.Module):
    """
    Transfer Learning GNN with shared encoder and task-specific heads.
    """
    
    def __init__(self, 
                 num_node_features: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        
        # Shared encoder
        self.encoder = SharedGNNEncoder(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Task-specific heads
        self.gcgr_head = TaskSpecificHead(hidden_dim, dropout=dropout)
        self.glp1r_head = TaskSpecificHead(hidden_dim, dropout=dropout)
        self.gipr_head = TaskSpecificHead(hidden_dim, dropout=dropout)
        
        # Track which heads are active
        self.active_heads = {'gcgr': True, 'glp1r': True, 'gipr': False}
        
        # Receptor mapping
        self.receptor_to_idx = {'GCGR': 0, 'GLP1R': 1, 'GIPR': 2}
        self.idx_to_receptor = {0: 'GCGR', 1: 'GLP1R', 2: 'GIPR'}
    
    def forward(self, x, edge_index, batch, receptors: Optional[List[str]] = None, return_attention_weights: bool = False):
        """Forward pass with optional attention weights return."""
        # Get shared representation
        if return_attention_weights:
            representation, attention_weights = self.encoder(x, edge_index, batch, return_attention_weights=True)
        else:
            representation = self.encoder(x, edge_index, batch, return_attention_weights=False)
            attention_weights = None
        
        outputs = {}
        
        # Predict for requested receptors
        if receptors is None:
            receptors = [r for r, active in self.active_heads.items() if active]
        
        for receptor in receptors:
            if receptor.lower() == 'gcgr' and self.active_heads['gcgr']:
                outputs['GCGR'] = self.gcgr_head(representation)
            elif receptor.lower() == 'glp1r' and self.active_heads['glp1r']:
                outputs['GLP1R'] = self.glp1r_head(representation)
            elif receptor.lower() == 'gipr' and self.active_heads['gipr']:
                outputs['GIPR'] = self.gipr_head(representation)
        
        if return_attention_weights:
            return outputs, attention_weights
        return outputs
    
    def extract_attention_weights(self, x, edge_index, batch, receptors: Optional[List[str]] = None):
        """Extract attention weights for visualization."""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index, batch, receptors, return_attention_weights=True)
        return attention_weights
    
    def freeze_encoder(self):
        """Freeze the shared encoder for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for transfer learning")
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen for fine-tuning")
    
    def activate_head(self, receptor: str):
        """Activate a specific receptor head."""
        self.active_heads[receptor.lower()] = True
        print(f"{receptor} head activated")
    
    def deactivate_head(self, receptor: str):
        """Deactivate a specific receptor head."""
        self.active_heads[receptor.lower()] = False
        print(f"{receptor} head deactivated")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class TransferLearningPipeline:
    """
    Complete pipeline for transfer learning on multi-receptor data.
    """
    
    def __init__(self, 
                 model_config: Dict,
                 training_config: Dict,
                 data_config: Dict,
                 output_dir: str = "transfer_learning_results"):
        
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_seeds(training_config.get('random_seed', 42))
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = TransferLearningGNN(**model_config).to(self.device)
        
        # Storage for results
        self.stage_results = {}
        self.training_histories = {}
        self.fold_results = []  # Store results for each fold
        self.aggregated_results = {}  # Aggregated results across folds
        
        print(f"Transfer Learning Pipeline initialized")
        print(f"Output directory: {output_dir}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seeds set to {seed}")

    def extract_attention_for_sequences(self, dataset, model, sequence_indices: List[int] = None, max_sequences: int = 10):
        """
        Extract attention weights for specific sequences.
        
        Args:
            dataset: PeptideDataset object
            model: Trained model
            sequence_indices: Indices of sequences to analyze (None for random selection)
            max_sequences: Maximum number of sequences to analyze
            
        Returns:
            Dictionary with attention data for each sequence
        """
        print(f"\n{'='*60}")
        print("EXTRACTING ATTENTION WEIGHTS FOR VISUALIZATION")
        print(f"{'='*60}")
        
        if sequence_indices is None:
            # Randomly select sequences with different lengths for diversity
            available_indices = list(range(len(dataset)))
            sequence_indices = np.random.choice(available_indices, 
                                              size=min(max_sequences, len(available_indices)), 
                                              replace=False).tolist()
        
        attention_data = {}
        model.eval()
        
        for idx in sequence_indices:
            try:
                data = dataset[idx]
                batch = Batch.from_data_list([data]).to(self.device)
                
                # Extract attention weights
                attention_weights = model.extract_attention_weights(
                    batch.x, batch.edge_index, batch.batch, ['GCGR', 'GLP1R', 'GIPR']
                )
                
                # Get predictions for context
                outputs = model(batch.x, batch.edge_index, batch.batch, ['GCGR', 'GLP1R', 'GIPR'])
                predictions = {receptor: torch.sigmoid(logits).item() 
                             for receptor, logits in outputs.items()}
                
                attention_data[idx] = {
                    'sequence_id': data.sequence_id,
                    'original_sequence': data.original_sequence,
                    'tokens': data.tokens,
                    'targets': data.y.numpy(),
                    'predictions': predictions,
                    'attention_weights': attention_weights,
                    'node_features': batch.x.cpu().numpy(),
                    'edge_index': batch.edge_index.cpu().numpy()
                }
                
                print(f"  Processed sequence {idx}: {data.sequence_id[:20]}...")
                
            except Exception as e:
                print(f"  Error processing sequence {idx}: {e}")
                continue
        
        print(f"Successfully extracted attention for {len(attention_data)} sequences")
        return attention_data
    
    def create_attention_graph(self, attention_data: Dict, save_dir):
        """Create attention graph visualizations with gradient coloring based on attention weights."""
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        
        for seq_idx, seq_data in attention_data.items():
            sequence = seq_data['original_sequence']
            tokens = seq_data['tokens']
            attention_weights = seq_data['attention_weights']
            predictions = seq_data['predictions']
            
            if not attention_weights:
                continue
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for i, token in enumerate(tokens):
                G.add_node(i, label=token)
            
            # Extract attention weights from first layer and build edge weights
            edge_weights = {}
            node_attention_sums = {i: 0.0 for i in range(len(tokens))}
            
            if attention_weights:
                edge_index_att, alpha = attention_weights[0]  # First layer
                
                # Convert to numpy and handle multi-head attention
                alpha_np = alpha.cpu().numpy()
                if alpha_np.ndim > 1:  # Multi-head attention
                    alpha_np = alpha_np.mean(axis=1)  # Average across heads
                
                edge_index_np = edge_index_att.cpu().numpy()
                
                # Build edge weights and node attention sums
                attention_threshold = 0.05
                for i, (src, tgt) in enumerate(edge_index_np.T):
                    if i < len(alpha_np):
                        weight = float(alpha_np[i])
                        if weight > attention_threshold:
                            G.add_edge(src, tgt, weight=weight)
                            edge_weights[(src, tgt)] = weight
                            node_attention_sums[src] += weight
                            node_attention_sums[tgt] += weight
            
            # If no attention weights, add basic connectivity
            if not edge_weights:
                for i in range(len(tokens) - 1):
                    G.add_edge(i, i + 1, weight=0.5)
                    edge_weights[(i, i + 1)] = 0.5
                    node_attention_sums[i] += 0.5
                    node_attention_sums[i + 1] += 0.5
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Prepare colormaps for gradient coloring
            if edge_weights:
                edge_weight_values = list(edge_weights.values())
                edge_norm = Normalize(vmin=min(edge_weight_values), vmax=max(edge_weight_values))
                edge_cmap = cm.plasma
                
                node_attention_values = list(node_attention_sums.values())
                node_norm = Normalize(vmin=min(node_attention_values), vmax=max(node_attention_values))
                node_cmap = cm.viridis
            else:
                edge_norm = Normalize(vmin=0, vmax=1)
                edge_cmap = cm.plasma
                node_norm = Normalize(vmin=0, vmax=1)
                node_cmap = cm.viridis
            
            # Draw edges with gradient coloring
            for (u, v) in G.edges():
                weight = edge_weights.get((u, v), edge_weights.get((v, u), 0.1))
                edge_color = edge_cmap(edge_norm(weight))
                edge_width = weight * 5  # Scale width by attention
                
                # Draw individual edge
                edge_pos = [pos[u], pos[v]]
                x_coords, y_coords = zip(*edge_pos)
                ax1.plot(x_coords, y_coords, color=edge_color, linewidth=edge_width, alpha=0.7)
            
            # Draw nodes with gradient coloring
            node_colors = []
            for node in G.nodes():
                attention_sum = node_attention_sums[node]
                color = node_cmap(node_norm(attention_sum))
                node_colors.append(color)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                node_size=1200, alpha=0.9, ax=ax1)
            
            # Add labels
            labels = {i: token for i, token in enumerate(tokens)}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax1)
            
            ax1.set_title(f'Peptide Graph with Attention Weights\n{sequence}', 
                        fontweight='bold', fontsize=12)
            ax1.axis('off')
            
            # Add colorbar for attention weights
            sm_edges = cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
            sm_edges.set_array([])
            cbar_edges = plt.colorbar(sm_edges, ax=ax1, fraction=0.046, pad=0.04)
            cbar_edges.set_label('Attention Weight', rotation=270, labelpad=15)
            
            # Predictions bar chart (right panel)
            receptors = list(predictions.keys())
            probs = list(predictions.values())
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            
            bars = ax2.bar(receptors, probs, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_ylabel('Binding Probability', fontweight='bold')
            ax2.set_title('Receptor Binding Predictions', fontweight='bold')
            ax2.set_ylim(0, 1.0)
            
            # Add threshold line
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Affinity Threshold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            save_path = save_dir / f"attention_graph_seq_{seq_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Graph visualization: {save_path.name}")
        
    def _create_single_attention_heatmap(self, seq_data: Dict, save_dir: Path, seq_idx: int):
        """Create attention heatmap for a single sequence."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sequence_id = seq_data['sequence_id']
        tokens = seq_data['tokens']
        attention_weights = seq_data['attention_weights']
        predictions = seq_data['predictions']
        
        if not attention_weights:
            print(f"No attention weights available for sequence {seq_idx}")
            return
        
        # Create figure with subplots for each attention layer
        n_layers = len(attention_weights)
        fig, axes = plt.subplots(n_layers, 1, figsize=(max(12, len(tokens)), 4 * n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        fig.suptitle(f'Attention Heatmaps - {sequence_id}\n'
                    f'Predictions: GCGR={predictions.get("GCGR", 0):.3f}, '
                    f'GLP1R={predictions.get("GLP1R", 0):.3f}, '
                    f'GIPR={predictions.get("GIPR", 0):.3f}', 
                    fontsize=14, fontweight='bold')
        
        for layer_idx, (edge_index_att, alpha) in enumerate(attention_weights):
            # Convert attention weights to adjacency matrix
            attention_matrix = self._attention_to_matrix(edge_index_att, alpha, len(tokens))
            
            # Create heatmap
            ax = axes[layer_idx]
            sns.heatmap(attention_matrix, 
                       xticklabels=tokens, 
                       yticklabels=tokens,
                       cmap='YlOrRd', 
                       ax=ax,
                       cbar_kws={'label': 'Attention Weight'})
            
            ax.set_title(f'Layer {layer_idx + 1} Attention Weights')
            ax.set_xlabel('Target Token')
            ax.set_ylabel('Source Token')
            
            # Rotate labels for better readability
            ax.tick_params(axis='x')
            ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / f"attention_heatmap_seq_{seq_idx}_{sequence_id[:20]}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path.name}")
    
    def _attention_to_matrix(self, edge_index, alpha, num_nodes):
        """Convert edge-based attention weights to adjacency matrix."""
        attention_matrix = np.zeros((num_nodes, num_nodes))
        
        edge_index_np = edge_index.cpu().numpy()
        alpha_np = alpha.cpu().numpy().mean(axis=1)  # Average across attention heads
        
        for i, (src, dst) in enumerate(edge_index_np.T):
            attention_matrix[src, dst] = alpha_np[i]
        
        return attention_matrix
    
    def _create_attention_summary(self, attention_data: Dict, save_dir: Path):
        """Create summary visualization of attention patterns."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("Creating attention summary visualization...")
        
        # Collect attention statistics
        layer_attention_stats = []
        sequence_lengths = []
        prediction_scores = {'GCGR': [], 'GLP1R': [], 'GIPR': []}
        
        for seq_data in attention_data.values():
            attention_weights = seq_data['attention_weights']
            predictions = seq_data['predictions']
            sequence_lengths.append(len(seq_data['tokens']))
            
            for receptor, score in predictions.items():
                if receptor in prediction_scores:
                    prediction_scores[receptor].append(score)
            
            # Calculate attention statistics per layer
            for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
                if len(layer_attention_stats) <= layer_idx:
                    layer_attention_stats.append([])
                
                alpha_mean = alpha.cpu().numpy().mean()
                alpha_std = alpha.cpu().numpy().std()
                alpha_max = alpha.cpu().numpy().max()
                
                layer_attention_stats[layer_idx].append({
                    'mean': alpha_mean,
                    'std': alpha_std,
                    'max': alpha_max
                })
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Attention statistics by layer
        if layer_attention_stats:
            layer_means = []
            layer_stds = []
            
            for layer_stats in layer_attention_stats:
                if layer_stats:  # Check if layer has data
                    means = [stat['mean'] for stat in layer_stats]
                    layer_means.append(np.mean(means))
                    layer_stds.append(np.std(means))
            
            if layer_means:  # Only plot if we have data
                x_pos = range(len(layer_means))
                axes[0, 0].bar(x_pos, layer_means, yerr=layer_stds, capsize=5, alpha=0.7)
                axes[0, 0].set_title('Average Attention Weights by Layer')
                axes[0, 0].set_xlabel('Layer')
                axes[0, 0].set_ylabel('Mean Attention Weight')
                axes[0, 0].set_xticks(x_pos)
                axes[0, 0].set_xticklabels([f'Layer {i+1}' for i in x_pos])
        
        # Plot 2: Prediction score distribution
        receptor_colors = {'GCGR': 'blue', 'GLP1R': 'orange', 'GIPR': 'green'}
        for receptor, scores in prediction_scores.items():
            if scores:
                axes[0, 1].hist(scores, bins=10, alpha=0.7, label=receptor, 
                            color=receptor_colors[receptor])
        
        axes[0, 1].set_title('Prediction Score Distribution')
        axes[0, 1].set_xlabel('Prediction Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Plot 3: Sequence length distribution
        if sequence_lengths:
            axes[1, 0].hist(sequence_lengths, bins=10, alpha=0.7, color='purple')
            axes[1, 0].set_title('Sequence Length Distribution')
            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Simple attention summary
        axes[1, 1].text(0.1, 0.5, f"Analyzed {len(attention_data)} sequences\n"
                                f"Layers with attention: {len(layer_attention_stats)}\n"
                                f"Avg sequence length: {np.mean(sequence_lengths):.1f}",
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_path = save_dir / "attention_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {summary_path.name}")

    
    def _find_best_fold(self) -> int:
        """Find the fold with best overall performance."""
        if not self.fold_results:
            return 1
        
        best_fold = 1
        best_score = 0
        
        for fold_result in self.fold_results:
            fold_idx = fold_result['fold']
            stage3_results = fold_result.get('stage_results', {}).get('stage3', {})
            
            # Calculate average AUC across receptors
            total_auc = 0
            receptor_count = 0
            
            for receptor in ['GCGR', 'GLP1R', 'GIPR']:
                if receptor in stage3_results and 'auc_roc' in stage3_results[receptor]:
                    total_auc += stage3_results[receptor]['auc_roc']
                    receptor_count += 1
            
            if receptor_count > 0:
                avg_auc = total_auc / receptor_count
                if avg_auc > best_score:
                    best_score = avg_auc
                    best_fold = fold_idx
        
        return best_fold
    
    def _select_diverse_sequences(self, dataset, max_sequences: int) -> List[int]:
        """Select diverse sequences for attention analysis."""
        # Group sequences by length and target patterns
        length_groups = {}
        for i, data in enumerate(dataset):
            length = len(data.tokens)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(i)
        
        # Select representatives from each length group
        selected_indices = []
        sequences_per_group = max(1, max_sequences // len(length_groups))
        
        for length, indices in length_groups.items():
            if len(selected_indices) >= max_sequences:
                break
            
            # Randomly select from this group
            n_select = min(sequences_per_group, len(indices), max_sequences - len(selected_indices))
            selected = np.random.choice(indices, size=n_select, replace=False)
            selected_indices.extend(selected.tolist())
        
        return selected_indices[:max_sequences]
    
    def generate_attention_analysis(self, csv_file: str, max_sequences: int = 15):
        """
        Complete attention analysis pipeline after k-fold training.
        
        Args:
            csv_file: Path to data file
            max_sequences: Maximum number of sequences to analyze
        """
        print(f"\n{'='*80}")
        print("GENERATING ATTENTION ANALYSIS AFTER K-FOLD TRAINING")
        print(f"{'='*80}")
        
        # Load dataset
        receptor_datasets, _ = self.prepare_data(csv_file)
        complete_dataset = receptor_datasets['complete']
        
        if len(complete_dataset) == 0:
            print("No data available for attention analysis")
            return
        
        # Use the best fold model for attention analysis
        best_fold_idx = self._find_best_fold()
        fold_model_path = os.path.join(self.output_dir, f"fold_{best_fold_idx}_final_model.pth")
        
        if not os.path.exists(fold_model_path):
            print(f"Best fold model not found: {fold_model_path}")
            return
        
        # Load best model
        print(f"Loading best fold model: fold_{best_fold_idx}")
        checkpoint = torch.load(fold_model_path, map_location=self.device)
        model = TransferLearningGNN(**self.model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Activate all heads
        model.activate_head('gcgr')
        model.activate_head('glp1r')  
        model.activate_head('gipr')
        
        # Select diverse sequences for analysis
        sequence_indices = self._select_diverse_sequences(complete_dataset, max_sequences)
        
        # Extract attention weights
        attention_data = self.extract_attention_for_sequences(
            complete_dataset, model, sequence_indices, max_sequences
        )
        
        # Create visualizations
        if attention_data:
            attention_dir = Path(self.output_dir) / "attention_heatmaps"
            attention_dir.mkdir(exist_ok=True)
            
            self.create_attention_graph(attention_data, attention_dir)
            self._create_attention_summary(attention_data, attention_dir)
            print(f"✅ Simple attention analysis completed!")
            print(f"📁 Results saved to: {attention_dir}")

        
        print(f"\n✅ Attention analysis completed successfully!")
    
    def prepare_data(self, csv_file: str) -> Tuple[Dict, Dict]:
        """
        Prepare data for different training stages.
        
        Returns:
            Tuple of (receptor_datasets, target_statistics)
        """
        print(f"\n{'='*60}")
        print("PREPARING DATA FOR TRANSFER LEARNING")
        print(f"{'='*60}")
        
        # Load complete dataset
        dataset = load_peptide_dataset(
            csv_file=csv_file,
            high_affinity_cutoff_pM=self.data_config['high_affinity_cutoff_pM']
        )
        
        if len(dataset) == 0:
            raise ValueError("No data loaded from dataset")
        
        # Separate data by receptor availability
        receptor_datasets = {
            'gcgr_glp1r': [],  # Samples with both GCGR and GLP1R data
            'gcgr_only': [],   # Samples with only GCGR data
            'glp1r_only': [],  # Samples with only GLP1R data
            'gipr_only': [],   # Samples with only GIPR data
            'all_three': [],   # Samples with all three receptors
            'complete': dataset  # Complete dataset
        }
        
        target_stats = {'GCGR': {'total': 0, 'high': 0}, 
                       'GLP1R': {'total': 0, 'high': 0}, 
                       'GIPR': {'total': 0, 'high': 0}}
        
        for data in dataset:
            targets = data.y.numpy()
            
            # Check data availability
            has_gcgr = targets[0] != -1
            has_glp1r = targets[1] != -1
            has_gipr = targets[2] != -1
            
            # Categorize samples
            if has_gcgr and has_glp1r and not has_gipr:
                receptor_datasets['gcgr_glp1r'].append(data)
            elif has_gcgr and not has_glp1r and not has_gipr:
                receptor_datasets['gcgr_only'].append(data)
            elif not has_gcgr and has_glp1r and not has_gipr:
                receptor_datasets['glp1r_only'].append(data)
            elif not has_gcgr and not has_glp1r and has_gipr:
                receptor_datasets['gipr_only'].append(data)
            elif has_gcgr and has_glp1r and has_gipr:
                receptor_datasets['all_three'].append(data)
            
            # Update statistics
            for i, receptor in enumerate(['GCGR', 'GLP1R', 'GIPR']):
                if targets[i] != -1:
                    target_stats[receptor]['total'] += 1
                    if targets[i] == 1:
                        target_stats[receptor]['high'] += 1
        
        # Print data distribution
        print(f"\nData Distribution:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  GCGR + GLP1R: {len(receptor_datasets['gcgr_glp1r'])}")
        print(f"  GCGR only: {len(receptor_datasets['gcgr_only'])}")
        print(f"  GLP1R only: {len(receptor_datasets['glp1r_only'])}")
        print(f"  GIPR only: {len(receptor_datasets['gipr_only'])}")
        print(f"  All three: {len(receptor_datasets['all_three'])}")
        
        print(f"\nTarget Statistics:")
        for receptor, stats in target_stats.items():
            if stats['total'] > 0:
                pct = 100 * stats['high'] / stats['total']
                print(f"  {receptor}: {stats['high']}/{stats['total']} high affinity ({pct:.1f}%)")
        
        return receptor_datasets, target_stats
    
    def create_data_loaders(self, 
                           dataset: List, 
                           batch_size: int,
                           test_size: float = 0.2,
                           val_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders with stratified splits."""
        
        if len(dataset) == 0:
            raise ValueError("Empty dataset provided")
        
        # Extract targets for stratification
        targets = []
        for data in dataset:
            # Use first available target for stratification
            target_vals = data.y.numpy()
            first_valid = next((val for val in target_vals if val != -1), 0)
            targets.append(first_valid)
        
        # Handle case where we don't want a separate test set (k-fold CV)
        if test_size == 0.0 or test_size is None:
            # Only create train/val split
            train_indices, val_indices = train_test_split(
                range(len(dataset)), 
                test_size=val_size, 
                stratify=targets,
                random_state=self.training_config['random_seed']
            )
            
            # Create datasets
            train_dataset = [dataset[i] for i in train_indices]
            val_dataset = [dataset[i] for i in val_indices]
            test_dataset = []  # Empty test dataset

            generator = torch.Generator()
            generator.manual_seed(self.training_config['random_seed'])
            
            # Create loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: 0 (k-fold)")
            
            return train_loader, val_loader, test_loader
        
        # Standard train/val/test split
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=test_size, 
            stratify=targets,
            random_state=self.training_config['random_seed']
        )
        
        # Second split: train vs val
        train_val_targets = [targets[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size / (1 - test_size),  # Adjust for remaining data
            stratify=train_val_targets,
            random_state=self.training_config['random_seed']
        )
        
        # Create datasets
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _ensure_2d_targets(self, batch, active_receptors):
        """
        Ensure batch.y is properly shaped as [batch_size, num_targets].
        
        Args:
            batch: Batch from DataLoader
            active_receptors: List of active receptor names
            
        Returns:
            None (modifies batch.y in place)
        """
        if batch.y.dim() == 1:
            # Determine batch size from other batch properties
            batch_size = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 1
            num_targets = 3
            
            # Reshape targets
            expected_size = batch_size * num_targets
            if batch.y.numel() == expected_size:
                batch.y = batch.y.view(batch_size, num_targets)
            else:
                # Fallback: create proper tensor shape based on available data
                print(f"Warning: Target tensor size mismatch. Expected {expected_size}, got {batch.y.numel()}")
                # Pad or truncate as needed
                if batch.y.numel() < expected_size:
                    # Pad with -1 (missing values)
                    padded = torch.full((expected_size,), -1, dtype=batch.y.dtype, device=batch.y.device)
                    padded[:batch.y.numel()] = batch.y
                    batch.y = padded.view(batch_size, num_targets)
                else:
                    # Truncate
                    batch.y = batch.y[:expected_size].view(batch_size, num_targets)
    
    def train_stage(self, 
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   stage_name: str,
                   active_receptors: List[str],
                   epochs: int,
                   learning_rate: float = 0.001) -> Dict:
        """
        Train model for a specific stage.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            stage_name: Name of the training stage
            active_receptors: List of receptors to train on
            epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"TRAINING STAGE: {stage_name.upper()}")
        print(f"{'='*60}")
        print(f"Active receptors: {active_receptors}")
        print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
        
        # Set active heads
        for receptor in ['gcgr', 'glp1r', 'gipr']:
            if receptor.upper() in active_receptors:
                self.model.activate_head(receptor)
            else:
                self.model.deactivate_head(receptor)
        
        # Loss and optimizer
        if self.training_config.get('use_focal_loss', True):
            criterion = FocalLoss(
                alpha=self.training_config.get('focal_alpha', 0.25),
                gamma=self.training_config.get('focal_gamma', 2.0)
            )
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Only optimize parameters that require gradients
        optimizer_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=learning_rate,
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'stage': stage_name,
            'active_receptors': active_receptors
        }
        
        best_val_loss = float('inf')
        patience = self.training_config.get('patience', 20)
        patience_counter = 0
        
        print(f"Starting training...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Ensure targets are properly shaped
                self._ensure_2d_targets(batch, active_receptors)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.batch, active_receptors)
                
                # Compute loss for active receptors
                total_loss = 0
                valid_losses = 0
                
                for receptor in active_receptors:
                    receptor_idx = self.model.receptor_to_idx[receptor]
                    valid_mask = batch.y[:, receptor_idx] != -1
                    
                    if valid_mask.sum() > 0:
                        receptor_outputs = outputs[receptor][valid_mask]
                        receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                        
                        loss = criterion(receptor_outputs, receptor_targets)
                        total_loss += loss
                        valid_losses += 1
                
                if valid_losses > 0:
                    total_loss = total_loss / valid_losses
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(total_loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_predictions = {receptor: [] for receptor in active_receptors}
            val_targets = {receptor: [] for receptor in active_receptors}
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    # Ensure targets are properly shaped
                    self._ensure_2d_targets(batch, active_receptors)
                    
                    outputs = self.model(batch.x, batch.edge_index, batch.batch, active_receptors)
                    
                    # Compute validation loss
                    total_loss = 0
                    valid_losses_val = 0
                    
                    for receptor in active_receptors:
                        receptor_idx = self.model.receptor_to_idx[receptor]
                        valid_mask = batch.y[:, receptor_idx] != -1
                        
                        if valid_mask.sum() > 0:
                            receptor_outputs = outputs[receptor][valid_mask]
                            receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                            
                            loss = criterion(receptor_outputs, receptor_targets)
                            total_loss += loss
                            valid_losses_val += 1
                            
                            # Store predictions for metrics
                            val_predictions[receptor].append(torch.sigmoid(receptor_outputs).cpu())
                            val_targets[receptor].append(receptor_targets.cpu())
                    
                    if valid_losses_val > 0:
                        val_losses.append((total_loss / valid_losses_val).item())
            
            # Aggregate predictions
            for receptor in active_receptors:
                if val_predictions[receptor]:
                    val_predictions[receptor] = torch.cat(val_predictions[receptor], dim=0)
                    val_targets[receptor] = torch.cat(val_targets[receptor], dim=0)
            
            # Compute metrics
            val_metrics = self.compute_metrics(val_predictions, val_targets, active_receptors)
            
            # Record history
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
                for receptor in active_receptors:
                    if receptor in val_metrics and 'auc_roc' in val_metrics[receptor]:
                        auc = val_metrics[receptor]['auc_roc']
                        print(f"    {receptor} AUC: {auc:.3f}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        print(f"Stage '{stage_name}' completed. Best val loss: {best_val_loss:.4f}")
        
        # Save model for this stage
        stage_model_path = os.path.join(self.output_dir, f"{stage_name}_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'stage': stage_name,
            'active_receptors': active_receptors,
            'best_val_loss': best_val_loss
        }, stage_model_path)
        print(f"Model saved to {stage_model_path}")
        
        return history
    
    def compute_metrics(self, 
                       predictions: Dict, 
                       targets: Dict, 
                       receptors: List[str]) -> Dict:
        """Compute comprehensive metrics for active receptors."""
        metrics = {}
        
        for receptor in receptors:
            if receptor in predictions and len(predictions[receptor]) > 0:
                preds = predictions[receptor].numpy().flatten()
                targs = targets[receptor].numpy().flatten()
                
                # Binary predictions
                binary_preds = (preds > 0.5).astype(int)
                
                receptor_metrics = {
                    'n_samples': len(targs),
                    'n_positive': int(targs.sum()),
                    'n_negative': int((targs == 0).sum())
                }
                
                # Only compute metrics if we have both classes
                if len(np.unique(targs)) > 1:
                    try:
                        receptor_metrics.update({
                            'auc_roc': roc_auc_score(targs, preds),
                            'auc_pr': average_precision_score(targs, preds),
                            'f1_score': f1_score(targs, binary_preds),
                            'precision': precision_score(targs, binary_preds, zero_division=0),
                            'recall': recall_score(targs, binary_preds, zero_division=0),
                            'balanced_accuracy': balanced_accuracy_score(targs, binary_preds),
                            'mcc': matthews_corrcoef(targs, binary_preds)
                        })
                    except Exception as e:
                        print(f"Warning: Could not compute metrics for {receptor}: {e}")
                        receptor_metrics.update({
                            'auc_roc': 0.5, 'auc_pr': 0.5, 'f1_score': 0.0,
                            'precision': 0.0, 'recall': 0.0, 
                            'balanced_accuracy': 0.5, 'mcc': 0.0
                        })
                else:
                    receptor_metrics.update({
                        'auc_roc': 0.5, 'auc_pr': 0.5, 'f1_score': 0.0,
                        'precision': 0.0, 'recall': 0.0, 
                        'balanced_accuracy': 0.5, 'mcc': 0.0
                    })
                
                metrics[receptor] = receptor_metrics
        
        return metrics
    
    def evaluate_stage(self, 
                      test_loader: DataLoader, 
                      stage_name: str, 
                      active_receptors: List[str]) -> Dict:
        """Evaluate model on test set for a specific stage."""
        print(f"\nEvaluating {stage_name} on test set...")
        
        self.model.eval()
        test_predictions = {receptor: [] for receptor in active_receptors}
        test_targets = {receptor: [] for receptor in active_receptors}
        
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # Ensure targets are properly shaped
                self._ensure_2d_targets(batch, active_receptors)
                
                outputs = self.model(batch.x, batch.edge_index, batch.batch, active_receptors)
                
                for receptor in active_receptors:
                    receptor_idx = self.model.receptor_to_idx[receptor]
                    valid_mask = batch.y[:, receptor_idx] != -1
                    
                    if valid_mask.sum() > 0:
                        receptor_outputs = outputs[receptor][valid_mask]
                        receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                        
                        test_predictions[receptor].append(torch.sigmoid(receptor_outputs).cpu())
                        test_targets[receptor].append(receptor_targets.cpu())
        
        # Aggregate predictions
        for receptor in active_receptors:
            if test_predictions[receptor]:
                test_predictions[receptor] = torch.cat(test_predictions[receptor], dim=0)
                test_targets[receptor] = torch.cat(test_targets[receptor], dim=0)
        
        # Compute metrics
        test_metrics = self.compute_metrics(test_predictions, test_targets, active_receptors)
        
        # Print results
        print(f"{stage_name} Test Results:")
        for receptor in active_receptors:
            if receptor in test_metrics:
                metrics = test_metrics[receptor]
                print(f"  {receptor}: AUC={metrics.get('auc_roc', 0):.3f}, "
                      f"F1={metrics.get('f1_score', 0):.3f}, "
                      f"Samples={metrics.get('n_samples', 0)}")
        
        return test_metrics
    
    def create_kfold_splits(self, dataset, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified k-fold splits for cross-validation.
        
        Args:
            dataset: Complete dataset
            n_splits: Number of folds
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        print(f"\nCreating {n_splits}-fold stratified splits...")
        
        # Extract stratification information
        stratification_keys = []
        for data in dataset:
            targets = data.y.numpy()
            # Create stratification key based on target availability and values
            key_parts = []
            for i, target in enumerate(targets):
                if target == -1:  # Missing
                    key_parts.append(f"T{i}_missing")
                else:
                    key_parts.append(f"T{i}_{int(target)}")
            stratification_keys.append("_".join(key_parts))
        
        # Use StratifiedKFold or regular KFold based on unique groups
        unique_keys = list(set(stratification_keys))
        print(f"Found {len(unique_keys)} unique stratification patterns")
        
        if len(unique_keys) < n_splits:
            print("Using regular KFold due to limited stratification groups")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.training_config['random_seed'])
            splits = list(kf.split(range(len(dataset))))
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.training_config['random_seed'])
            splits = list(skf.split(range(len(dataset)), stratification_keys))
        
        print(f"Created {len(splits)} folds")
        return splits

    def run_single_fold(self, dataset, train_indices: List[int], test_indices: List[int], fold_idx: int) -> Dict:
        """
        Run complete transfer learning pipeline for a single fold.
        
        Args:
            dataset: Complete dataset
            train_indices: Training sample indices
            test_indices: Test sample indices  
            fold_idx: Current fold index
            
        Returns:
            Dictionary with fold results for all stages
        """
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1} - COMPLETE TRANSFER LEARNING PIPELINE")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        
        # Reset model for this fold
        self.model = TransferLearningGNN(**self.model_config).to(self.device)
        
        # Create fold datasets
        train_dataset = [dataset[i] for i in train_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        fold_results = {
            'fold': fold_idx + 1,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'stage_results': {},
            'training_histories': {}
        }
        
        # ===== STAGE 1: Initial Training (GCGR + GLP1R) =====
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} - STAGE 1: Initial Training (GCGR + GLP1R)")
        print(f"{'='*60}")
        
        # Filter for GCGR/GLP1R data
        stage1_train = []
        stage1_test = []
        
        for data in train_dataset:
            targets = data.y.numpy()
            if targets[0] != -1 or targets[1] != -1:  # Has GCGR or GLP1R data
                stage1_train.append(data)
        
        for data in test_dataset:
            targets = data.y.numpy()
            if targets[0] != -1 or targets[1] != -1:
                stage1_test.append(data)
        
        if len(stage1_train) == 0:
            print("No GCGR/GLP1R training data for this fold")
            fold_results['stage_results']['stage1'] = {}
            fold_results['training_histories']['stage1'] = {}
        else:
            print(f"Stage 1 - Train: {len(stage1_train)}, Test: {len(stage1_test)}")
            
            # Create train/val split for stage 1
            stage1_train_loader, stage1_val_loader, _ = self.create_data_loaders(
                stage1_train, 
                batch_size=self.training_config['batch_size'],
                test_size=0.0  # No separate test set, use validation for monitoring
            )
            stage1_test_loader = DataLoader(stage1_test, batch_size=self.training_config['batch_size'], shuffle=False)
            
            # Train stage 1
            history1 = self.train_stage(
                stage1_train_loader, stage1_val_loader,
                stage_name=f"fold{fold_idx + 1}_stage1_gcgr_glp1r",
                active_receptors=['GCGR', 'GLP1R'],
                epochs=self.training_config['stage1_epochs'],
                learning_rate=self.training_config['learning_rate']
            )
            
            # Evaluate stage 1
            test_metrics1 = self.evaluate_stage(stage1_test_loader, f"Fold {fold_idx + 1} Stage 1", ['GCGR', 'GLP1R'])
            
            fold_results['stage_results']['stage1'] = test_metrics1
            fold_results['training_histories']['stage1'] = history1
        
        # ===== STAGE 2: Transfer Learning (GIPR) =====
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} - STAGE 2: Transfer Learning (GIPR)")
        print(f"{'='*60}")
        
        # Freeze encoder
        self.model.freeze_encoder()
        
        # Filter for GIPR data
        stage2_train = []
        stage2_test = []
        
        for data in train_dataset:
            targets = data.y.numpy()
            if targets[2] != -1:  # Has GIPR data
                stage2_train.append(data)
        
        for data in test_dataset:
            targets = data.y.numpy()
            if targets[2] != -1:
                stage2_test.append(data)
        
        if len(stage2_train) == 0:
            print("No GIPR training data for this fold")
            fold_results['stage_results']['stage2'] = {}
            fold_results['training_histories']['stage2'] = {}
        else:
            print(f"Stage 2 - Train: {len(stage2_train)}, Test: {len(stage2_test)}")
            
            stage2_train_loader, stage2_val_loader, _ = self.create_data_loaders(
                stage2_train,
                batch_size=self.training_config['batch_size'],
                test_size=0.0
            )
            stage2_test_loader = DataLoader(stage2_test, batch_size=self.training_config['batch_size'], shuffle=False)
            
            # Train stage 2
            history2 = self.train_stage(
                stage2_train_loader, stage2_val_loader,
                stage_name=f"fold{fold_idx + 1}_stage2_gipr_transfer",
                active_receptors=['GIPR'],
                epochs=self.training_config['stage2_epochs'],
                learning_rate=self.training_config['transfer_learning_rate']
            )
            
            # Evaluate stage 2
            test_metrics2 = self.evaluate_stage(stage2_test_loader, f"Fold {fold_idx + 1} Stage 2", ['GIPR'])
            
            fold_results['stage_results']['stage2'] = test_metrics2
            fold_results['training_histories']['stage2'] = history2
        
        # ===== STAGE 3: Unified Fine-tuning (All Receptors) =====
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} - STAGE 3: Unified Fine-tuning")
        print(f"{'='*60}")
        
        # Unfreeze encoder
        self.model.unfreeze_encoder()
        
        # Use all data
        train_loader3, val_loader3, _ = self.create_data_loaders(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            test_size=0.0
        )
        test_loader3 = DataLoader(test_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Train stage 3
        history3 = self.train_stage(
            train_loader3, val_loader3,
            stage_name=f"fold{fold_idx + 1}_stage3_unified",
            active_receptors=['GCGR', 'GLP1R', 'GIPR'],
            epochs=self.training_config['stage3_epochs'],
            learning_rate=self.training_config['finetuning_learning_rate']
        )
        
        # Evaluate stage 3
        test_metrics3 = self.evaluate_stage(test_loader3, f"Fold {fold_idx + 1} Stage 3", ['GCGR', 'GLP1R', 'GIPR'])
        
        fold_results['stage_results']['stage3'] = test_metrics3
        fold_results['training_histories']['stage3'] = history3
        
        # Save fold model
        fold_model_path = os.path.join(self.output_dir, f"fold_{fold_idx + 1}_final_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'fold': fold_idx + 1,
            'fold_results': fold_results
        }, fold_model_path)
        
        print(f"\nFold {fold_idx + 1} completed successfully!")
        return fold_results

    def aggregate_fold_results(self) -> Dict:
        """
        Aggregate results across all folds.
        
        Returns:
            Dictionary with aggregated metrics and statistics
        """
        print(f"\n{'='*60}")
        print("AGGREGATING RESULTS ACROSS ALL FOLDS")
        print(f"{'='*60}")
        
        aggregated = {
            'n_folds': len(self.fold_results),
            'stages': {}
        }
        
        receptors = ['GCGR', 'GLP1R', 'GIPR']
        stages = ['stage1', 'stage2', 'stage3']
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall', 'balanced_accuracy', 'mcc']
        
        for stage in stages:
            stage_aggregated = {}
            
            for receptor in receptors:
                receptor_metrics = {}
                
                for metric in metrics:
                    values = []
                    sample_counts = []
                    
                    for fold_result in self.fold_results:
                        stage_results = fold_result.get('stage_results', {}).get(stage, {})
                        if (receptor in stage_results and 
                            metric in stage_results[receptor] and
                            stage_results[receptor].get('n_samples', 0) > 0):
                            values.append(stage_results[receptor][metric])
                            sample_counts.append(stage_results[receptor]['n_samples'])
                    
                    if values:
                        receptor_metrics[f'{metric}_mean'] = np.mean(values)
                        receptor_metrics[f'{metric}_std'] = np.std(values)
                        receptor_metrics[f'{metric}_values'] = values
                        receptor_metrics[f'{metric}_n_folds'] = len(values)
                    else:
                        receptor_metrics[f'{metric}_mean'] = 0.0
                        receptor_metrics[f'{metric}_std'] = 0.0
                        receptor_metrics[f'{metric}_values'] = []
                        receptor_metrics[f'{metric}_n_folds'] = 0
                
                # Sample statistics
                if sample_counts:
                    receptor_metrics['total_samples_mean'] = np.mean(sample_counts)
                    receptor_metrics['total_samples_std'] = np.std(sample_counts)
                    receptor_metrics['total_samples_sum'] = np.sum(sample_counts)
                else:
                    receptor_metrics['total_samples_mean'] = 0
                    receptor_metrics['total_samples_std'] = 0
                    receptor_metrics['total_samples_sum'] = 0
                
                stage_aggregated[receptor] = receptor_metrics
            
            aggregated['stages'][stage] = stage_aggregated
        
        return aggregated

    def run_kfold_pipeline(self, csv_file: str, n_folds: int = 5) -> Dict:
        """
        Run complete k-fold cross-validation pipeline.
        
        Args:
            csv_file: Path to data file
            n_folds: Number of folds for cross-validation
            
        Returns:
            Complete results dictionary
        """
        print(f"\n{'='*80}")
        print(f"STARTING {n_folds}-FOLD CROSS-VALIDATION TRANSFER LEARNING PIPELINE")
        print(f"{'='*80}")
        
        # Prepare data
        receptor_datasets, target_stats = self.prepare_data(csv_file)
        complete_dataset = receptor_datasets['complete']
        
        if len(complete_dataset) == 0:
            raise ValueError("No data available for k-fold cross-validation")
        
        # Create k-fold splits
        fold_splits = self.create_kfold_splits(complete_dataset, n_folds)
        
        # Run each fold
        self.fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(fold_splits):
            fold_result = self.run_single_fold(complete_dataset, train_indices, test_indices, fold_idx)
            self.fold_results.append(fold_result)
        
        # NEW: Extract embeddings from k-fold ensemble
        ensemble_embeddings, sequence_ids, fold_models = self.extract_embeddings_from_fold_ensemble(csv_file)

        self.aggregated_results = self.aggregate_fold_results()
        
        # Compile complete results
        complete_results = {
            'target_statistics': target_stats,
            'n_folds': n_folds,
            'fold_results': self.fold_results,
            'aggregated_results': self.aggregated_results,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            # NEW: Add ensemble info
            'kfold_ensemble_info': {
                'ensemble_size': len(fold_models),
                'embeddings_shape': ensemble_embeddings.shape,
                'num_sequences': len(sequence_ids),
                'embedding_dim': ensemble_embeddings.shape[1],
                'saved_to': os.path.join(self.output_dir, "kfold_ensemble_embeddings.npy")
            }
        }
        
        # Save complete results
        results_path = os.path.join(self.output_dir, "kfold_complete_results.json")
        with open(results_path, 'w') as f:
            json_results = self._convert_for_json(complete_results)
            json.dump(json_results, f, indent=2)
        print(f"Complete k-fold results saved to {results_path}")
        
        # Generate reports
        self.generate_kfold_summary_report(complete_results)
        self.plot_comprehensive_results()
        
        return complete_results
    
    def extract_embeddings_from_fold_ensemble(self, csv_file: str) -> Tuple[np.ndarray, List[str], List]:
        """
        Extract embeddings using the ensemble of k-fold models (already trained).
        
        Args:
            csv_file: Path to data file
            
        Returns:
            Tuple of (ensemble_embeddings, sequence_ids, fold_models)
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING EMBEDDINGS FROM K-FOLD ENSEMBLE")
        print(f"{'='*60}")
        
        # Load complete dataset
        receptor_datasets, _ = self.prepare_data(csv_file)
        complete_dataset = receptor_datasets['complete']
        
        # Create data loader
        data_loader = DataLoader(complete_dataset, 
                                batch_size=self.training_config['batch_size'], 
                                shuffle=False)  # No shuffle for consistent ordering
        
        # Load all fold models
        fold_models = []
        n_folds = len(self.fold_results)
        
        for fold_idx in range(n_folds):
            fold_model_path = os.path.join(self.output_dir, f"fold_{fold_idx + 1}_final_model.pth")
            
            if os.path.exists(fold_model_path):
                print(f"Loading fold {fold_idx + 1} model...")
                checkpoint = torch.load(fold_model_path, map_location=self.device)
                
                # Initialize model
                model = TransferLearningGNN(**self.model_config).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                fold_models.append(model)
            else:
                print(f"Warning: Fold {fold_idx + 1} model not found at {fold_model_path}")
        
        print(f"Loaded {len(fold_models)} fold models for ensemble")
        
        # Extract embeddings from each model
        all_embeddings = []
        sequence_ids = []
        
        for model_idx, model in enumerate(fold_models):
            print(f"Extracting embeddings from fold model {model_idx + 1}...")
            
            model_embeddings = []
            model_sequence_ids = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(self.device)
                    
                    # Get embeddings from shared encoder
                    batch_embeddings = model.encoder(batch.x, batch.edge_index, batch.batch)
                    model_embeddings.append(batch_embeddings.cpu().numpy())
                    
                    # Collect sequence IDs (only once)
                    if model_idx == 0:
                        batch_size = batch.batch.max().item() + 1
                        for i in range(batch_size):
                            model_sequence_ids.append(f"seq_{len(model_sequence_ids)}")
            
            # Concatenate embeddings for this model
            model_embeddings = np.vstack(model_embeddings)
            all_embeddings.append(model_embeddings)
            
            # Store sequence IDs only once
            if model_idx == 0:
                sequence_ids = model_sequence_ids
        
        # Average embeddings across fold models
        ensemble_embeddings = np.mean(all_embeddings, axis=0)
        embedding_std = np.std(all_embeddings, axis=0)
        
        # Save ensemble data
        ensemble_embeddings_path = os.path.join(self.output_dir, "kfold_ensemble_embeddings.npy")
        ensemble_metadata_path = os.path.join(self.output_dir, "kfold_ensemble_metadata.npz")
        
        np.save(ensemble_embeddings_path, ensemble_embeddings)
        np.savez(ensemble_metadata_path,
                sequence_ids=np.array(sequence_ids, dtype=object),
                embedding_dim=ensemble_embeddings.shape[1],
                num_samples=ensemble_embeddings.shape[0],
                num_ensemble_models=len(fold_models),
                embedding_mean=ensemble_embeddings,
                embedding_std=embedding_std)
        
        print(f"\nK-fold ensemble embeddings extracted:")
        print(f"  Ensemble size: {len(fold_models)} models")
        print(f"  Embeddings shape: {ensemble_embeddings.shape}")
        print(f"  Embeddings saved to: {ensemble_embeddings_path}")
        print(f"  Metadata saved to: {ensemble_metadata_path}")
        print(f"  Ready for genetic algorithm!")
        
        return ensemble_embeddings, sequence_ids, fold_models
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_summary_report(self, results: Dict):
        """Generate a comprehensive summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"summary_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRANSFER LEARNING GNN - COMPREHENSIVE SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Target statistics
            f.write("TARGET STATISTICS\n")
            f.write("-"*40 + "\n")
            for receptor, stats in results['target_statistics'].items():
                if stats['total'] > 0:
                    pct = 100 * stats['high'] / stats['total']
                    f.write(f"{receptor}: {stats['high']}/{stats['total']} high affinity ({pct:.1f}%)\n")
            f.write("\n")
            
            # Stage results
            for stage_name, stage_results in results['stage_results'].items():
                f.write(f"{stage_name.upper()} RESULTS\n")
                f.write("-"*40 + "\n")
                
                for receptor, metrics in stage_results.items():
                    if 'auc_roc' in metrics:
                        f.write(f"{receptor}:\n")
                        f.write(f"  Samples: {metrics['n_samples']}\n")
                        f.write(f"  AUC-ROC: {metrics['auc_roc']:.3f}\n")
                        f.write(f"  AUC-PR: {metrics['auc_pr']:.3f}\n")
                        f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
                        f.write(f"  Precision: {metrics['precision']:.3f}\n")
                        f.write(f"  Recall: {metrics['recall']:.3f}\n")
                        f.write(f"  MCC: {metrics['mcc']:.3f}\n\n")
                
                f.write("\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.model_config.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nTRAINING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.training_config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Summary report saved to {report_path}")
    
    def plot_comprehensive_results(self):
        """
        Create comprehensive visualization of k-fold cross-validation results.
        
        Creates two figures:
        Figure 1: Performance comparison and distribution analysis
        Figure 2: Training analysis and convergence metrics
        """
        if not self.fold_results or not self.aggregated_results:
            print("No results available for plotting")
            return
        
        receptors = ['GCGR', 'GLP1R', 'GIPR']
        stages = ['stage1', 'stage2', 'stage3']
        stage_labels = ['Stage 1', 'Stage 2', 'Stage 3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # ========== FIGURE 1: Performance Comparison and Distribution ==========
        fig1 = plt.figure(figsize=(16, 12))
        gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ===== (a) AUC-ROC Comparison Across Receptors and Stages =====
        ax1 = fig1.add_subplot(gs1[0, 0])
        
        auc_data = []
        receptor_labels = []
        stage_positions = []
        
        x_pos = 0
        for stage_idx, stage in enumerate(stages):
            stage_data = self.aggregated_results['stages'].get(stage, {})
            
            for receptor_idx, receptor in enumerate(receptors):
                if receptor in stage_data:
                    auc_mean = stage_data[receptor].get('auc_roc_mean', 0)
                    auc_std = stage_data[receptor].get('auc_roc_std', 0)
                    
                    if auc_mean > 0:  # Only plot if we have data
                        ax1.bar(x_pos, auc_mean, yerr=auc_std, capsize=5, 
                            color=colors[receptor_idx], alpha=0.7, 
                            label=f'{receptor}' if stage_idx == 0 else "")
                        ax1.text(x_pos, auc_mean + auc_std + 0.02, f'{auc_mean:.3f}', 
                                ha='center', va='bottom', fontsize=8)
                        
                        auc_data.append(auc_mean)
                        receptor_labels.append(f'{stage_labels[stage_idx]}')
                        stage_positions.append(x_pos)
                        x_pos += 1
            
            if stage_idx < len(stages) - 1:
                x_pos += 0.5  # Add space between stages
        
        ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold')
        ax1.set_title('AUC-ROC Comparison Across Receptors and Stages', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC-ROC Score')
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(stage_positions)
        ax1.set_xticklabels(receptor_labels, ha='center')
        ax1.legend(['GCGR', 'GLP1R', 'GIPR'], loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # ===== (b) F1-Score Comparison =====
        ax2 = fig1.add_subplot(gs1[0, 1])
        
        f1_data = []
        f1_positions = []
        
        x_pos = 0
        for stage_idx, stage in enumerate(stages):
            stage_data = self.aggregated_results['stages'].get(stage, {})
            
            for receptor_idx, receptor in enumerate(receptors):
                if receptor in stage_data:
                    f1_mean = stage_data[receptor].get('f1_score_mean', 0)
                    f1_std = stage_data[receptor].get('f1_score_std', 0)
                    
                    if f1_mean > 0:
                        ax2.bar(x_pos, f1_mean, yerr=f1_std, capsize=5,
                            color=colors[receptor_idx], alpha=0.7)
                        ax2.text(x_pos, f1_mean + f1_std + 0.02, f'{f1_mean:.3f}',
                                ha='center', va='bottom', fontsize=8)
                        
                        f1_data.append(f1_mean)
                        f1_positions.append(x_pos)
                        x_pos += 1
            
            if stage_idx < len(stages) - 1:
                x_pos += 0.5
        
        ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold')
        ax2.set_title('F1-Score Comparison Across Receptors and Stages', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1.1)
        ax2.set_xticks(f1_positions)
        ax2.legend(['GCGR', 'GLP1R', 'GIPR'], loc='lower right')
        ax2.set_xticklabels(receptor_labels, ha='center')
        ax2.grid(True, alpha=0.3)
        
        # ===== (c) AUC-ROC Distribution Across Folds (Box Plots) =====
        ax3 = fig1.add_subplot(gs1[1, 0])
        
        # Focus on final stage (stage3) for distribution analysis
        stage3_data = self.aggregated_results['stages'].get('stage3', {})
        
        box_data = []
        box_labels = []
        
        for receptor in receptors:
            if receptor in stage3_data:
                auc_values = stage3_data[receptor].get('auc_roc_values', [])
                if auc_values:
                    box_data.append(auc_values)
                    box_labels.append(f'{receptor}\n(n={len(auc_values)} folds)')
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.text(-0.1, 1.05, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold')
            ax3.set_title('AUC-ROC Distribution Across Folds (Final Stage)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('AUC-ROC Score')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # ===== (d) Stage Progression Analysis =====
        ax4 = fig1.add_subplot(gs1[1, 1])
        
        # Show how performance improves across stages for each receptor
        stage_names = ['Initial', 'Transfer', 'Unified']
        
        for receptor_idx, receptor in enumerate(receptors):
            stage_aucs = []
            stage_errors = []
            
            for stage in stages:
                stage_data = self.aggregated_results['stages'].get(stage, {})
                if receptor in stage_data:
                    auc_mean = stage_data[receptor].get('auc_roc_mean', 0)
                    auc_std = stage_data[receptor].get('auc_roc_std', 0)
                    stage_aucs.append(auc_mean if auc_mean > 0 else None)
                    stage_errors.append(auc_std if auc_mean > 0 else 0)
                else:
                    stage_aucs.append(None)
                    stage_errors.append(0)
            
            # Plot line for this receptor
            valid_indices = [i for i, auc in enumerate(stage_aucs) if auc is not None]
            valid_aucs = [stage_aucs[i] for i in valid_indices]
            valid_errors = [stage_errors[i] for i in valid_indices]
            valid_stages = [valid_indices[i] for i in range(len(valid_indices))]
            
            if valid_aucs:
                ax4.errorbar(valid_stages, valid_aucs, yerr=valid_errors, 
                            marker='o', linewidth=2, markersize=8, capsize=5,
                            color=colors[receptor_idx], label=receptor)
        
        ax4.text(-0.1, 1.05, '(d)', transform=ax4.transAxes, fontsize=16, fontweight='bold')
        ax4.set_title('Performance Progression Across Stages', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Stage')
        ax4.set_ylabel('AUC-ROC Score')
        ax4.set_xticks(range(len(stage_names)))
        ax4.set_xticklabels(stage_names)
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Adjust layout and save Figure 1
        fig1.suptitle(f'Performance Analysis - K-Fold Cross-Validation ({self.aggregated_results["n_folds"]} folds)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plot_path1 = os.path.join(self.output_dir, "kfold_performance_analysis.png")
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        print(f"Performance analysis plot saved to {plot_path1}")
        
        # ========== FIGURE 2: Training Analysis and Convergence ==========
        fig2 = plt.figure(figsize=(16, 12))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ===== (a) Comprehensive Metrics Heatmap =====
        ax5 = fig2.add_subplot(gs2[0, 0])
        
        # Create heatmap for final stage metrics
        metrics_names = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision', 'Recall', 'Bal.Acc', 'MCC']
        metrics_keys = ['auc_roc_mean', 'auc_pr_mean', 'f1_score_mean', 'precision_mean', 
                    'recall_mean', 'balanced_accuracy_mean', 'mcc_mean']
        
        heatmap_data = []
        heatmap_receptors = []
        
        stage3_data = self.aggregated_results['stages'].get('stage3', {})
        for receptor in receptors:
            if receptor in stage3_data:
                row = []
                for metric_key in metrics_keys:
                    value = stage3_data[receptor].get(metric_key, 0)
                    row.append(value)
                heatmap_data.append(row)
                heatmap_receptors.append(receptor)
        
        if heatmap_data:
            heatmap_data = np.array(heatmap_data)
            im = ax5.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            ax5.text(-0.1, 1.05, '(a)', transform=ax5.transAxes, fontsize=16, fontweight='bold')
            ax5.set_title('Comprehensive Metrics Heatmap (Final Stage)', fontsize=12, fontweight='bold')
            ax5.set_xticks(range(len(metrics_names)))
            ax5.set_xticklabels(metrics_names, rotation=45, ha='right')
            ax5.set_yticks(range(len(heatmap_receptors)))
            ax5.set_yticklabels(heatmap_receptors)
            
            # Add text annotations
            for i in range(len(heatmap_receptors)):
                for j in range(len(metrics_names)):
                    text = ax5.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=10)
            
            plt.colorbar(im, ax=ax5, label='Score')
        
        # ===== (b) Training Curves (Average Across Folds) =====
        ax6 = fig2.add_subplot(gs2[0, 1])
        
        # Aggregate training curves for stage 3 (unified training)
        all_train_losses = []
        all_val_losses = []
        
        for fold_result in self.fold_results:
            stage3_history = fold_result.get('training_histories', {}).get('stage3', {})
            if 'train_loss' in stage3_history and 'val_loss' in stage3_history:
                all_train_losses.append(stage3_history['train_loss'])
                all_val_losses.append(stage3_history['val_loss'])
        
        if all_train_losses and all_val_losses:
            # Find minimum length across all folds
            min_epochs = min(len(losses) for losses in all_train_losses)
            
            if min_epochs > 0:
                # Truncate all histories to minimum length
                train_losses_truncated = [losses[:min_epochs] for losses in all_train_losses]
                val_losses_truncated = [losses[:min_epochs] for losses in all_val_losses]
                
                # Calculate mean and std
                mean_train_loss = np.mean(train_losses_truncated, axis=0)
                std_train_loss = np.std(train_losses_truncated, axis=0)
                mean_val_loss = np.mean(val_losses_truncated, axis=0)
                std_val_loss = np.std(val_losses_truncated, axis=0)
                
                epochs = range(1, len(mean_train_loss) + 1)
                
                # Plot mean curves
                ax6.plot(epochs, mean_train_loss, 'b-', linewidth=2, label='Training Loss')
                ax6.plot(epochs, mean_val_loss, 'r-', linewidth=2, label='Validation Loss')
                
                # Add confidence intervals
                ax6.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                                alpha=0.2, color='blue')
                ax6.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                                alpha=0.2, color='red')
                
                ax6.text(-0.1, 1.05, '(b)', transform=ax6.transAxes, fontsize=16, fontweight='bold')
                ax6.set_title(f'Training Curves (Unified Stage, {len(all_train_losses)} folds)', 
                            fontsize=12, fontweight='bold')
                ax6.set_xlabel('Epoch')
                ax6.set_ylabel('Loss')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        # ===== (c) Sample Distribution Analysis =====
        ax7 = fig2.add_subplot(gs2[1, 0])
        
        # Show sample counts across folds and stages
        sample_data = {'GCGR': [], 'GLP1R': [], 'GIPR': []}
        
        for receptor in receptors:
            stage3_data = self.aggregated_results['stages'].get('stage3', {})
            if receptor in stage3_data:
                total_samples = stage3_data[receptor].get('total_samples_sum', 0)
                sample_data[receptor] = [total_samples]
        
        x_positions = np.arange(len(receptors))
        bar_values = [sample_data[receptor][0] if sample_data[receptor] else 0 for receptor in receptors]
        
        bars = ax7.bar(x_positions, bar_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, bar_values):
            if value > 0:
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar_values)*0.01,
                        f'{int(value)}', ha='center', va='bottom', fontsize=10)
        
        ax7.text(-0.1, 1.05, '(c)', transform=ax7.transAxes, fontsize=16, fontweight='bold')
        ax7.set_title('Total Sample Distribution (Final Stage)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Receptor')
        ax7.set_ylabel('Total Samples Across Folds')
        ax7.set_xticks(x_positions)
        ax7.set_xticklabels(receptors)
        ax7.grid(True, alpha=0.3)
        
        # ===== (d) Model Convergence Analysis =====
        ax8 = fig2.add_subplot(gs2[1, 1])
        
        # Show final loss values across folds
        final_losses = {'Training': [], 'Validation': []}
        
        for fold_result in self.fold_results:
            stage3_history = fold_result.get('training_histories', {}).get('stage3', {})
            if 'train_loss' in stage3_history and 'val_loss' in stage3_history:
                if stage3_history['train_loss']:
                    final_losses['Training'].append(stage3_history['train_loss'][-1])
                if stage3_history['val_loss']:
                    final_losses['Validation'].append(stage3_history['val_loss'][-1])
        
        if final_losses['Training'] and final_losses['Validation']:
            x = ['Training', 'Validation']
            means = [np.mean(final_losses['Training']), np.mean(final_losses['Validation'])]
            stds = [np.std(final_losses['Training']), np.std(final_losses['Validation'])]
            
            bars = ax8.bar(x, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(means)*0.02,
                        f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax8.text(-0.1, 1.05, '(d)', transform=ax8.transAxes, fontsize=16, fontweight='bold')
            ax8.set_title('Final Loss Values Across Folds', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Loss Value')
            ax8.grid(True, alpha=0.3)
        
        # Adjust layout and save Figure 2
        fig2.suptitle(f'Training Analysis - K-Fold Cross-Validation ({self.aggregated_results["n_folds"]} folds)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plot_path2 = os.path.join(self.output_dir, "kfold_training_analysis.png")
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Training analysis plot saved to {plot_path2}")
        
        plt.show()

    def generate_kfold_summary_report(self, results: Dict):
        """Generate comprehensive summary report for k-fold results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"kfold_summary_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("K-FOLD CROSS-VALIDATION TRANSFER LEARNING GNN - COMPREHENSIVE REPORT\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of folds: {results['n_folds']}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*50 + "\n")
            
            aggregated = results['aggregated_results']
            stage3_data = aggregated['stages'].get('stage3', {})
            
            for receptor in ['GCGR', 'GLP1R', 'GIPR']:
                if receptor in stage3_data:
                    metrics = stage3_data[receptor]
                    auc_mean = metrics.get('auc_roc_mean', 0)
                    auc_std = metrics.get('auc_roc_std', 0)
                    f1_mean = metrics.get('f1_score_mean', 0)
                    f1_std = metrics.get('f1_score_std', 0)
                    n_folds = metrics.get('auc_roc_n_folds', 0)
                    
                    f.write(f"{receptor}: AUC-ROC = {auc_mean:.3f} ± {auc_std:.3f} "
                           f"(F1 = {f1_mean:.3f} ± {f1_std:.3f}, {n_folds} folds)\n")
            
            f.write("\n")
            
            # Detailed Results by Stage
            for stage_name, stage_data in aggregated['stages'].items():
                f.write(f"{stage_name.upper()} RESULTS\n")
                f.write("-"*50 + "\n")
                
                for receptor, metrics in stage_data.items():
                    if metrics.get('auc_roc_n_folds', 0) > 0:
                        f.write(f"\n{receptor}:\n")
                        f.write(f"  Folds with data: {metrics['auc_roc_n_folds']}\n")
                        f.write(f"  AUC-ROC: {metrics['auc_roc_mean']:.3f} ± {metrics['auc_roc_std']:.3f}\n")
                        f.write(f"  AUC-PR: {metrics['auc_pr_mean']:.3f} ± {metrics['auc_pr_std']:.3f}\n")
                        f.write(f"  F1-Score: {metrics['f1_score_mean']:.3f} ± {metrics['f1_score_std']:.3f}\n")
                        f.write(f"  Precision: {metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}\n")
                        f.write(f"  Recall: {metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}\n")
                        f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy_mean']:.3f} ± {metrics['balanced_accuracy_std']:.3f}\n")
                        f.write(f"  MCC: {metrics['mcc_mean']:.3f} ± {metrics['mcc_std']:.3f}\n")
                        f.write(f"  Total samples: {int(metrics['total_samples_sum'])}\n")
                
                f.write("\n")
            
            # Configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*50 + "\n")
            for key, value in results['model_config'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nTRAINING CONFIGURATION\n")
            f.write("-"*50 + "\n")
            for key, value in results['training_config'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nDATASET STATISTICS\n")
            f.write("-"*50 + "\n")
            for receptor, stats in results['target_statistics'].items():
                if stats['total'] > 0:
                    pct = 100 * stats['high'] / stats['total']
                    f.write(f"{receptor}: {stats['high']}/{stats['total']} high affinity ({pct:.1f}%)\n")
        
        print(f"K-fold summary report saved to {report_path}")


    def _extract_embeddings_from_model(self, model, data_loader):
        """Extract embeddings from a single model."""
        model.eval()
        embeddings = []
        sequence_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Get embeddings from shared encoder
                batch_embeddings = model.encoder(batch.x, batch.edge_index, batch.batch)
                embeddings.append(batch_embeddings.cpu().numpy())
                
                # Collect sequence IDs
                batch_size = batch.batch.max().item() + 1
                for i in range(batch_size):
                    sequence_ids.append(f"seq_{len(sequence_ids)}")
        
        # Concatenate all embeddings
        final_embeddings = np.vstack(embeddings)
        return final_embeddings, sequence_ids

    def run_complete_pipeline(self, csv_file: str) -> Dict:
        """Plot training progress for all stages."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training curves
        for i, (stage_name, history) in enumerate(self.training_histories.items()):
            if i < 3:  # Max 3 stages
                row = i // 2
                col = i % 2
                
                if 'train_loss' in history and 'val_loss' in history:
                    epochs = range(1, len(history['train_loss']) + 1)
                    axes[row, col].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
                    axes[row, col].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
                    axes[row, col].set_title(f'{stage_name} Training Progress')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel('Loss')
                    axes[row, col].legend()
                    axes[row, col].grid(True)
        
        # Plot final metrics comparison
        if len(self.training_histories) <= 3:
            axes[1, 1].axis('off')
            
            # Create metrics summary table
            summary_text = "FINAL METRICS SUMMARY\n\n"
            
            if 'stage3' in self.stage_results:
                for receptor, metrics in self.stage_results['stage3'].items():
                    if 'auc_roc' in metrics:
                        summary_text += f"{receptor}:\n"
                        summary_text += f"  AUC-ROC: {metrics['auc_roc']:.3f}\n"
                        summary_text += f"  F1-Score: {metrics['f1_score']:.3f}\n"
                        summary_text += f"  Samples: {metrics['n_samples']}\n\n"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to {plot_path}")
        
        plt.show()




def main():
    """Main function to run the complete k-fold cross-validation transfer learning pipeline."""
    
    print("🧬 K-FOLD CROSS-VALIDATION TRANSFER LEARNING GNN")
    print("="*80)
    
    # Configuration
    model_config = {
        'num_node_features': 7,
        'hidden_dim': 96,
        'num_layers': 4,
        'dropout': 0.2,
        'use_attention': True
    }
    
    training_config = {
    'random_seed': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'transfer_learning_rate': 0.001,
    'finetuning_learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'patience': 15,
    'stage1_epochs': 80,
    'stage2_epochs': 100,
    'stage3_epochs': 60,
    }
    
    data_config = {
        'high_affinity_cutoff_pM': 1000.0
    }
    
    # K-fold configuration
    n_folds = 5
    
    # Print configuration
    print("Configuration:")
    print(f"  Model: GAT with {model_config['hidden_dim']} hidden dims, {model_config['num_layers']} layers")
    print(f"  K-fold CV: {n_folds} folds")
    print(f"  Stages: {training_config['stage1_epochs']} + {training_config['stage2_epochs']} + {training_config['stage3_epochs']} epochs")
    print(f"  Transfer LR: {training_config['transfer_learning_rate']}")
    print(f"  Random seed: {training_config['random_seed']}")
    print(f"  Batch size: {training_config['batch_size']}")
    
    # Initialize pipeline
    pipeline = TransferLearningPipeline(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir="kfold_transfer_learning_results_GAT"
    )
    
    # Run k-fold cross-validation pipeline
    try:
        print(f"\n🚀 Starting {n_folds}-fold cross-validation...")
        results = pipeline.run_kfold_pipeline('all_sequences_aligned.csv', n_folds=n_folds)
        
        print(f"\n✅ K-fold cross-validation pipeline completed successfully!")
        print(f"Results saved to: {pipeline.output_dir}")


         # NEW: Generate attention analysis
        print(f"\n🔍 Starting attention analysis...")
        pipeline.generate_attention_analysis('all_sequences_aligned.csv', max_sequences=20)
        
        # Print final summary
        print(f"\n📊 FINAL K-FOLD RESULTS SUMMARY:")
        print(f"{'='*80}")
        
        aggregated = results['aggregated_results']
        stage3_data = aggregated['stages'].get('stage3', {})
        
        for receptor in ['GCGR', 'GLP1R', 'GIPR']:
            if receptor in stage3_data:
                metrics = stage3_data[receptor]
                auc_mean = metrics.get('auc_roc_mean', 0)
                auc_std = metrics.get('auc_roc_std', 0)
                f1_mean = metrics.get('f1_score_mean', 0)
                f1_std = metrics.get('f1_score_std', 0)
                n_folds_data = metrics.get('auc_roc_n_folds', 0)
                total_samples = int(metrics.get('total_samples_sum', 0))
                
                print(f"{receptor:>6}: AUC = {auc_mean:.3f} ± {auc_std:.3f}, "
                      f"F1 = {f1_mean:.3f} ± {f1_std:.3f} "
                      f"({n_folds_data}/{n_folds} folds, {total_samples} samples)")
        
        print(f"\n📈 Comprehensive visualization generated!")
        print(f"📋 Detailed reports saved!")
        
        # Additional insights
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"{'='*60}")
        
        # Best performing receptor
        best_receptor = None
        best_auc = 0
        for receptor in ['GCGR', 'GLP1R', 'GIPR']:
            if receptor in stage3_data:
                auc = stage3_data[receptor].get('auc_roc_mean', 0)
                if auc > best_auc:
                    best_auc = auc
                    best_receptor = receptor
        
        if best_receptor:
            print(f"• Best performing receptor: {best_receptor} (AUC = {best_auc:.3f})")
        
        # Transfer learning effectiveness
        stage1_data = aggregated['stages'].get('stage1', {})
        stage3_data = aggregated['stages'].get('stage3', {})
        
        for receptor in ['GCGR', 'GLP1R']:
            if receptor in stage1_data and receptor in stage3_data:
                stage1_auc = stage1_data[receptor].get('auc_roc_mean', 0)
                stage3_auc = stage3_data[receptor].get('auc_roc_mean', 0)
                if stage1_auc > 0 and stage3_auc > 0:
                    improvement = ((stage3_auc - stage1_auc) / stage1_auc) * 100
                    print(f"• {receptor} improvement: {improvement:+.1f}% (Stage 1→3)")
        
        # GIPR transfer learning success
        if 'GIPR' in stage3_data:
            gipr_auc = stage3_data['GIPR'].get('auc_roc_mean', 0)
            gipr_folds = stage3_data['GIPR'].get('auc_roc_n_folds', 0)
            if gipr_auc > 0.5:
                print(f"• GIPR transfer learning: Successful (AUC = {gipr_auc:.3f} > 0.5)")
            else:
                print(f"• GIPR transfer learning: Limited success (AUC = {gipr_auc:.3f})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during k-fold pipeline execution: {e}")
        raise


if __name__ == "__main__":
    results = main()