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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.nn import Set2Set
import warnings
warnings.filterwarnings('ignore')

# Import the peptide pipeline
from gat_dual_regression_pipeline import load_peptide_dataset


class SharedGNNEncoder(nn.Module):
    """Shared GNN encoder for learning peptide representations."""
    
    def __init__(self, 
                 num_node_features: int = 9,
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
        
         # First layer
        if use_attention:
            self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=6, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
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
    
    def forward(self, x, edge_index, batch):
        """Forward pass with residual connections."""

        # Graph convolutions with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Store input for residual (before transformation)
            if i > 0:  # Skip first layer (dimension mismatch)
                residual = x
            
            # Apply GAT layer
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = nn.LayerNorm(x.size(1)).to(x.device)(x)
            
            # Add residual connection every 2 layers
            if i > 0 and i % 2 == 1:  # Layers 1, 3, 5, etc.
                x = x + residual
                
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling and final representation (unchanged)
        x = self.pooling(x, batch)
        x = self.representation_layer(x)
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
            nn.BatchNorm1d(hidden_dim // 2), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
            # Remove sigmoid activation for regression
        )
    
    def forward(self, x):
        return self.head(x)


class TransferLearningGNN(nn.Module):
    """
    Transfer Learning GNN with shared encoder and task-specific heads.
    """
    
    def __init__(self, 
                 num_node_features: int = 9,
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

        self.task_log_vars = nn.ParameterDict({
            'GCGR': nn.Parameter(torch.zeros(1)),
            'GLP1R': nn.Parameter(torch.zeros(1)),
        })
        

        # Task-specific heads
        self.gcgr_head = TaskSpecificHead(hidden_dim, dropout=dropout)
        self.glp1r_head = TaskSpecificHead(hidden_dim, dropout=dropout)
        
        # Track which heads are active - only using T1 and T2
        self.active_heads = {'gcgr': True, 'glp1r': True}
        
        # Receptor mapping
        self.receptor_to_idx = {'GCGR': 0, 'GLP1R': 1}
        self.idx_to_receptor = {0: 'GCGR', 1: 'GLP1R'}
    
    def forward(self, x, edge_index, batch, receptors: Optional[List[str]] = None):
        """
        Forward pass with optional receptor selection.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
            receptors: List of receptors to predict for
            
        Returns:
            Dictionary of predictions for requested receptors
        """
        # Get shared representation
        representation = self.encoder(x, edge_index, batch)
        
        outputs = {}
        
        # Predict for requested receptors
        if receptors is None:
            receptors = [r for r, active in self.active_heads.items() if active]
        
        for receptor in receptors:
            if receptor.lower() == 'gcgr' and self.active_heads['gcgr']:
                outputs['GCGR'] = self.gcgr_head(representation)
            elif receptor.lower() == 'glp1r' and self.active_heads['glp1r']:
                outputs['GLP1R'] = self.glp1r_head(representation)
        
        return outputs
    
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

    def print_model_comparison_table(self, results: Dict):
            """
            Print model comparison table following Puszkarska et al. Table 1 format.
            """
            print(f"\n{'='*80}")
            print("MODEL PERFORMANCE COMPARISON")
            print("Following Puszkarska et al. Nature Chemistry 2024 - Table 1 Format")
            print("="*80)
            
            aggregated = results['aggregated_results']
            regression_data = aggregated['stages'].get('regression', {})
            
            # Header
            print(f"{'Models':<20} {'Cross-validation on training data':<40}")
            print(f"{'':20} {'R.m.s.e. GCGR':<20} {'R.m.s.e. GLP1R':<20}")
            print("-" * 80)
            
            # Our multi-task GNN results
            gcgr_metrics = regression_data.get('GCGR', {})
            glp1r_metrics = regression_data.get('GLP1R', {})
            
            gcgr_rmse = f"{gcgr_metrics.get('r.m.s.e._mean', 0):.2f} ± {gcgr_metrics.get('r.m.s.e._std', 0):.2f}"
            glp1r_rmse = f"{glp1r_metrics.get('r.m.s.e._mean', 0):.2f} ± {glp1r_metrics.get('r.m.s.e._std', 0):.2f}"
            
            print(f"{'Multi-task GNN':<20} {gcgr_rmse:<20} {glp1r_rmse:<20}")
            
            # Note about comparison with original paper
            print("-" * 80)
            print("Note: Original Puszkarska et al. results for comparison:")
            print("NN multi-task ensemble:   0.59 ± 0.05        0.68 ± 0.04")
            print("Random forest:           0.62 ± 0.04        0.77 ± 0.06")
            print("SVR:                     0.60 ± 0.04        0.81 ± 0.07")
            print("Ridge:                   0.63 ± 0.06        0.75 ± 0.10")
            print("\n(Original paper used 125 training sequences with different data preprocessing)")
    
    def prepare_data(self, csv_file: str) -> Tuple[Dict, Dict]:
        """
        Prepare data for GCGR/GLP1R regression training.
        Following Puszkarska et al. Nature Chemistry 2024 methodology.
        
        Returns:
            Tuple of (receptor_datasets, target_statistics)
        """
        print(f"\n{'='*60}")
        print("PREPARING DATA FOR REGRESSION TRAINING")
        print(f"{'='*60}")
        print("Target: Continuous EC50 prediction for GCGR and GLP1R")
        
        # Load complete dataset - note: high_affinity_cutoff not used for regression
        dataset = load_peptide_dataset(
            csv_file=csv_file,
            high_affinity_cutoff_pM=self.data_config.get('high_affinity_cutoff_pM', 1000.0)
        )

            # DEBUG: Check the dataset before training
        print("\n=== DEBUGGING DATASET ===")
        dataset = load_peptide_dataset('training_data.csv')
        
        if len(dataset) > 0:
            print(f"Dataset loaded: {len(dataset)} samples")
            
            # Check first few samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}:")
                print(f"  Sequence ID: {sample.sequence_id}")
                print(f"  Target shape: {sample.y.shape}")
                print(f"  Target values: {sample.y}")
                print(f"  Target dtype: {sample.y.dtype}")
                
            # Check if all samples have 2 targets
            target_shapes = [sample.y.shape for sample in dataset]
            unique_shapes = list(set(target_shapes))
            print(f"Unique target shapes in dataset: {unique_shapes}")
            
            # Check for any -999.0 values
            all_targets = torch.stack([sample.y for sample in dataset])
            missing_count = (all_targets == -999.0).sum().item()
            print(f"Missing target values (-999.0): {missing_count}")
            print(f"Total target values: {all_targets.numel()}")
            
            # Check target value ranges
            valid_targets = all_targets[all_targets != -999.0]
            if len(valid_targets) > 0:
                print(f"Valid target range: {valid_targets.min().item():.2f} to {valid_targets.max().item():.2f}")
                print(f"Valid target mean: {valid_targets.mean().item():.2f}")
        
            print("=== END DEBUG ===\n")
        
        if len(dataset) == 0:
            raise ValueError("No data loaded from dataset")
        
        # Separate data by receptor availability (simplified for 2 receptors)
        receptor_datasets = {
            'gcgr_glp1r': [],  # Samples with both GCGR and GLP1R data
            'gcgr_only': [],   # Samples with only GCGR data
            'glp1r_only': [],  # Samples with only GLP1R data
            'complete': dataset  # Complete dataset
        }
        
        # Initialize statistics for regression targets
        target_stats = {
            'GCGR': {
                'total': 0, 
                'values': [],
                'log_ec50_range': None,
                'mean_log_ec50': None,
                'std_log_ec50': None
            }, 
            'GLP1R': {
                'total': 0, 
                'values': [],
                'log_ec50_range': None,
                'mean_log_ec50': None,
                'std_log_ec50': None
            }
        }
        
        for data in dataset:
            targets = data.y.numpy()
            
            # Check data availability (using -999.0 as missing data marker for regression)
            has_gcgr = targets[0] != -999.0
            has_glp1r = targets[1] != -999.0
            
            # Categorize samples (only 2 receptors now)
            if has_gcgr and has_glp1r:
                receptor_datasets['gcgr_glp1r'].append(data)
            elif has_gcgr and not has_glp1r:
                receptor_datasets['gcgr_only'].append(data)
            elif not has_gcgr and has_glp1r:
                receptor_datasets['glp1r_only'].append(data)
            
            # Collect target values for statistics
            for i, receptor in enumerate(['GCGR', 'GLP1R']):
                if targets[i] != -999.0:  # Valid regression target
                    target_stats[receptor]['total'] += 1
                    target_stats[receptor]['values'].append(targets[i])
        
        # Compute regression statistics
        for receptor in ['GCGR', 'GLP1R']:
            if target_stats[receptor]['values']:
                values = np.array(target_stats[receptor]['values'])
                target_stats[receptor]['log_ec50_range'] = (np.min(values), np.max(values))
                target_stats[receptor]['mean_log_ec50'] = np.mean(values)
                target_stats[receptor]['std_log_ec50'] = np.std(values)
        
        # Print data distribution following Nature Chemistry style
        print(f"\nData Distribution (following Puszkarska et al. methodology):")
        print(f"  Total peptide sequences: {len(dataset)}")
        print(f"  Both GCGR + GLP1R data: {len(receptor_datasets['gcgr_glp1r'])}")
        print(f"  GCGR only: {len(receptor_datasets['gcgr_only'])}")
        print(f"  GLP1R only: {len(receptor_datasets['glp1r_only'])}")
        
        print(f"\nTarget Statistics (log10[EC50 (M)]):")
        print(f"{'Receptor':<8} {'Samples':<8} {'Range':<20} {'Mean ± SD':<15}")
        print("-" * 55)
        
        for receptor in ['GCGR', 'GLP1R']:
            stats = target_stats[receptor]
            if stats['total'] > 0:
                range_str = f"{stats['log_ec50_range'][0]:.2f} to {stats['log_ec50_range'][1]:.2f}"
                mean_std_str = f"{stats['mean_log_ec50']:.2f} ± {stats['std_log_ec50']:.2f}"
                print(f"{receptor:<8} {stats['total']:<8} {range_str:<20} {mean_std_str:<15}")
            else:
                print(f"{receptor:<8} {0:<8} {'No data':<20} {'N/A':<15}")
        
        # Additional information matching the original paper
        print(f"\nDataset Composition:")
        total_measurements = sum(stats['total'] for stats in target_stats.values())
        print(f"  Total EC50 measurements: {total_measurements}")
        
        # Note about data preprocessing
        print(f"\nNote: Following Puszkarska et al. preprocessing:")
        print(f"  - Sequences aligned and truncated to 30 amino acids")
        print(f"  - EC50 values log-transformed: log10[EC50 (M)]")
        print(f"  - Missing data marked as -999.0")
        
        return receptor_datasets, target_stats
    
    def create_data_loaders(self, 
                           dataset: List, 
                           batch_size: int,
                           test_size: float = 0.2,
                           val_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders with appropriate splits for regression."""
        
        if len(dataset) == 0:
            raise ValueError("Empty dataset provided")
        
        print(f"  Using random splits for regression targets (no stratification)")
        
        # Handle case where we don't want a separate test set (k-fold CV)
        if test_size == 0.0 or test_size is None:
            # Only create train/val split
            train_indices, val_indices = train_test_split(
                range(len(dataset)), 
                test_size=val_size, 
                random_state=self.training_config['random_seed']
            )
            
            # Create datasets
            train_dataset = [dataset[i] for i in train_indices]
            val_dataset = [dataset[i] for i in val_indices]
            test_dataset = []  # Empty test dataset
            
            # Create loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: 0 (k-fold)")
            
            return train_loader, val_loader, test_loader
        
        # Standard train/val/test split for regression (no stratification)
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=test_size, 
            random_state=self.training_config['random_seed']
        )
        
        # Second split: train vs val
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size / (1 - test_size),  # Adjust for remaining data
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
        Ensure batch.y is properly shaped as [batch_size, num_targets] for regression.
        
        Args:
            batch: Batch from DataLoader
            active_receptors: List of active receptor names
            
        Returns:
            None (modifies batch.y in place)
        """
        if batch.y.dim() == 1:
            # Determine batch size from batch vector
            batch_size = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 1
            num_targets = 2  # Only GCGR and GLP1R for regression
            
            # Reshape targets
            expected_size = batch_size * num_targets
            if batch.y.numel() == expected_size:
                batch.y = batch.y.view(batch_size, num_targets)
            else:
                # Handle size mismatch - this shouldn't happen with proper data loading
                print(f"Warning: Target tensor size mismatch. Expected {expected_size}, got {batch.y.numel()}")
                print(f"  Batch size: {batch_size}, Targets per sample: {num_targets}")
                print(f"  Active receptors: {active_receptors}")
                
                # For regression, use -999.0 as missing value marker
                if batch.y.numel() < expected_size:
                    # Pad with -999.0 (missing values for regression)
                    padded = torch.full((expected_size,), -999.0, dtype=batch.y.dtype, device=batch.y.device)
                    padded[:batch.y.numel()] = batch.y
                    batch.y = padded.view(batch_size, num_targets)
                else:
                    # Truncate to expected size
                    batch.y = batch.y[:expected_size].view(batch_size, num_targets)
        
        # Ensure we have the right shape
        if batch.y.shape[1] != 2:
            print(f"Warning: Target tensor has {batch.y.shape[1]} columns, expected 2")
            # Take only first 2 columns if we have more
            if batch.y.shape[1] > 2:
                batch.y = batch.y[:, :2]
    
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
        criterion = nn.MSELoss()
        
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
                 # DEBUG: Check batch properties

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
                    valid_mask = batch.y[:, receptor_idx] != -999.0  # Check for missing data marker
                    
                    if valid_mask.sum() > 0:
                        receptor_outputs = outputs[receptor][valid_mask]
                        receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                        
                        loss = criterion(receptor_outputs, receptor_targets)
                        log_var = self.model.task_log_vars[receptor]
                        loss = (1 / (2 * torch.exp(log_var))) * loss + 0.5 * log_var
                        total_loss += loss
                        valid_losses += 1
                
                if valid_losses > 0:
                    total_loss = total_loss / valid_losses

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
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
                        valid_mask = batch.y[:, receptor_idx] != -999.0
                        
                        if valid_mask.sum() > 0:
                            receptor_outputs = outputs[receptor][valid_mask]
                            receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                            
                            loss = criterion(receptor_outputs, receptor_targets)
                            total_loss += loss
                            valid_losses_val += 1
                            
                            # Store predictions for metrics
                            val_predictions[receptor].append(receptor_outputs.cpu())
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
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # 2. Learning rate scheduling
            if hasattr(self, 'lr_scheduler'):
                scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            best_r2 = -float('inf')
            current_r2 = val_metrics.get('GCGR', {}).get('r2_score', -float('inf'))
            if current_r2 > best_r2:
                best_r2 = current_r2
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        print(f"Stage '{stage_name}' completed. Best val loss: {best_val_loss:.4f}")
        
        
        return history
    
    def compute_metrics(self, 
                       predictions: Dict, 
                       targets: Dict, 
                       receptors: List[str]) -> Dict:
        """Compute regression metrics following Puszkarska et al. Nature Chemistry 2024 methodology."""
        metrics = {}
        
        for receptor in receptors:
            if receptor in predictions and len(predictions[receptor]) > 0:
                preds = predictions[receptor].numpy().flatten()
                targs = targets[receptor].numpy().flatten()
                
                receptor_metrics = {
                    'n_samples': int(len(targs)),
                    'target_range': f"{float(np.min(targs)):.2f} to {float(np.max(targs)):.2f}",
                    'mean_target': float(np.mean(targs)),
                }
                
                # Compute regression metrics following the original paper methodology
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # Primary metrics matching Puszkarska et al. Table 1
                    rmse_val = np.sqrt(mean_squared_error(targs, preds))
                    mae_val = mean_absolute_error(targs, preds)
                    r2_val = r2_score(targs, preds)
                    
                    # Pearson correlation coefficient
                    pearson_corr = np.corrcoef(targs, preds)[0, 1] if len(targs) > 1 else 0.0
                    
                    receptor_metrics.update({
                        'r.m.s.e.': float(rmse_val),  # Ensure JSON serializable
                        'm.a.e.': float(mae_val),     
                        'R²': float(r2_val),          
                        'Pearson_r': float(pearson_corr),
                        # Keep internal names for compatibility
                        'rmse': float(rmse_val),
                        'mae': float(mae_val),
                        'r2_score': float(r2_val),
                        'pearson_corr': float(pearson_corr)
                    })
                except Exception as e:
                    print(f"Warning: Could not compute metrics for {receptor}: {e}")
                    receptor_metrics.update({
                        'r.m.s.e.': float('inf'), 'm.a.e.': float('inf'), 'R²': 0.0,
                        'Pearson_r': 0.0, 'rmse': float('inf'), 'mae': float('inf'),
                        'r2_score': 0.0, 'pearson_corr': 0.0
                    })
                
                metrics[receptor] = receptor_metrics
        
        return metrics
    
    def create_kfold_splits(self, dataset, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Create k-fold splits for cross-validation (stratified for classification, random for regression).
        
        Args:
            dataset: Complete dataset
            n_splits: Number of folds
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        print(f"\nCreating {n_splits}-fold splits...")
        
        # Check if we're doing regression
        sample_targets = dataset[0].y.numpy()
        is_regression = len(sample_targets) == 2 and any(abs(val) > 10 for val in sample_targets if val != -999.0)
        
        if is_regression:
            print(f"Using random K-fold for regression targets")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.training_config['random_seed'])
            splits = list(kf.split(range(len(dataset))))
            
        else:
            print(f"Using stratified K-fold for classification targets")
            
            # Extract stratification information for classification
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

    def run_ensemble_regression_pipeline(self, csv_file: str, n_folds: int = 12, n_models: int = 6) -> Dict:
        """
        Run ensemble k-fold cross-validation for GCGR/GLP1R regression.
        Following Puszkarska et al. Nature Chemistry 2024 ensemble methodology.
        
        Args:
            csv_file: Path to data file
            n_folds: Number of folds for cross-validation (default: 12)
            n_models: Number of models per fold (default: 6)
            
        Returns:
            Complete results dictionary with ensemble predictions
        """
        print(f"\n{'='*80}")
        print(f"STARTING {n_models}-MODEL × {n_folds}-FOLD ENSEMBLE REGRESSION PIPELINE")
        print(f"{'='*80}")
        print("Following Puszkarska et al. Nature Chemistry 2024 ensemble methodology")
        print(f"Total model trainings: {n_models * n_folds}")
        print("Training on GCGR (T1) and GLP1R (T2) regression targets")
        
        # Prepare data
        receptor_datasets, target_stats = self.prepare_data(csv_file)
        complete_dataset = receptor_datasets['complete']
        
        if len(complete_dataset) == 0:
            raise ValueError("No data available for k-fold cross-validation")
        
        # Create k-fold splits
        fold_splits = self.create_kfold_splits(complete_dataset, n_folds)
        
        # Store all individual model results
        self.fold_results = []
        self.ensemble_predictions = {}  # Store ensemble predictions per fold
        
        # Generate base seeds for reproducibility
        base_seed = self.training_config['random_seed']
        model_seeds = [base_seed + i * 100 for i in range(n_models)]
        
        print(f"\nModel seeds: {model_seeds}")
        
        # Run each fold with ensemble training
        for fold_idx, (train_indices, test_indices) in enumerate(fold_splits):
            print(f"\n{'='*100}")
            print(f"PROCESSING FOLD {fold_idx + 1}/{n_folds}")
            print(f"{'='*100}")
            
            fold_ensemble_result = self.run_ensemble_fold_regression(
                complete_dataset, train_indices, test_indices, fold_idx, 
                model_seeds, n_models
            )
            
            self.fold_results.append(fold_ensemble_result)
        
        # Aggregate ensemble results
        self.aggregated_results = self.aggregate_ensemble_results(n_models, n_folds)
        
        # Compile complete results
        complete_results = {
            'methodology': 'Ensemble Multi-task GNN following Puszkarska et al. 2024',
            'ensemble_config': {
                'n_models_per_fold': n_models,
                'n_folds': n_folds,
                'total_model_trainings': n_models * n_folds,
                'model_seeds': model_seeds
            },
            'target_statistics': target_stats,
            'fold_results': self.fold_results,
            'aggregated_results': self.aggregated_results,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, "ensemble_regression_results.json")
        with open(results_path, 'w') as f:
            json_results = self._convert_for_json(complete_results)
            json.dump(json_results, f, indent=2)
        print(f"Complete ensemble results saved to {results_path}")
        
        # Generate enhanced reports
        self.generate_ensemble_summary_report(complete_results)
        self.print_ensemble_comparison_table(complete_results)
        
        return complete_results

    def run_ensemble_fold_regression(self, dataset, train_indices: List[int], test_indices: List[int], 
                                    fold_idx: int, model_seeds: List[int], n_models: int) -> Dict:
        """
        Run ensemble regression training for a single fold with multiple models.
        
        Args:
            dataset: Complete dataset
            train_indices: Training sample indices
            test_indices: Test sample indices  
            fold_idx: Current fold index
            model_seeds: List of seeds for each model
            n_models: Number of models to train per fold
            
        Returns:
            Dictionary with ensemble fold results
        """
        print(f"Training {n_models} models for fold {fold_idx + 1}")
        print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        
        # Create fold datasets
        train_dataset = [dataset[i] for i in train_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        # Create data loaders (shared across all models in this fold)
        train_loader, val_loader, _ = self.create_data_loaders(
            train_dataset, 
            batch_size=self.training_config['batch_size'],
            test_size=0.0
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Store results for all models in this fold
        fold_model_results = []
        fold_predictions = {'GCGR': [], 'GLP1R': []}
        fold_targets = {'GCGR': [], 'GLP1R': []}
        
        # Train each model with different seed
        for model_idx in range(n_models):
            model_seed = model_seeds[model_idx]
            print(f"\n--- Model {model_idx + 1}/{n_models} (Seed: {model_seed}) ---")
            
            # Set seed for this specific model
            self._set_seeds(model_seed)
            
            # Initialize fresh model for this iteration
            model = TransferLearningGNN(**self.model_config).to(self.device)
            
            # Store the current model and then restore it after training
            original_model = self.model
            self.model = model
            
            # Train this model instance
            history = self.train_stage(
                train_loader, val_loader,
                stage_name=f"fold{fold_idx + 1}_model{model_idx + 1}_regression",
                active_receptors=['GCGR', 'GLP1R'],
                epochs=self.training_config.get('regression_epochs', 100),
                learning_rate=self.training_config['learning_rate']
            )
            
            # Evaluate this model
            model_predictions, model_targets = self.evaluate_single_model(test_loader, ['GCGR', 'GLP1R'])
            
            # Store individual model results
            model_metrics = self.compute_metrics(model_predictions, model_targets, ['GCGR', 'GLP1R'])
            
            model_result = {
                'model_idx': model_idx + 1,
                'seed': model_seed,
                'training_history': history,
                'test_metrics': model_metrics,
                'predictions': {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in model_predictions.items()},
                'targets': {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in model_targets.items()}
            }
            fold_model_results.append(model_result)
            
            # Accumulate predictions for ensemble averaging
            for receptor in ['GCGR', 'GLP1R']:
                if receptor in model_predictions and len(model_predictions[receptor]) > 0:
                    fold_predictions[receptor].append(model_predictions[receptor].numpy().flatten())
                    # Only store targets once (they're the same for all models)
                    if model_idx == 0:
                        fold_targets[receptor] = model_targets[receptor].numpy().flatten()
            
            # Save individual model
            model_path = os.path.join(self.output_dir, f"fold_{fold_idx + 1}_model_{model_idx + 1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.model_config,
                'fold': fold_idx + 1,
                'model_idx': model_idx + 1,
                'seed': model_seed,
                'model_metrics': model_metrics
            }, model_path)
            
            # Restore original model reference
            self.model = original_model
        
        # Compute ensemble predictions (average across models)
        ensemble_predictions = {}
        ensemble_metrics = {}
        
        for receptor in ['GCGR', 'GLP1R']:
            if receptor in fold_predictions and fold_predictions[receptor]:
                # Average predictions across all models
                receptor_preds = np.array(fold_predictions[receptor])  # Shape: (n_models, n_samples)
                ensemble_pred = np.mean(receptor_preds, axis=0)  # Shape: (n_samples,)
                ensemble_std = np.std(receptor_preds, axis=0)    # Standard deviation across models
                
                ensemble_predictions[receptor] = ensemble_pred
                
                # Compute ensemble metrics
                if receptor in fold_targets:
                    targets = fold_targets[receptor]
                    
                    # Convert to torch tensors for metric computation
                    pred_tensor = torch.tensor(ensemble_pred, dtype=torch.float).unsqueeze(1)
                    target_tensor = torch.tensor(targets, dtype=torch.float).unsqueeze(1)
                    
                    ensemble_metrics[receptor] = self.compute_metrics(
                        {receptor: pred_tensor}, 
                        {receptor: target_tensor}, 
                        [receptor]
                    )[receptor]
                    
                    # Add ensemble-specific metrics
                    ensemble_metrics[receptor]['prediction_std_mean'] = float(np.mean(ensemble_std))
                    ensemble_metrics[receptor]['prediction_std_max'] = float(np.max(ensemble_std))
                    ensemble_metrics[receptor]['n_models'] = len(fold_predictions[receptor])
        
        # Compile fold results
        fold_ensemble_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'n_models': n_models,
            'individual_models': fold_model_results,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_metrics': ensemble_metrics,
            'model_seeds': model_seeds
        }
        
        # Print fold summary
        print(f"\nFold {fold_idx + 1} Ensemble Results:")
        for receptor in ['GCGR', 'GLP1R']:
            if receptor in ensemble_metrics:
                metrics = ensemble_metrics[receptor]
                rmse = metrics.get('r.m.s.e.', float('inf'))
                r2 = metrics.get('R²', 0)
                pred_std = metrics.get('prediction_std_mean', 0)
                n_samples = metrics.get('n_samples', 0)
                print(f"  {receptor}: r.m.s.e. = {rmse:.3f}, R² = {r2:.3f}, pred_std = {pred_std:.3f} (n = {n_samples})")
        
        return fold_ensemble_result

    def evaluate_single_model(self, test_loader: DataLoader, active_receptors: List[str]) -> Tuple[Dict, Dict]:
        """
        Evaluate a single model and return predictions and targets.
        
        Returns:
            Tuple of (predictions_dict, targets_dict)
        """
        self.model.eval()
        test_predictions = {receptor: [] for receptor in active_receptors}
        test_targets = {receptor: [] for receptor in active_receptors}
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                self._ensure_2d_targets(batch, active_receptors)
                
                outputs = self.model(batch.x, batch.edge_index, batch.batch, active_receptors)
                
                for receptor in active_receptors:
                    receptor_idx = self.model.receptor_to_idx[receptor]
                    valid_mask = batch.y[:, receptor_idx] != -999.0
                    
                    if valid_mask.sum() > 0:
                        receptor_outputs = outputs[receptor][valid_mask]
                        receptor_targets = batch.y[valid_mask, receptor_idx].float().unsqueeze(1)
                        
                        test_predictions[receptor].append(receptor_outputs.cpu())
                        test_targets[receptor].append(receptor_targets.cpu())
        
        # Aggregate predictions
        for receptor in active_receptors:
            if test_predictions[receptor]:
                test_predictions[receptor] = torch.cat(test_predictions[receptor], dim=0)
                test_targets[receptor] = torch.cat(test_targets[receptor], dim=0)
        
        return test_predictions, test_targets

    def aggregate_ensemble_results(self, n_models: int, n_folds: int) -> Dict:
        """
        Aggregate ensemble results across all folds and models.
        """
        print(f"\n{'='*60}")
        print("AGGREGATING ENSEMBLE RESULTS")
        print(f"{'='*60}")
        print(f"Total model instances: {n_models * n_folds}")
        
        aggregated = {
            'n_folds': n_folds,
            'n_models_per_fold': n_models,
            'total_model_instances': n_models * n_folds,
            'stages': {}
        }
        
        receptors = ['GCGR', 'GLP1R']
        metrics = ['r.m.s.e.', 'm.a.e.', 'R²', 'Pearson_r']
        
        # Aggregate ensemble results (one per fold)
        ensemble_aggregated = {}
        
        for receptor in receptors:
            receptor_metrics = {}
            
            for metric in metrics:
                values = []
                sample_counts = []
                
                for fold_result in self.fold_results:
                    ensemble_metrics = fold_result.get('ensemble_metrics', {})
                    if (receptor in ensemble_metrics and 
                        metric in ensemble_metrics[receptor] and
                        ensemble_metrics[receptor].get('n_samples', 0) > 0):
                        values.append(ensemble_metrics[receptor][metric])
                        sample_counts.append(ensemble_metrics[receptor]['n_samples'])
                
                if values:
                    receptor_metrics[f'{metric}_mean'] = float(np.mean(values))
                    receptor_metrics[f'{metric}_std'] = float(np.std(values))
                    receptor_metrics[f'{metric}_values'] = [float(v) for v in values]
                    receptor_metrics[f'{metric}_n_folds'] = int(len(values))
                else:
                    receptor_metrics[f'{metric}_mean'] = 0.0
                    receptor_metrics[f'{metric}_std'] = 0.0
                    receptor_metrics[f'{metric}_values'] = []
                    receptor_metrics[f'{metric}_n_folds'] = 0
            
            # Sample statistics
            if sample_counts:
                receptor_metrics['total_samples_mean'] = float(np.mean(sample_counts))
                receptor_metrics['total_samples_std'] = float(np.std(sample_counts))
                receptor_metrics['total_samples_sum'] = int(np.sum(sample_counts))
            else:
                receptor_metrics['total_samples_mean'] = 0.0
                receptor_metrics['total_samples_std'] = 0.0
                receptor_metrics['total_samples_sum'] = 0
            
            ensemble_aggregated[receptor] = receptor_metrics
        
        aggregated['stages']['ensemble'] = ensemble_aggregated
        
        return aggregated

    def print_ensemble_comparison_table(self, results: Dict):
        """
        Print ensemble model comparison table following Puszkarska et al. Table 1 format.
        """
        print(f"\n{'='*80}")
        print("ENSEMBLE MODEL PERFORMANCE COMPARISON")
        print("Following Puszkarska et al. Nature Chemistry 2024 - Table 1 Format")
        print("="*80)
        
        ensemble_config = results['ensemble_config']
        aggregated = results['aggregated_results']
        ensemble_data = aggregated['stages'].get('ensemble', {})
        
        # Header
        print(f"{'Models':<25} {'Cross-validation on training data':<40}")
        print(f"{'':25} {'R.m.s.e. GCGR':<20} {'R.m.s.e. GLP1R':<20}")
        print("-" * 85)
        
        # Our ensemble results
        gcgr_metrics = ensemble_data.get('GCGR', {})
        glp1r_metrics = ensemble_data.get('GLP1R', {})
        
        gcgr_rmse = f"{gcgr_metrics.get('r.m.s.e._mean', 0):.2f} ± {gcgr_metrics.get('r.m.s.e._std', 0):.2f}"
        glp1r_rmse = f"{glp1r_metrics.get('r.m.s.e._mean', 0):.2f} ± {glp1r_metrics.get('r.m.s.e._std', 0):.2f}"
        
        ensemble_name = f"Ensemble GNN ({ensemble_config['n_models_per_fold']}×{ensemble_config['n_folds']})"
        print(f"{ensemble_name:<25} {gcgr_rmse:<20} {glp1r_rmse:<20}")
        
        # Note about comparison with original paper
        print("-" * 85)
        print("Note: Original Puszkarska et al. results for comparison:")
        print("NN multi-task ensemble:      0.59 ± 0.05        0.68 ± 0.04")
        print("Random forest:              0.62 ± 0.04        0.77 ± 0.06")
        print("SVR:                        0.60 ± 0.04        0.81 ± 0.07")
        print("Ridge:                      0.63 ± 0.06        0.75 ± 0.10")
        print(f"\nOur approach: {ensemble_config['total_model_trainings']} total model trainings")
        print(f"Original approach: Multiple model instantiations with different random seeds")

    def generate_ensemble_summary_report(self, results: Dict):
        """Generate comprehensive summary report for ensemble results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"ensemble_summary_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("ENSEMBLE REGRESSION GNN - COMPREHENSIVE REPORT\n")
            f.write("Following Puszkarska et al. Nature Chemistry 2024 Ensemble Methodology\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Ensemble configuration
            ensemble_config = results['ensemble_config']
            f.write("ENSEMBLE CONFIGURATION\n")
            f.write("-"*50 + "\n")
            f.write(f"Models per fold: {ensemble_config['n_models_per_fold']}\n")
            f.write(f"Number of folds: {ensemble_config['n_folds']}\n")
            f.write(f"Total model trainings: {ensemble_config['total_model_trainings']}\n")
            f.write(f"Model seeds: {ensemble_config['model_seeds']}\n\n")
            
            # Performance results
            aggregated = results['aggregated_results']
            ensemble_data = aggregated['stages'].get('ensemble', {})
            
            f.write("ENSEMBLE PERFORMANCE RESULTS\n")
            f.write("-"*50 + "\n")
            f.write("Multi-task GNN Ensemble Performance:\n\n")
            
            for receptor in ['GCGR', 'GLP1R']:
                if receptor in ensemble_data:
                    metrics = ensemble_data[receptor]
                    f.write(f"{receptor} (EC50 log10[M] Prediction):\n")
                    f.write(f"  Validation methodology: {metrics['r.m.s.e._n_folds']}-fold cross-validation\n")
                    f.write(f"  Models per fold: {ensemble_config['n_models_per_fold']}\n")
                    f.write(f"  Root Mean Square Error: {metrics['r.m.s.e._mean']:.3f} ± {metrics['r.m.s.e._std']:.3f}\n")
                    f.write(f"  Mean Absolute Error: {metrics['m.a.e._mean']:.3f} ± {metrics['m.a.e._std']:.3f}\n")
                    f.write(f"  Coefficient of Determination (R²): {metrics['R²_mean']:.3f} ± {metrics['R²_std']:.3f}\n")
                    f.write(f"  Pearson Correlation: {metrics['Pearson_r_mean']:.3f} ± {metrics['Pearson_r_std']:.3f}\n")
                    f.write(f"  Total samples: {int(metrics['total_samples_sum'])}\n\n")
        
        print(f"Ensemble summary report saved to {report_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import torch
        
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_for_json(v) for v in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Catch any remaining numpy scalars
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
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
        
        Includes:
        - AUC-ROC comparison across receptors and stages
        - F1-Score comparison
        - AUC-ROC distribution across folds 
        - Training curves averaged across folds
        - Comprehensive metrics heatmap
        - Sample distribution analysis
        """
        if not self.fold_results or not self.aggregated_results:
            print("No results available for plotting")
            return
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        receptors = ['GCGR', 'GLP1R', 'GIPR']
        stages = ['stage1', 'stage2', 'stage3']
        stage_labels = ['Stage 1\n(GCGR+GLP1R)', 'Stage 2\n(GIPR Transfer)', 'Stage 3\n(Unified)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # ===== 1. AUC-ROC Comparison Across Receptors and Stages =====
        ax1 = fig.add_subplot(gs[0, 0:2])
        
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
                        receptor_labels.append(f'{receptor}\n{stage_labels[stage_idx]}')
                        stage_positions.append(x_pos)
                        x_pos += 1
            
            if stage_idx < len(stages) - 1:
                x_pos += 0.5  # Add space between stages
        
        ax1.set_title('AUC-ROC Comparison Across Receptors and Stages', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUC-ROC Score')
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(stage_positions)
        ax1.set_xticklabels(receptor_labels, rotation=45, ha='right')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # ===== 2. F1-Score Comparison =====
        ax2 = fig.add_subplot(gs[0, 2:4])
        
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
        
        ax2.set_title('F1-Score Comparison Across Receptors and Stages', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1.1)
        ax2.set_xticks(f1_positions)
        ax2.set_xticklabels(receptor_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # ===== 3. AUC-ROC Distribution Across Folds (Box Plots) =====
        ax3 = fig.add_subplot(gs[1, 0:2])
        
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
            
            ax3.set_title('AUC-ROC Distribution Across Folds (Final Stage)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('AUC-ROC Score')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # ===== 4. Training Curves (Average Across Folds) =====
        ax4 = fig.add_subplot(gs[1, 2:4])
        
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
                ax4.plot(epochs, mean_train_loss, 'b-', linewidth=2, label='Training Loss')
                ax4.plot(epochs, mean_val_loss, 'r-', linewidth=2, label='Validation Loss')
                
                # Add confidence intervals
                ax4.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                                alpha=0.2, color='blue')
                ax4.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                                alpha=0.2, color='red')
                
                ax4.set_title(f'Training Curves (Unified Stage, {len(all_train_losses)} folds)', 
                             fontsize=14, fontweight='bold')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # ===== 5. Comprehensive Metrics Heatmap =====
        ax5 = fig.add_subplot(gs[2, 0:2])
        
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
            
            ax5.set_title('Comprehensive Metrics Heatmap (Final Stage)', fontsize=14, fontweight='bold')
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
        
        # ===== 6. Sample Distribution Analysis =====
        ax6 = fig.add_subplot(gs[2, 2:4])
        
        # Show sample counts across folds and stages
        sample_data = {'GCGR': [], 'GLP1R': [], 'GIPR': []}
        
        for receptor in receptors:
            stage3_data = self.aggregated_results['stages'].get('stage3', {})
            if receptor in stage3_data:
                total_samples = stage3_data[receptor].get('total_samples_sum', 0)
                sample_data[receptor] = [total_samples]
        
        x_positions = np.arange(len(receptors))
        bar_values = [sample_data[receptor][0] if sample_data[receptor] else 0 for receptor in receptors]
        
        bars = ax6.bar(x_positions, bar_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, bar_values):
            if value > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar_values)*0.01,
                        f'{int(value)}', ha='center', va='bottom', fontsize=10)
        
        ax6.set_title('Total Sample Distribution (Final Stage)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Receptor')
        ax6.set_ylabel('Total Samples Across Folds')
        ax6.set_xticks(x_positions)
        ax6.set_xticklabels(receptors)
        ax6.grid(True, alpha=0.3)
        
        # ===== 7. Stage Progression Analysis =====
        ax7 = fig.add_subplot(gs[3, 0:2])
        
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
                ax7.errorbar(valid_stages, valid_aucs, yerr=valid_errors, 
                            marker='o', linewidth=2, markersize=8, capsize=5,
                            color=colors[receptor_idx], label=receptor)
        
        ax7.set_title('Performance Progression Across Stages', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Training Stage')
        ax7.set_ylabel('AUC-ROC Score')
        ax7.set_xticks(range(len(stage_names)))
        ax7.set_xticklabels(stage_names)
        ax7.set_ylim(0, 1.1)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # ===== 8. Model Convergence Analysis =====
        ax8 = fig.add_subplot(gs[3, 2:4])
        
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
                        f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax8.set_title('Final Loss Values Across Folds', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Loss Value')
            ax8.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.suptitle(f'Comprehensive K-Fold Cross-Validation Results ({self.aggregated_results["n_folds"]} folds)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "comprehensive_kfold_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive results plot saved to {plot_path}")
        
        plt.show()

   

def main():
    """Main function to run the complete k-fold cross-validation transfer learning pipeline."""
    
    print("🧬 K-FOLD CROSS-VALIDATION TRANSFER LEARNING GNN")
    print("="*80)
    
    # Configuration
    model_config = {
        'num_node_features': 5,
        'hidden_dim': 96,
        'num_layers': 4,
        'dropout': 0.2,
        'use_attention': True
    }
    
    training_config = {
        'random_seed': 42,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'patience': 15,
        'regression_epochs': 150  # Single training stage
            # Add gradient clipping
        # Remove transfer learning specific configs
    }
    
    data_config = {
        # Note: high_affinity_cutoff not used for regression, kept for compatibility
        'high_affinity_cutoff_pM': 1000.0,
        'target_type': 'regression',
        'missing_data_marker': -999.0
    }
    
    # K-fold configuration
    n_folds = 4  # Changed from 10 to 12
    n_models = 4  # New parameter for ensemble
    
    # Print configuration
    print("Configuration:")
    print(f"  Model: GAT with {model_config['hidden_dim']} hidden dims, {model_config['num_layers']} layers")
    print(f"  K-fold CV: {n_folds} folds")
 
    
    # Initialize pipeline
    pipeline = TransferLearningPipeline(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir="ensemble_regression_results_GAT"
    )
    
    # Run k-fold cross-validation pipeline
    try:
        print(f"\n🚀 Starting {n_folds}-fold regression cross-validation...")
        results = pipeline.run_ensemble_regression_pipeline('training_data.csv', n_folds=n_folds, n_models=n_models)
        
        print(f"\n✅ K-fold cross-validation pipeline completed successfully!")
        print(f"Results saved to: {pipeline.output_dir}")
        
        # Print final summary
        print(f"\n📊 FINAL K-FOLD RESULTS SUMMARY (Nature Chemistry Style):")
        print(f"{'='*80}")
        print("Multi-task GNN Performance Evaluation")
        print("Following Puszkarska et al. Nature Chemistry 2024 methodology")
        print("-" * 80)
        
        aggregated = results['aggregated_results']
        regression_data = aggregated['stages'].get('regression', {})
        
        # Create results table matching Table 1 format from the paper
        print(f"{'Model':<20} {'Cross-validation on training data':<35} {'Samples':<10}")
        print(f"{'':20} {'R.m.s.e. GCGR':<17} {'R.m.s.e. GLP1R':<18} {'Total':<10}")
        print("-" * 80)
        
        for receptor in ['GCGR', 'GLP1R']:
            if receptor in regression_data:
                metrics = regression_data[receptor]
                rmse_mean = metrics.get('r.m.s.e._mean', float('inf'))
                rmse_std = metrics.get('r.m.s.e._std', 0)
                n_folds_data = metrics.get('r.m.s.e._n_folds', 0)
                total_samples = int(metrics.get('total_samples_sum', 0))
                
                if receptor == 'GCGR':
                    print(f"{'Multi-task GNN':<20} {rmse_mean:.2f} ± {rmse_std:.2f}", end="")
                else:
                    print(f"{rmse_mean:.2f} ± {rmse_std:.2f} {total_samples:<10}")
        
        print("\nDetailed Performance Metrics:")
        print("-" * 50)
        for receptor in ['GCGR', 'GLP1R']:
            if receptor in regression_data:
                metrics = regression_data[receptor]
                rmse_mean = metrics.get('r.m.s.e._mean', float('inf'))
                rmse_std = metrics.get('r.m.s.e._std', 0)
                r2_mean = metrics.get('R²_mean', 0)
                r2_std = metrics.get('R²_std', 0)
                mae_mean = metrics.get('m.a.e._mean', float('inf'))
                mae_std = metrics.get('m.a.e._std', 0)
                n_folds_data = metrics.get('r.m.s.e._n_folds', 0)
                
                print(f"{receptor}:")
                print(f"  r.m.s.e.: {rmse_mean:.3f} ± {rmse_std:.3f}")
                print(f"  m.a.e.:   {mae_mean:.3f} ± {mae_std:.3f}")
                print(f"  R²:       {r2_mean:.3f} ± {r2_std:.3f}")
                print(f"  Folds:    {n_folds_data}/{n_folds}")
        
        print(f"\n📈 Comprehensive visualization generated!")
        print(f"📋 Detailed reports saved!")

        
    
        
        return results
        
    except Exception as e:
        print(f"❌ Error during k-fold pipeline execution: {e}")
        raise


if __name__ == "__main__":
    results = main()