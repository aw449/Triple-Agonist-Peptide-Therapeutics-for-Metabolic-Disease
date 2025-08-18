"""
Peptide Activity Regression Model Comparison with Confidence Intervals
Compares GAT-based and CNN-based ensemble models for EC50 prediction
Enhanced with bootstrap confidence intervals and uncertainty quantification
"""

import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import necessary modules from the provided code
from peptide_models.aminoacids import AMINOACIDS, LETTER2AA
from peptide_models.utils_data import get_peptides
from peptide_models.utils_models import get_predictions  # <-- Using the provided function
from gat_dual_regression_pipeline import load_peptide_dataset
from gat_dual_regression_trainer import TransferLearningGNN

class PeptideModelComparison:
    """
    Comprehensive comparison between GAT and CNN ensemble models
    for peptide EC50 regression prediction with confidence intervals.
    """
    
    def __init__(self, data_file: str, 
                 gat_model_dir: str = "ensemble_regression_results_GAT",
                 cnn_model_dir: str = "peptide_models",
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize the comparison framework.
        
        Args:
            data_file: Path to reference data CSV
            gat_model_dir: Directory containing GAT model folds
            cnn_model_dir: Directory containing CNN model folds
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.data_file = data_file
        self.gat_model_dir = Path(gat_model_dir)
        self.cnn_model_dir = Path(cnn_model_dir)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Storage for models and predictions
        self.gat_models = []
        self.cnn_individual_predictions = []  # Store individual model predictions
        self.gat_individual_predictions = []  # Store individual model predictions
        self.data = None
        self.results = {}
        
        print("🧬 Peptide Model Comparison Framework Initialized (with Confidence Intervals)")
        print(f"Data file: {data_file}")
        print(f"GAT models: {gat_model_dir}")
        print(f"CNN models: {cnn_model_dir}")
        print(f"Bootstrap samples: {n_bootstrap}")
        print(f"Confidence level: {confidence_level}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate the reference data."""
        print("\n📊 Loading reference data...")
        
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.data)} peptide sequences")
            
            # Validate required columns
            required_cols = ['sequence', 'aligned_sequence', 'EC50_LOG_T1', 'EC50_LOG_T2']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for missing values in targets
            target_cols = ['EC50_LOG_T1', 'EC50_LOG_T2']
            for col in target_cols:
                missing_count = self.data[col].isna().sum()
                if missing_count > 0:
                    print(f"⚠️  Warning: {missing_count} missing values in {col}")
            
            # Print data summary
            print(f"Sequence length range: {self.data['sequence'].str.len().min()}-{self.data['sequence'].str.len().max()}")
            print(f"EC50_Log_T1 range: {self.data['EC50_LOG_T1'].min():.2f} to {self.data['EC50_LOG_T1'].max():.2f}")
            print(f"EC50_Log_T2 range: {self.data['EC50_LOG_T2'].min():.2f} to {self.data['EC50_LOG_T2'].max():.2f}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        metric_func, n_bootstrap: int = None) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_func: Function to calculate metric
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (metric_value, lower_ci, upper_ci)
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
            
        # Original metric
        original_metric = metric_func(y_true, y_pred)
        
        # Bootstrap sampling
        bootstrap_metrics = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                boot_metric = metric_func(y_true_boot, y_pred_boot)
                if not np.isnan(boot_metric) and not np.isinf(boot_metric):
                    bootstrap_metrics.append(boot_metric)
            except:
                continue
        
        if len(bootstrap_metrics) == 0:
            return original_metric, np.nan, np.nan
        
        # Calculate confidence intervals
        bootstrap_metrics = np.array(bootstrap_metrics)
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
        upper_ci = np.percentile(bootstrap_metrics, upper_percentile)
        
        return original_metric, lower_ci, upper_ci
    
    def load_gat_models(self) -> List[Any]:
        """Load all GAT model folds."""
        print("\n🧠 Loading GAT ensemble models...")
        
        self.gat_models = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for fold in range(1, 13):  # fold1_model.pth to fold12_model.pth
            for model in range(1, 7):  # model1 to model6
                model_path = self.gat_model_dir / f"fold_{fold}_model_{model}.pth"
                
                if not model_path.exists():
                    print(f"⚠️  Warning: {model_path} not found, skipping...")
                    continue
                
                try:
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Extract model configuration
                    model_config = checkpoint.get('model_config', {
                        'num_node_features': 7,
                        'hidden_dim': 96,
                        'num_layers': 5,
                        'dropout': 0.3,
                        'use_attention': True
                    })
                    
                    # Initialize model
                    model = TransferLearningGNN(**model_config).to(device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    self.gat_models.append(model)
                    
                except Exception as e:
                    print(f"❌ Error loading GAT fold {fold}: {e}")
            
        print(f"📈 Loaded {len(self.gat_models)} GAT models")
        return self.gat_models
    
    def prepare_gat_input(self, aligned_sequences: List[str]):
        """Prepare input data for GAT models using actual training data for proper normalization."""
        print("🔄 Preparing GAT input data with proper normalization...")
        
        try:
            # Use the original training data to get proper normalization parameters
            original_training_data = pd.read_csv('training_data.csv')  # The file used for GAT training
            
            # Create dataset from original training data to get normalization params
            print("📊 Loading original training dataset for normalization parameters...")
            training_dataset = load_peptide_dataset('training_data.csv')
            
            # Store the training normalization parameters
            if hasattr(training_dataset, 'target_means') and hasattr(training_dataset, 'target_stds'):
                self.gat_normalization_params = {
                    0: {'mean': training_dataset.target_means[0], 'std': training_dataset.target_stds[0]},
                    1: {'mean': training_dataset.target_means[1], 'std': training_dataset.target_stds[1]}
                }
            else:
                # Fallback: compute from training data the same way as during training
                self._compute_training_normalization_params(original_training_data)
            
            # Now create prediction dataset with dummy values (they won't be used for prediction)
            temp_df = pd.DataFrame({
                'pep_ID': [f'seq_{i}' for i in range(len(aligned_sequences))],
                'sequence': aligned_sequences,
                'EC50_LOG_T1': [-999.0] * len(aligned_sequences),  # Missing data markers
                'EC50_LOG_T2': [-999.0] * len(aligned_sequences)   # Missing data markers
            })
            
            temp_file = 'temp_gat_input.csv'
            temp_df.to_csv(temp_file, index=False)
            
            # Load using GAT pipeline (but don't normalize since we have dummy data)
            dataset = load_peptide_dataset(temp_file)
            
            # Clean up temp file
            Path(temp_file).unlink()
            
            print(f"✅ Prepared GAT input for {len(dataset)} sequences")
            print(f"📊 Using training normalization: T1 mean={self.gat_normalization_params[0]['mean']:.3f}, std={self.gat_normalization_params[0]['std']:.3f}")
            print(f"📊 Using training normalization: T2 mean={self.gat_normalization_params[1]['mean']:.3f}, std={self.gat_normalization_params[1]['std']:.3f}")
            
            return dataset
        
        except Exception as e:
            print(f"❌ Error preparing GAT input: {e}")
            raise
    
    def prepare_cnn_input(self, sequences: List[str]) -> np.ndarray:
        """Prepare input data for CNN models using the provided utils."""
        print("🔄 Preparing CNN input data using utils_data...")
        
        try:
            # Filter out NaN values and convert to strings
            valid_sequences = [str(seq) for seq in sequences if pd.notna(seq) and str(seq).strip() != '']
            
            if len(valid_sequences) == 0:
                raise ValueError("No valid sequences found after filtering NaN values")
            
            print(f"📊 Filtered {len(sequences)} -> {len(valid_sequences)} valid sequences")
            
            # Create temporary DataFrame in the expected format
            temp_df = pd.DataFrame({
                'pep_ID': [f'seq_{i}' for i in range(len(valid_sequences))],
                'sequence': valid_sequences
            })
            
            # Use the provided get_peptides function from utils_data
            peptides = get_peptides(temp_df)
            
            # Extract one-hot encoded sequences following the original pipeline
            X = np.asarray([p.encoded_onehot for p in peptides])
            X_test = X.reshape(X.shape[0], X.shape[1], len(AMINOACIDS))
            
            print(f"✅ Prepared CNN input shape: {X_test.shape}")
            print(f"   Number of amino acids in encoding: {len(AMINOACIDS)}")
            
            return X_test
            
        except Exception as e:
            print(f"❌ Error preparing CNN input: {e}")
            raise

    def _compute_training_normalization_params(self, training_data: pd.DataFrame):
        """Compute normalization parameters the same way as during GAT training."""
        print("🔢 Computing normalization parameters from original training data...")
        
        # Collect all valid targets from training data
        all_targets = []
        for idx, row in training_data.iterrows():
            ec50_t1 = row.get('EC50_LOG_T1', np.nan)
            ec50_t2 = row.get('EC50_LOG_T2', np.nan)
            
            if not pd.isna(ec50_t1):
                all_targets.append((0, ec50_t1))
            if not pd.isna(ec50_t2):
                all_targets.append((1, ec50_t2))
        
        # Calculate statistics for each receptor separately (same as training)
        receptor_stats = {}
        for receptor_idx, target_val in all_targets:
            if receptor_idx not in receptor_stats:
                receptor_stats[receptor_idx] = []
            receptor_stats[receptor_idx].append(target_val)
        
        # Store normalization parameters
        self.gat_normalization_params = {}
        for receptor_idx, values in receptor_stats.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            self.gat_normalization_params[receptor_idx] = {'mean': mean_val, 'std': std_val}
            print(f"  Receptor {receptor_idx}: mean={mean_val:.3f}, std={std_val:.3f}")

    def predict_gat_ensemble(self, gat_input_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using GAT ensemble.
        
        Returns:
            Tuple of (ensemble_predictions, prediction_uncertainty)
        """
        print("🧠 Generating GAT ensemble predictions with uncertainty...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_predictions = []
        
        from torch_geometric.loader import DataLoader
        
        # Create DataLoader
        data_loader = DataLoader(gat_input_data, batch_size=32, shuffle=False)
        
        for i, model in enumerate(self.gat_models):
            model.eval()
            fold_predictions = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    
                    # Get predictions for both receptors
                    outputs = model(batch.x, batch.edge_index, batch.batch, ['GCGR', 'GLP1R'])
                    
                    # Extract predictions (these are in normalized space)
                    gcgr_pred = outputs['GCGR'].cpu().numpy().flatten()
                    glp1r_pred = outputs['GLP1R'].cpu().numpy().flatten()
                    
                    # Stack predictions [T1, T2]
                    batch_pred = np.column_stack([gcgr_pred, glp1r_pred])
                    fold_predictions.append(batch_pred)
            
            # Concatenate all batches for this fold
            fold_pred = np.vstack(fold_predictions)
            all_predictions.append(fold_pred)
            
            print(f"✅ GAT fold {i+1} predictions: {fold_pred.shape}")
        
        # Store individual predictions for uncertainty calculation
        self.gat_individual_predictions = np.array(all_predictions)
        
        # Average across folds (still in normalized space)
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        # Calculate prediction uncertainty (standard deviation across models)
        prediction_std = np.std(all_predictions, axis=0)
        
        # Denormalize predictions using the SAME parameters as training
        denormalized_predictions = self._denormalize_predictions(ensemble_predictions)
        denormalized_std = self._denormalize_predictions(prediction_std, is_std=True)
        
        print(f"🎯 GAT ensemble predictions shape: {denormalized_predictions.shape}")
        print(f"📊 GAT prediction uncertainty shape: {denormalized_std.shape}")
        print("📊 GAT predictions denormalized using original training parameters")
        
        return denormalized_predictions, denormalized_std
    
    def _denormalize_predictions(self, normalized_preds: np.ndarray, is_std: bool = False) -> np.ndarray:
        """Denormalize predictions using the same parameters as training."""
        denormalized = normalized_preds.copy()
        
        for receptor_idx in range(normalized_preds.shape[1]):
            if receptor_idx in self.gat_normalization_params:
                mean_val = self.gat_normalization_params[receptor_idx]['mean']
                std_val = self.gat_normalization_params[receptor_idx]['std']
                
                if is_std:
                    # For standard deviations, only multiply by scale (std), don't add mean
                    denormalized[:, receptor_idx] = normalized_preds[:, receptor_idx] * std_val
                else:
                    # Denormalize: original = normalized * std + mean
                    denormalized[:, receptor_idx] = (normalized_preds[:, receptor_idx] * std_val) + mean_val
                
                print(f"🔈 Denormalized receptor {receptor_idx} using mean={mean_val:.3f}, std={std_val:.3f}")
        
        return denormalized
    
    def predict_cnn_ensemble(self, cnn_input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using CNN ensemble with uncertainty.
        
        Returns:
            Tuple of (ensemble_predictions, prediction_uncertainty)
        """
        print("🔗 Generating CNN ensemble predictions with uncertainty...")
        
        try:
            # Find model directories (model0 to model5)
            model_dirs = [d for d in self.cnn_model_dir.iterdir() if d.is_dir() and d.name.startswith('models')]
            model_dirs = sorted(model_dirs, key=lambda x: int(x.name[6:]))  # Sort by model number

            if len(model_dirs) == 0:
                raise ValueError(f"No model directories found in {self.cnn_model_dir}")

            # Collect all .keras files from all model directories
            keras_files = []
            for model_dir in model_dirs:
                keras_files.extend([f for f in model_dir.iterdir() if f.suffix == '.keras'])

            if len(keras_files) == 0:
                raise ValueError(f"No .keras model files found in model directories")

            print(f"🔍 Found {len(model_dirs)} model directories with {len(keras_files)} total .keras files")
            
            # Load models and generate predictions
            y_predicted = []
            
            for f_path in sorted(keras_files):
                print(f'Loading model: {f_path.parent.name}/{f_path.name}')
                
                # Import here to avoid conflicts
                from tensorflow import keras
                from keras.models import load_model
                
                model = load_model(str(f_path))
                y_hat_current = model.predict(cnn_input_data, verbose=0)
                y_predicted.append(y_hat_current)
                del model
            
            y_predicted = np.asarray(y_predicted)
            
            print(f"✅ CNN predictions shape: {y_predicted.shape}")
            print(f"   Expected shape: [num_models, num_targets, num_samples]")
            
            # Store individual predictions for uncertainty calculation
            self.cnn_individual_predictions = y_predicted
            
            # Handle 4D shape: (num_models, num_targets, num_samples, 1)
            if len(y_predicted.shape) == 4:
                # Remove last dimension and average across models
                y_predicted = y_predicted.squeeze(-1)  # Remove last dim: (72, 2, 19)
                ensemble_predictions = np.mean(y_predicted, axis=0)  # Shape: (2, 19)
                prediction_std = np.std(y_predicted, axis=0)  # Shape: (2, 19)
                ensemble_predictions = ensemble_predictions.T  # Shape: (19, 2)
                prediction_std = prediction_std.T  # Shape: (19, 2)
                
            elif len(y_predicted.shape) == 3:
                # Average across models (axis 0)
                ensemble_predictions = np.mean(y_predicted, axis=0)  # Shape: [num_targets, num_samples]
                prediction_std = np.std(y_predicted, axis=0)  # Shape: [num_targets, num_samples]
                # Transpose to get [num_samples, num_targets]
                ensemble_predictions = ensemble_predictions.T
                prediction_std = prediction_std.T
                
            else:
                raise ValueError(f"Unexpected prediction shape: {y_predicted.shape}")
            
            print(f"🎯 CNN ensemble predictions final shape: {ensemble_predictions.shape}")
            print(f"📊 CNN prediction uncertainty shape: {prediction_std.shape}")
            
            return ensemble_predictions, prediction_std
            
        except Exception as e:
            print(f"❌ Error generating CNN predictions: {e}")
            raise
    
    def compute_metrics_with_ci(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               target_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compute regression metrics with confidence intervals."""
        print("📊 Computing metrics with bootstrap confidence intervals...")
        
        metrics = {}
        
        for i, target in enumerate(target_names):
            # Filter out missing values
            mask = ~np.isnan(y_true[:, i])
            true_vals = y_true[mask, i]
            pred_vals = y_pred[mask, i]
            
            if len(true_vals) == 0:
                metrics[target] = {
                    'mse': np.nan, 'mse_ci': (np.nan, np.nan),
                    'rmse': np.nan, 'rmse_ci': (np.nan, np.nan),
                    'r2': np.nan, 'r2_ci': (np.nan, np.nan),
                    'pearson_r': np.nan, 'pearson_r_ci': (np.nan, np.nan),
                    'n_samples': 0
                }
                continue
            
            # Compute metrics with confidence intervals
            mse, mse_low, mse_high = self.bootstrap_metric(
                true_vals, pred_vals, lambda t, p: mean_squared_error(t, p))
            
            rmse, rmse_low, rmse_high = self.bootstrap_metric(
                true_vals, pred_vals, lambda t, p: np.sqrt(mean_squared_error(t, p)))
            
            r2, r2_low, r2_high = self.bootstrap_metric(
                true_vals, pred_vals, lambda t, p: r2_score(t, p))
            
            pearson_r, pearson_low, pearson_high = self.bootstrap_metric(
                true_vals, pred_vals, lambda t, p: stats.pearsonr(t, p)[0])
            
            metrics[target] = {
                'mse': mse,
                'mse_ci': (mse_low, mse_high),
                'rmse': rmse,
                'rmse_ci': (rmse_low, rmse_high),
                'r2': r2,
                'r2_ci': (r2_low, r2_high),
                'pearson_r': pearson_r,
                'pearson_r_ci': (pearson_low, pearson_high),
                'n_samples': len(true_vals)
            }
            
            print(f"✅ {target}: RMSE = {rmse:.3f} [{rmse_low:.3f}, {rmse_high:.3f}]")
            print(f"   R² = {r2:.3f} [{r2_low:.3f}, {r2_high:.3f}]")
        
        return metrics
    
    def perform_statistical_tests(self, y_true: np.ndarray, 
                                 gat_pred: np.ndarray, cnn_pred: np.ndarray,
                                 target_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Perform paired statistical tests between model predictions."""
        print("\n📊 Performing statistical significance tests...")
        
        test_results = {}
        
        for i, target in enumerate(target_names):
            # Filter out missing values
            mask = ~np.isnan(y_true[:, i])
            true_vals = y_true[mask, i]
            gat_vals = gat_pred[mask, i]
            cnn_vals = cnn_pred[mask, i]
            
            if len(true_vals) < 2:
                test_results[target] = {'t_stat': np.nan, 'p_value': np.nan, 'significance': 'insufficient_data'}
                continue
            
            # Compute squared errors for each model
            gat_errors = (true_vals - gat_vals) ** 2
            cnn_errors = (true_vals - cnn_vals) ** 2
            
            # Paired t-test on squared errors (lower is better)
            t_stat, p_value = stats.ttest_rel(gat_errors, cnn_errors)
            
            # Determine significance and direction
            alpha = 0.05
            if p_value < alpha:
                if t_stat < 0:  # GAT has lower errors
                    significance = 'GAT_significantly_better'
                else:  # CNN has lower errors
                    significance = 'CNN_significantly_better'
            else:
                significance = 'no_significant_difference'
            
            test_results[target] = {
                't_stat': t_stat,
                'p_value': p_value,
                'significance': significance,
                'gat_mean_error': np.mean(gat_errors),
                'cnn_mean_error': np.mean(cnn_errors)
            }
            
            print(f"📈 {target}: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        return test_results
    
    def create_comparison_table_with_ci(self, gat_metrics: Dict, cnn_metrics: Dict, 
                                       test_results: Dict) -> pd.DataFrame:
        """Create a comprehensive comparison table with confidence intervals."""
        
        comparison_data = []
        
        for target in ['EC50_LOG_T1', 'EC50_LOG_T2']:
            gat_m = gat_metrics.get(target, {})
            cnn_m = cnn_metrics.get(target, {})
            test_r = test_results.get(target, {})
            
            # Format confidence intervals
            gat_rmse_ci = gat_m.get('rmse_ci', (np.nan, np.nan))
            cnn_rmse_ci = cnn_m.get('rmse_ci', (np.nan, np.nan))
            gat_r2_ci = gat_m.get('r2_ci', (np.nan, np.nan))
            cnn_r2_ci = cnn_m.get('r2_ci', (np.nan, np.nan))
            gat_pearson_ci = gat_m.get('pearson_r_ci', (np.nan, np.nan))
            cnn_pearson_ci = cnn_m.get('pearson_r_ci', (np.nan, np.nan))
            
            comparison_data.append({
                'Target': target,
                'GAT_RMSE': gat_m.get('rmse', np.nan),
                'GAT_RMSE_CI_Low': gat_rmse_ci[0],
                'GAT_RMSE_CI_High': gat_rmse_ci[1],
                'CNN_RMSE': cnn_m.get('rmse', np.nan),
                'CNN_RMSE_CI_Low': cnn_rmse_ci[0],
                'CNN_RMSE_CI_High': cnn_rmse_ci[1],
                'GAT_R²': gat_m.get('r2', np.nan),
                'GAT_R²_CI_Low': gat_r2_ci[0],
                'GAT_R²_CI_High': gat_r2_ci[1],
                'CNN_R²': cnn_m.get('r2', np.nan),
                'CNN_R²_CI_Low': cnn_r2_ci[0],
                'CNN_R²_CI_High': cnn_r2_ci[1],
                'GAT_Pearson_r': gat_m.get('pearson_r', np.nan),
                'GAT_Pearson_r_CI_Low': gat_pearson_ci[0],
                'GAT_Pearson_r_CI_High': gat_pearson_ci[1],
                'CNN_Pearson_r': cnn_m.get('pearson_r', np.nan),
                'CNN_Pearson_r_CI_Low': cnn_pearson_ci[0],
                'CNN_Pearson_r_CI_High': cnn_pearson_ci[1],
                'p_value': test_r.get('p_value', np.nan),
                'Significance': test_r.get('significance', 'unknown'),
                'n_samples': gat_m.get('n_samples', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_results_with_ci(self, y_true: np.ndarray, gat_pred: np.ndarray, 
                                 cnn_pred: np.ndarray, gat_std: np.ndarray, 
                                 cnn_std: np.ndarray, comparison_table: pd.DataFrame):
        """Create comprehensive visualization of results (original plots + CI metrics plot)."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        target_names = ['EC50_LOG_T1', 'EC50_LOG_T2']
        
        # Subplot labels
        subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        
        for i, target in enumerate(target_names):
            # Filter out missing values
            mask = ~np.isnan(y_true[:, i])
            true_vals = y_true[mask, i]
            gat_vals = gat_pred[mask, i]
            cnn_vals = cnn_pred[mask, i]
            
            # Scatter plot: True vs Predicted (original version without error bars)
            ax1 = axes[i, 0]
            ax1.scatter(true_vals, gat_vals, alpha=0.6, label='GAT', color='blue', s=30)
            ax1.scatter(true_vals, cnn_vals, alpha=0.6, label='CNN', color='red', s=30)
            
            # Perfect prediction line
            min_val, max_val = min(true_vals.min(), gat_vals.min(), cnn_vals.min()), \
                            max(true_vals.max(), gat_vals.max(), cnn_vals.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
            
            ax1.set_xlabel(f'True {target}')
            ax1.set_ylabel(f'Predicted {target}')
            ax1.set_title(f'{target}: True vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add subplot label above the plot
            ax1.text(-0.1, 1.02, subplot_labels[i*3], transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='left')
            
            # Residuals plot (original version without error bars)
            ax2 = axes[i, 1]
            gat_residuals = true_vals - gat_vals
            cnn_residuals = true_vals - cnn_vals
            
            ax2.scatter(true_vals, gat_residuals, alpha=0.6, label='GAT', color='blue', s=30)
            ax2.scatter(true_vals, cnn_residuals, alpha=0.6, label='CNN', color='red', s=30)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
            
            ax2.set_xlabel(f'True {target}')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'{target}: Residuals vs True Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add subplot label above the plot
            ax2.text(-0.1, 1.02, subplot_labels[i*3 + 1], transform=ax2.transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='left')
            
            # Distribution of residuals
            ax3 = axes[i, 2]
            ax3.hist(gat_residuals, bins=30, alpha=0.7, label='GAT', color='blue', density=True)
            ax3.hist(cnn_residuals, bins=30, alpha=0.7, label='CNN', color='red', density=True)
            
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Density')
            ax3.set_title(f'{target}: Residual Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add subplot label above the plot
            ax3.text(-0.1, 1.02, subplot_labels[i*3 + 2], transform=ax3.transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='left')
        
        plt.tight_layout()
        plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create metrics comparison plot with confidence intervals
        self._plot_metrics_comparison_with_ci(comparison_table)

    def _plot_metrics_comparison_with_ci(self, comparison_table: pd.DataFrame):
        """Create a metrics comparison bar plot with confidence intervals and p-values."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        targets = comparison_table['Target']
        x_pos = np.arange(len(targets))
        width = 0.35
        
        # Subplot labels
        subplot_labels = ['a)', 'b)', 'c)']
        
        # Extract p-values and create significance indicators
        p_values = comparison_table['p_value'].values
        significance_indicators = []
        for p in p_values:
            if pd.isna(p):
                significance_indicators.append('n.s.')
            elif p < 0.001:
                significance_indicators.append('***')
            elif p < 0.01:
                significance_indicators.append('**')
            elif p < 0.05:
                significance_indicators.append('*')
            else:
                significance_indicators.append('n.s.')
        
        # RMSE comparison with confidence intervals
        ax1 = axes[0]
        gat_rmse = comparison_table['GAT_RMSE']
        cnn_rmse = comparison_table['CNN_RMSE']
        gat_rmse_err = [(gat_rmse - comparison_table['GAT_RMSE_CI_Low']).values,
                       (comparison_table['GAT_RMSE_CI_High'] - gat_rmse).values]
        cnn_rmse_err = [(cnn_rmse - comparison_table['CNN_RMSE_CI_Low']).values,
                       (comparison_table['CNN_RMSE_CI_High'] - cnn_rmse).values]
        
        bars1 = ax1.bar(x_pos - width/2, gat_rmse, width, label='GAT', color='blue', alpha=0.7,
                       yerr=gat_rmse_err, capsize=5)
        bars2 = ax1.bar(x_pos + width/2, cnn_rmse, width, label='CNN', color='red', alpha=0.7,
                       yerr=cnn_rmse_err, capsize=5)
        
        # Add p-values as text annotations
        for i, (p_val, sig) in enumerate(zip(p_values, significance_indicators)):
            max_height = max(gat_rmse.iloc[i], cnn_rmse.iloc[i])
            y_pos = max_height * 1.1
            
            if not pd.isna(p_val):
                ax1.text(x_pos[i], y_pos, f'p={p_val:.3f}\n{sig}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax1.text(x_pos[i], y_pos, 'p=n.a.', 
                        ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Target')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Root Mean Square Error Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(targets, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add subplot label above the plot
        ax1.text(-0.1, 1.02, subplot_labels[0], transform=ax1.transAxes, 
                fontsize=14, fontweight='bold', va='bottom', ha='left')
        
        # R² comparison with confidence intervals
        ax2 = axes[1]
        gat_r2 = comparison_table['GAT_R²']
        cnn_r2 = comparison_table['CNN_R²']
        gat_r2_err = [(gat_r2 - comparison_table['GAT_R²_CI_Low']).values,
                     (comparison_table['GAT_R²_CI_High'] - gat_r2).values]
        cnn_r2_err = [(cnn_r2 - comparison_table['CNN_R²_CI_Low']).values,
                     (comparison_table['CNN_R²_CI_High'] - cnn_r2).values]
        
        bars3 = ax2.bar(x_pos - width/2, gat_r2, width, label='GAT', color='blue', alpha=0.7,
                       yerr=gat_r2_err, capsize=5)
        bars4 = ax2.bar(x_pos + width/2, cnn_r2, width, label='CNN', color='red', alpha=0.7,
                       yerr=cnn_r2_err, capsize=5)
        
        # Add p-values as text annotations
        for i, (p_val, sig) in enumerate(zip(p_values, significance_indicators)):
            max_height = max(gat_r2.iloc[i], cnn_r2.iloc[i])
            y_pos = max_height 
            if i == 1:
                y_pos += 0.25  # Adjust for better visibility of p-value text
            
            if not pd.isna(p_val):
                ax2.text(x_pos[i], y_pos, f'p={p_val:.3f}\n{sig}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax2.text(x_pos[i], y_pos, 'p=n.a.', 
                        ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Target')
        ax2.set_ylabel('R²')
        ax2.set_title('Coefficient of Determination (R²) Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(targets, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add subplot label above the plot
        ax2.text(-0.1, 1.02, subplot_labels[1], transform=ax2.transAxes, 
                fontsize=14, fontweight='bold', va='bottom', ha='left')
        
        # Pearson correlation comparison with confidence intervals
        ax3 = axes[2]
        gat_pearson = comparison_table['GAT_Pearson_r']
        cnn_pearson = comparison_table['CNN_Pearson_r']
        gat_pearson_err = [(gat_pearson - comparison_table['GAT_Pearson_r_CI_Low']).values,
                          (comparison_table['GAT_Pearson_r_CI_High'] - gat_pearson).values]
        cnn_pearson_err = [(cnn_pearson - comparison_table['CNN_Pearson_r_CI_Low']).values,
                          (comparison_table['CNN_Pearson_r_CI_High'] - cnn_pearson).values]
        
        bars5 = ax3.bar(x_pos - width/2, gat_pearson, width, label='GAT', color='blue', alpha=0.7,
                       yerr=gat_pearson_err, capsize=5)
        bars6 = ax3.bar(x_pos + width/2, cnn_pearson, width, label='CNN', color='red', alpha=0.7,
                       yerr=cnn_pearson_err, capsize=5)
        
        # Add p-values as text annotations
        for i, (p_val, sig) in enumerate(zip(p_values, significance_indicators)):
            max_height = max(gat_pearson.iloc[i], cnn_pearson.iloc[i])
            y_pos = max_height + 0.05
            
            if not pd.isna(p_val):
                ax3.text(x_pos[i], y_pos, f'p={p_val:.3f}\n{sig}', 
                        ha='center', va='center', fontsize=9, fontweight='bold')
            else:
                ax3.text(x_pos[i], y_pos, 'p=n.a.', 
                        ha='center', va='center', fontsize=9)
        
        ax3.set_xlabel('Target')
        ax3.set_ylabel('Pearson r')
        ax3.set_title('Pearson Correlation Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(targets, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add subplot label above the plot
        ax3.text(-0.1, 1.02, subplot_labels[2], transform=ax3.transAxes, 
                fontsize=14, fontweight='bold', va='bottom', ha='left')
        
        # Add overall legend for significance indicators
        fig.text(0.02, 0.95, 'Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant\nError bars show 95% confidence intervals', 
                fontsize=10, ha='left', va='top')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)  # Make room for significance legend
        plt.savefig('metrics_comparison_bars_with_ci.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_comparison(self):
        """Run the complete model comparison pipeline with confidence intervals."""
        print("\n🚀 STARTING COMPREHENSIVE MODEL COMPARISON WITH CONFIDENCE INTERVALS")
        print("=" * 80)
        
        # 1. Load data
        self.load_data()
        
        # 2. Load GAT models
        self.load_gat_models()
        
        # 3. Verify CNN model directory exists
        if not self.cnn_model_dir.exists():
            raise ValueError(f"CNN model directory not found: {self.cnn_model_dir}")
        
        if len(self.gat_models) == 0:
            raise ValueError("Failed to load GAT models. Check GAT model directory.")
        
        # 4. Filter data for consistency (both models need same samples)
        valid_mask = pd.notna(self.data['aligned_sequence']) & (self.data['aligned_sequence'].str.strip() != '')
        filtered_data = self.data[valid_mask].copy()

        # Prepare input data using filtered dataset
        gat_input = self.prepare_gat_input(filtered_data['sequence'].tolist())
        cnn_input = self.prepare_cnn_input(filtered_data['aligned_sequence'].tolist())

        # 5. Generate predictions with uncertainty
        gat_predictions, gat_uncertainty = self.predict_gat_ensemble(gat_input)
        cnn_predictions, cnn_uncertainty = self.predict_cnn_ensemble(cnn_input)

        # 6. Prepare true values (use filtered data to match predictions)
        y_true = filtered_data[['EC50_LOG_T1', 'EC50_LOG_T2']].values
        
        # 7. Compute metrics with confidence intervals
        target_names = ['EC50_LOG_T1', 'EC50_LOG_T2']
        gat_metrics = self.compute_metrics_with_ci(y_true, gat_predictions, target_names)
        cnn_metrics = self.compute_metrics_with_ci(y_true, cnn_predictions, target_names)
        
        # 8. Statistical tests
        test_results = self.perform_statistical_tests(y_true, gat_predictions, cnn_predictions, target_names)
        
        # 9. Create comparison table with confidence intervals
        comparison_table = self.create_comparison_table_with_ci(gat_metrics, cnn_metrics, test_results)
        
        # 10. Print results
        self.print_results_with_ci(comparison_table, gat_metrics, cnn_metrics, test_results)
        
        # 11. Visualize results with confidence intervals
        self.visualize_results_with_ci(y_true, gat_predictions, cnn_predictions, 
                                      gat_uncertainty, cnn_uncertainty, comparison_table)
        
        # Store results
        self.results = {
            'gat_metrics': gat_metrics,
            'cnn_metrics': cnn_metrics,
            'test_results': test_results,
            'comparison_table': comparison_table,
            'gat_predictions': gat_predictions,
            'cnn_predictions': cnn_predictions,
            'gat_uncertainty': gat_uncertainty,
            'cnn_uncertainty': cnn_uncertainty,
            'y_true': y_true
        }
        
        print("\n✅ Comparison with confidence intervals completed successfully!")
        return self.results
    
    def print_results_with_ci(self, comparison_table: pd.DataFrame, 
                             gat_metrics: Dict, cnn_metrics: Dict, test_results: Dict):
        """Print comprehensive results summary with confidence intervals."""
        
        print("\n📊 COMPREHENSIVE MODEL COMPARISON RESULTS WITH CONFIDENCE INTERVALS")
        print("=" * 90)
        
        # Summary table (showing only key metrics with CI for readability)
        print("\n📋 PERFORMANCE COMPARISON TABLE")
        print("-" * 90)
        summary_table = comparison_table[['Target', 'GAT_RMSE', 'GAT_RMSE_CI_Low', 'GAT_RMSE_CI_High',
                                        'CNN_RMSE', 'CNN_RMSE_CI_Low', 'CNN_RMSE_CI_High',
                                        'GAT_R²', 'GAT_R²_CI_Low', 'GAT_R²_CI_High',
                                        'CNN_R²', 'CNN_R²_CI_Low', 'CNN_R²_CI_High',
                                        'p_value', 'Significance']].copy()
        
        # Format for display
        for col in ['GAT_RMSE', 'GAT_RMSE_CI_Low', 'GAT_RMSE_CI_High',
                   'CNN_RMSE', 'CNN_RMSE_CI_Low', 'CNN_RMSE_CI_High',
                   'GAT_R²', 'GAT_R²_CI_Low', 'GAT_R²_CI_High',
                   'CNN_R²', 'CNN_R²_CI_Low', 'CNN_R²_CI_High', 'p_value']:
            summary_table[col] = summary_table[col].round(4)
        
        print(summary_table.to_string(index=False))
        
        # Detailed metrics with confidence intervals
        print("\n📈 DETAILED PERFORMANCE METRICS WITH 95% CONFIDENCE INTERVALS")
        print("-" * 70)
        
        for target in ['EC50_LOG_T1', 'EC50_LOG_T2']:
            print(f"\n🎯 {target}:")
            print(f"{'Metric':<15} {'GAT (95% CI)':<25} {'CNN (95% CI)':<25} {'Better':<10}")
            print("-" * 70)
            
            gat_m = gat_metrics.get(target, {})
            cnn_m = cnn_metrics.get(target, {})
            
            # RMSE (lower is better)
            gat_rmse = gat_m.get('rmse', np.nan)
            gat_rmse_ci = gat_m.get('rmse_ci', (np.nan, np.nan))
            cnn_rmse = cnn_m.get('rmse', np.nan)
            cnn_rmse_ci = cnn_m.get('rmse_ci', (np.nan, np.nan))
            rmse_better = 'GAT' if gat_rmse < cnn_rmse else 'CNN'
            
            print(f"{'RMSE':<15} {gat_rmse:.3f} [{gat_rmse_ci[0]:.3f}, {gat_rmse_ci[1]:.3f}]" + 
                  f" {cnn_rmse:.3f} [{cnn_rmse_ci[0]:.3f}, {cnn_rmse_ci[1]:.3f}]".rjust(25) +
                  f" {rmse_better:<10}")
            
            # R² (higher is better)
            gat_r2 = gat_m.get('r2', np.nan)
            gat_r2_ci = gat_m.get('r2_ci', (np.nan, np.nan))
            cnn_r2 = cnn_m.get('r2', np.nan)
            cnn_r2_ci = cnn_m.get('r2_ci', (np.nan, np.nan))
            r2_better = 'GAT' if gat_r2 > cnn_r2 else 'CNN'
            
            print(f"{'R²':<15} {gat_r2:.3f} [{gat_r2_ci[0]:.3f}, {gat_r2_ci[1]:.3f}]" + 
                  f" {cnn_r2:.3f} [{cnn_r2_ci[0]:.3f}, {cnn_r2_ci[1]:.3f}]".rjust(25) +
                  f" {r2_better:<10}")
            
            # Pearson r (higher is better)
            gat_pearson = gat_m.get('pearson_r', np.nan)
            gat_pearson_ci = gat_m.get('pearson_r_ci', (np.nan, np.nan))
            cnn_pearson = cnn_m.get('pearson_r', np.nan)
            cnn_pearson_ci = cnn_m.get('pearson_r_ci', (np.nan, np.nan))
            pearson_better = 'GAT' if gat_pearson > cnn_pearson else 'CNN'
            
            print(f"{'Pearson r':<15} {gat_pearson:.3f} [{gat_pearson_ci[0]:.3f}, {gat_pearson_ci[1]:.3f}]" + 
                  f" {cnn_pearson:.3f} [{cnn_pearson_ci[0]:.3f}, {cnn_pearson_ci[1]:.3f}]".rjust(25) +
                  f" {pearson_better:<10}")
            
            # Statistical significance
            test_r = test_results.get(target, {})
            p_value = test_r.get('p_value', np.nan)
            significance = test_r.get('significance', 'unknown')
            
            print(f"\n📊 Statistical Test Results:")
            print(f"   p-value: {p_value:.4f}")
            print(f"   Result: {significance}")
            if p_value < 0.05:
                print(f"   ⭐ Significant difference detected!")
            else:
                print(f"   ➖ No significant difference")
        
        # Overall summary
        print(f"\n🏆 OVERALL SUMMARY")
        print("-" * 30)
        
        # Count wins
        gat_wins = 0
        cnn_wins = 0
        
        for target in ['EC50_LOG_T1', 'EC50_LOG_T2']:
            gat_m = gat_metrics.get(target, {})
            cnn_m = cnn_metrics.get(target, {})
            
            # Check RMSE (lower is better)
            if gat_m.get('rmse', float('inf')) < cnn_m.get('rmse', float('inf')):
                gat_wins += 1
            else:
                cnn_wins += 1
        
        if gat_wins > cnn_wins:
            print("🥇 GAT ensemble performs better overall")
        elif cnn_wins > gat_wins:
            print("🥇 CNN ensemble performs better overall")
        else:
            print("🤝 Both models perform similarly overall")
        
        print(f"\nGAT wins: {gat_wins}/{len(['EC50_LOG_T1', 'EC50_LOG_T2'])}")
        print(f"CNN wins: {cnn_wins}/{len(['EC50_LOG_T1', 'EC50_LOG_T2'])}")
        
        print(f"\n📊 Bootstrap confidence intervals based on {self.n_bootstrap} samples")
        print(f"🎯 Confidence level: {self.confidence_level*100}%")


def main():
    """Main function to run the complete comparison with confidence intervals."""
    print("🧬 Peptide Activity Regression Model Comparison with Confidence Intervals")
    print("=" * 80)
    print("Comparing GAT-based vs CNN-based ensemble models")
    print("Targets: EC50_Log_T1 and EC50_Log_T2")
    print("Enhanced with bootstrap confidence intervals and uncertainty quantification")
    
    # Initialize comparison with confidence intervals
    comparator = PeptideModelComparison(
        data_file="reference_data.csv",
        gat_model_dir="ensemble_regression_results_GAT",
        cnn_model_dir="peptide_models",
        n_bootstrap=1000,  # Number of bootstrap samples
        confidence_level=0.95  # 95% confidence intervals
    )
    
    try:
        # Run complete comparison with confidence intervals
        results = comparator.run_complete_comparison()
        
        # Save results to file
        comparison_table = results['comparison_table']
        comparison_table.to_csv('model_comparison_results_with_ci.csv', index=False)
        print(f"\n💾 Results saved to 'model_comparison_results_with_ci.csv'")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        raise


if __name__ == "__main__":
    results = main()