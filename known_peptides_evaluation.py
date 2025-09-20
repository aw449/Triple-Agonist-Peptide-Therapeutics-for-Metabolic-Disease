"""
GAT Model Evaluation for Known Peptides
Evaluates trained GAT models on known peptides with qualitative activity labels (high/low).
Generates a comparison table of known vs predicted activities.
"""

import pandas as pd
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.table import Table

warnings.filterwarnings('ignore')

# Import existing components from the GAT validation script
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
            print(f"⚠ Warning: Could not compute normalization stats: {e}")
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


def load_known_peptides(file_path: str) -> pd.DataFrame:
    """Load known peptides data."""
    print(f"Loading known peptides from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} known peptides")
        
        # Display column names to verify format
        print(f"Columns found: {list(df.columns)}")
        
        # Check for expected columns
        expected_cols = ['name', 'sequence', 'descriptor', 'GCGR_known', 'GLP1R_known', 'GIPR_known']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠ Warning: Missing expected columns: {missing_cols}")
            print("Attempting to map from available columns...")
            
            # Try to map common variations
            column_mappings = {
                'Drug Name': 'name',
                'Sequence': 'sequence', 
                'Description': 'descriptor',
                'GCGR_known': 'GCGR_known',
                'GLP1R_known': 'GLP1R_known',
                'GIPR_known': 'GIPR_known'
            }
            
            df_renamed = df.copy()
            for old_col, new_col in column_mappings.items():
                if old_col in df.columns:
                    df_renamed = df_renamed.rename(columns={old_col: new_col})
            
            df = df_renamed
            print(f"Mapped columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading known peptides: {e}")
        raise


def convert_known_activity_to_binary(activity_str: str) -> int:
    """Convert known activity string to binary (1=high, 0=low, -1=unknown)."""
    if pd.isna(activity_str) or activity_str == '':
        return -1
    
    activity_str = str(activity_str).lower().strip()
    
    if activity_str in ['high', 'h', '1', 'active', 'strong']:
        return 1
    elif activity_str in ['low', 'l', '0', 'inactive', 'weak']:
        return 0
    else:
        return -1  # Unknown/unclear


def evaluate_known_peptides(predictions: Dict[str, np.ndarray], 
                          known_df: pd.DataFrame,
                          probability_threshold: float = 0.5) -> pd.DataFrame:
    """
    Evaluate predictions against known activities and create comparison table.
    
    Args:
        predictions: Dictionary with receptor predictions
        known_df: DataFrame with known peptide activities
        probability_threshold: Threshold to convert probabilities to binary predictions
        
    Returns:
        DataFrame with comparison results
    """
    results_data = []
    
    for i, row in known_df.iterrows():
        peptide_name = row.get('name', f'Peptide_{i}')
        sequence = row.get('sequence', '')
        descriptor = row.get('descriptor', '')
        
        result_row = {
            'peptide_name': peptide_name,
            'sequence': sequence,
            'descriptor': descriptor
        }
        
        # Process each receptor
        for receptor in ['GCGR', 'GLP1R', 'GIPR']:
            # Get known activity
            known_col = f'{receptor}_known'
            known_activity_str = row.get(known_col, '')
            known_binary = convert_known_activity_to_binary(known_activity_str)
            
            # Get predicted activity
            if receptor in predictions and i < len(predictions[receptor]):
                pred_prob = predictions[receptor][i]
                pred_binary = 1 if pred_prob > probability_threshold else 0
                
                # Determine agreement
                if known_binary == -1:
                    agreement = 'Unknown'
                elif known_binary == pred_binary:
                    agreement = 'Correct'
                else:
                    agreement = 'Incorrect'
                    
            else:
                pred_prob = np.nan
                pred_binary = -1
                agreement = 'No_Prediction'
            
            # Add to result row
            result_row.update({
                f'{receptor}_known_str': known_activity_str,
                f'{receptor}_known_binary': known_binary,
                f'{receptor}_pred_prob': pred_prob,
                f'{receptor}_pred_binary': pred_binary,
                f'{receptor}_agreement': agreement
            })
        
        results_data.append(result_row)
    
    return pd.DataFrame(results_data)


def create_publication_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean publication-ready table."""
    # Start with peptide names
    pub_table = pd.DataFrame({
        'Peptide': results_df['peptide_name']
    })
    
    # Add actual and predicted columns for each receptor
    for receptor in ['GCGR', 'GLP1R', 'GIPR']:
        # Convert binary to High/Low, handle unknowns
        actual_col = []
        predicted_col = []
        
        for _, row in results_df.iterrows():
            # Actual activity
            known_binary = row[f'{receptor}_known_binary']
            if known_binary == 1:
                actual_col.append('High')
            elif known_binary == 0:
                actual_col.append('Low')
            else:
                actual_col.append('Unknown')
            
            # Predicted activity
            pred_binary = row[f'{receptor}_pred_binary']
            if pred_binary == 1:
                predicted_col.append('High')
            elif pred_binary == 0:
                predicted_col.append('Low')
            else:
                predicted_col.append('N/A')
        
        pub_table[f'{receptor} Actual'] = actual_col
        pub_table[f'{receptor} Predicted'] = predicted_col
    
    return pub_table


def calculate_receptor_accuracies(results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate accuracy for each receptor."""
    accuracies = {}
    
    for receptor in ['GCGR', 'GLP1R', 'GIPR']:
        # Filter out unknown/missing values
        valid_mask = (results_df[f'{receptor}_known_binary'] != -1) & \
                     (~pd.isna(results_df[f'{receptor}_pred_prob']))
        
        if valid_mask.sum() == 0:
            accuracies[receptor] = 0.0
            continue
            
        valid_df = results_df[valid_mask]
        correct = (valid_df[f'{receptor}_agreement'] == 'Correct').sum()
        total_valid = len(valid_df)
        accuracies[receptor] = correct / total_valid if total_valid > 0 else 0.0
    
    return accuracies


def create_publication_table_image(pub_table_df: pd.DataFrame, results_df: pd.DataFrame, filename: str = 'known_peptides_table.png'):
    """Create a publication-ready PNG image of the table with disagreement highlighting."""
    
    # Set up the figure with appropriate size
    n_rows, n_cols = pub_table_df.shape
    fig_width = max(12, n_cols * 1.5)
    fig_height = max(8, n_rows * 0.4 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table_data = []
    
    # Add header row
    headers = list(pub_table_df.columns)
    table_data.append(headers)
    
    # Add data rows
    for _, row in pub_table_df.iterrows():
        table_data.append(list(row.values))
    
    # Create table with matplotlib
    table = ax.table(cellText=table_data[1:],  # Data rows
                    colLabels=table_data[0],   # Header row
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Define colors for each receptor
    receptor_colors = {
        'GCGR': {'high': '#2E8B57', 'low': '#DC143C'},      # Sea Green / Crimson
        'GLP1R': {'high': '#4169E1', 'low': '#FF6347'},     # Royal Blue / Tomato
        'GIPR': {'high': '#9932CC', 'low': '#FF8C00'}       # Dark Orchid / Dark Orange
    }
    
    # Style header row
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#2F4F4F')  # Dark slate gray
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_height(0.08)
    
    # Style data rows
    for i in range(1, n_rows + 1):
        peptide_idx = i - 1  # Index for results_df
        
        for j in range(n_cols):
            cell = table[(i, j)]
            col_name = headers[j]
            
            # Alternate row colors for better readability
            if i % 2 == 0:
                base_color = '#F8F8F8'
            else:
                base_color = 'white'
            
            cell.set_facecolor(base_color)
            
            # Highlight accuracy row
            if i == n_rows and 'Overall Accuracy' in str(cell.get_text().get_text()):
                cell.set_facecolor('#E6E6FA')  # Lavender
                cell.set_text_props(weight='bold', fontsize=11)
                cell.set_height(0.06)

                continue
            
            # Skip peptide name column for styling
            if j == 0:
                cell.set_text_props(fontsize=10, weight='bold')
                cell.set_height(0.06)
                continue
            
            # Color-code receptor columns and highlight disagreements
            cell_text = str(cell.get_text().get_text())
            
            # Determine which receptor this column belongs to
            receptor = None
            is_actual = False
            is_predicted = False
            
            if 'GCGR' in col_name:
                receptor = 'GCGR'
            elif 'GLP1R' in col_name:
                receptor = 'GLP1R'
            elif 'GIPR' in col_name:
                receptor = 'GIPR'
            
            if receptor:
                is_actual = 'Actual' in col_name
                is_predicted = 'Predicted' in col_name
                
                # Get the corresponding actual and predicted values for disagreement check
                if peptide_idx < len(results_df):
                    actual_binary = results_df.iloc[peptide_idx][f'{receptor}_known_binary']
                    pred_binary = results_df.iloc[peptide_idx][f'{receptor}_pred_binary']
                    
                    # Check for disagreement (only if both are valid)
                    has_disagreement = False
                    if actual_binary != -1 and pred_binary != -1:
                        has_disagreement = (actual_binary != pred_binary)
                
                # Apply colors based on High/Low and receptor
                if cell_text == 'High':
                    color = receptor_colors[receptor]['high']
                    cell.set_text_props(color=color, weight='bold', fontsize=10)
                elif cell_text == 'Low':
                    color = receptor_colors[receptor]['low']
                    cell.set_text_props(color=color, weight='bold', fontsize=10)
                elif cell_text == 'Unknown':
                    cell.set_text_props(color='#808080', style='italic', fontsize=10)
                elif cell_text == 'N/A':
                    cell.set_text_props(color='#A0A0A0', fontsize=9)
                else:
                    cell.set_text_props(fontsize=10)
                
                # Highlight cells with disagreement
                if has_disagreement and (is_actual or is_predicted):
                    cell.set_facecolor('#FFE4E1')  # Light pink for disagreement
                    # Add a border for disagreement
                    cell.set_edgecolor('#FF0000')
                    cell.set_linewidth(2)
            
            cell.set_height(0.06)
    
    # Add thick vertical lines between receptor pairs
    # Get table position and size
    table_bbox = table.get_window_extent(fig.canvas.get_renderer())
    table_bbox = table_bbox.transformed(ax.transData.inverted())
    
    # Calculate positions for vertical lines
    # Between GCGR Predicted (col 2) and GLP1R Actual (col 3)
    x1 = table_bbox.x0 + (table_bbox.width / n_cols) * 3
    # Between GLP1R Predicted (col 4) and GIPR Actual (col 5) 
    x2 = table_bbox.x0 + (table_bbox.width / n_cols) * 5
    
    y_top = table_bbox.y1
    y_bottom = table_bbox.y0
    
    # Draw thick vertical lines
    ax.plot([x1, x1], [y_bottom, y_top], color='black', linewidth=4, clip_on=False)
    ax.plot([x2, x2], [y_bottom, y_top], color='black', linewidth=4, clip_on=False)
    
    # Add title
    plt.title('GAT Model Predictions vs Known Peptide Activities', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = []
    for receptor, colors in receptor_colors.items():
        legend_elements.extend([
            plt.Line2D([0], [0], color=colors['high'], lw=4, label=f'{receptor} High'),
            plt.Line2D([0], [0], color=colors['low'], lw=4, label=f'{receptor} Low')
        ])
    
    # Add disagreement indicator
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFE4E1', edgecolor='#FF0000', 
                     linewidth=2, label='Disagreement')
    )
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=10)
    
    # Save with high DPI for publication quality
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Publication table saved as: {filename}")
    
    # Also show the plot
    plt.show()
    
    return filename


def main():
    """Main evaluation pipeline for known peptides."""
    print("🧬 GAT Model Evaluation on Known Peptides")
    print("="*60)
    
    # Configuration
    known_peptides_file = 'known_peptides.csv'
    model_dir = 'kfold_transfer_learning_results_GAT'
    probability_threshold = 0.5
    
    # 1. Load known peptides
    known_df = load_known_peptides(known_peptides_file)
    
    # 2. Load GAT ensemble
    print("\nLoading GAT ensemble...")
    predictor = GATEnsemblePredictor(model_dir, 'all_sequences_aligned.csv')
    
    # 3. Make predictions
    print(f"\nMaking predictions for {len(known_df)} known peptides...")
    sequences = known_df['sequence'].tolist()
    predictions = predictor.predict_sequences(sequences)
    
    # 4. Evaluate against known activities
    print("\nEvaluating predictions against known activities...")
    results_df = evaluate_known_peptides(predictions, known_df, probability_threshold)
    
    # 5. Create publication-ready table
    print("\nCreating publication-ready table...")
    pub_table = create_publication_table(results_df)
    
    # Calculate accuracies for each receptor
    accuracies = calculate_receptor_accuracies(results_df)
    
    # Add accuracy row at the bottom
    accuracy_row = {'Peptide': 'Overall Accuracy'}
    for receptor in ['GCGR', 'GLP1R', 'GIPR']:
        accuracy_row[f'{receptor} Actual'] = ""
        accuracy_row[f'{receptor} Predicted'] = f"{accuracies[receptor]:.3f}"
    
    # Convert to DataFrame and append accuracy row
    accuracy_df = pd.DataFrame([accuracy_row])
    pub_table_with_accuracy = pd.concat([pub_table, accuracy_df], ignore_index=True)
    
    # 6. Create publication-ready PNG image
    print("\nCreating publication-ready table image...")
    image_filename = create_publication_table_image(pub_table_with_accuracy, results_df)
    
    # 7. Display results
    print("\n" + "="*80)
    print("PUBLICATION-READY EVALUATION TABLE")
    print("="*80)
    
    print(pub_table_with_accuracy.to_string(index=False))
    
    # 8. Save results
    pub_table_file = 'known_peptides_publication_table.csv'
    detailed_results_file = 'known_peptides_detailed_results.csv'
    
    pub_table_with_accuracy.to_csv(pub_table_file, index=False)
    results_df.to_csv(detailed_results_file, index=False)
    
    print(f"\n✅ Results saved:")
    print(f"  Publication table image: {image_filename}")
    print(f"  Publication table CSV: {pub_table_file}")
    print(f"  Detailed results: {detailed_results_file}")
    
    # 9. Print summary statistics
    print(f"\n📊 ACCURACY SUMMARY:")
    for receptor, accuracy in accuracies.items():
        valid_count = ((results_df[f'{receptor}_known_binary'] != -1) & 
                      (~pd.isna(results_df[f'{receptor}_pred_prob']))).sum()
        print(f"  {receptor}: {accuracy:.3f} ({valid_count} peptides evaluated)")
    
    overall_accuracy = np.mean(list(accuracies.values()))
    print(f"  Overall: {overall_accuracy:.3f}")
    
    return pub_table_with_accuracy, image_filename


if __name__ == "__main__":
    main()