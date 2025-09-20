# Multi-Target Peptide Design Using Graph Neural Networks and Genetic Algorithms

A comprehensive machine learning framework for designing peptides with enhanced biological activity across multiple targets using Graph Attention Networks (GATs), Convolutional Neural Networks (CNNs), and Genetic Algorithm optimization.

## Project Overview

This project implements state-of-the-art machine learning approaches for multi-target peptide design, specifically focusing on GCGR, GLP-1R, and GIPR receptor targeting. The pipeline combines advanced neural network architectures with evolutionary optimization to discover novel peptide sequences with enhanced biological potency.

## Key Features

- **Multi-Task Graph Neural Networks**: Graph Attention Networks for capturing molecular topology and spatial relationships
- **Comparative Analysis**: Systematic comparison between GNN and CNN approaches for peptide activity prediction
- **Genetic Algorithm Optimization**: Evolutionary sequence generation guided by machine learning fitness functions
- **Multi-Target Design**: Simultaneous optimization for multiple biological targets
- **Comprehensive Analysis**: In-depth peptide property analysis and visualization

## Project Workflow

### 1. GNN Training with `multi-task-GNN.py`

**Purpose**: Train Graph Attention Networks for multi-target peptide activity prediction

**Key Components**:
- Graph representation of peptide molecular structure
- Multi-head attention mechanisms for capturing complex interactions
- Multi-task learning framework for simultaneous target optimization
- Transfer learning capabilities from pre-trained molecular models

**Input**: 
- Training sequences with experimentally measured EC50 values
- Molecular graph representations of peptide structures

**Output**:
- Trained GAT ensemble models
- Performance metrics and validation results
- Model checkpoints for downstream applications

### 2. GNN vs CNN Comparison with `peptide_comparison.py`

**Purpose**: Systematic comparison of Graph Neural Networks and Convolutional Neural Networks for peptide activity prediction

**Analysis Includes**:
- Performance metrics comparison (RMSE, MAE, R²)
- Statistical significance testing
- Prediction accuracy across different target types
- Computational efficiency analysis
- Visualization of prediction quality

**Key Findings**:
- Graph-based methods excel with structural information
- CNN approaches provide computational efficiency
- Hybrid approaches show superior performance

### 3. Genetic Algorithm Generation

**Purpose**: Evolutionary optimization of peptide sequences using ML-guided fitness functions

**GA Framework**:
- **Population Management**: Adaptive population sizing and diversity maintenance
- **Selection Strategies**: Tournament selection with elitism
- **Crossover Operations**: Single-point and multi-point crossover methods
- **Mutation Operators**: Adaptive mutation rates with amino acid substitution matrices
- **Fitness Evaluation**: ML model-guided activity prediction

**Optimization Targets**:
- High activity against multiple receptors
- Selective activity profiles
- Improved bioavailability properties
- Enhanced stability characteristics

### 4. Peptide Analysis

**Comprehensive Analysis Pipeline**:

**Sequence Analysis**:
- Amino acid composition and properties
- Hydrophobicity profiles and secondary structure prediction
- Similarity analysis and clustering
- Conservation pattern identification

**Activity Prediction**:
- Multi-target EC50 prediction
- Confidence interval estimation
- Activity profile visualization
- Structure-activity relationship analysis

**Bioavailability Assessment**:
- Molecular weight and charge distribution
- Lipophilicity and membrane permeability
- Metabolic stability prediction

## Dataset Description

### Training Data (`training_data.csv`)
- **125 peptide sequences** with experimental validation
- **EC50 measurements** for targets T1 and T2
- **Sequence properties**: Length, composition, alignment information

### Validation Set (`validation_sequences.csv`) 
- **24 independent sequences** for model validation
- **Multi-target activity data** (T1, T2, T3)
- **Unbiased evaluation** of model performance

### Complete Dataset (`all_sequences_aligned.csv`)
- **234 total sequences** with aligned representations
- **Comprehensive activity profiles** across three targets
- **Ready for machine learning** with preprocessed features

## Installation and Setup

```bash
# Clone repository
git clone <repository-url>
cd peptide-design-project

# Install dependencies
pip install -r requirements.txt

# Set up environment
conda env create -f environment.yml
conda activate peptide-design
```

## Usage

### 1. Train Multi-Task GNN Models

```bash
python multi-task-GNN.py --config config/gnn_config.yaml
```

**Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--hidden_dim`: Hidden layer dimensions (default: 128)

### 2. Compare GNN vs CNN Performance

```bash
python peptide_comparison.py --gat_models kfold_transfer_learning_results_GAT --cnn_models kfold_transfer_learning_results_CNN
```

**Outputs**:
- Comparative performance metrics
- Statistical significance tests
- Visualization plots
- Model ranking and recommendations

### 3. Run Genetic Algorithm Optimization

```bash
python peptide_genetic_algorithm.py --config config/ga_config.yaml --generations 50
```

**GA Configuration**:
- Population size: 100
- Crossover rate: 0.8
- Mutation rate: 0.25
- Selection strategy: Tournament selection

### 4. Analyze Generated Peptides

```bash
python peptide_analysis.py --input generated_sequences.csv --models trained_models/
```

## Results and Performance

### Model Performance
- **GAT Models**: Superior performance with structural information (R² > 0.85)
- **CNN Models**: Efficient training with sequence data (R² > 0.80)
- **Ensemble Approach**: Best overall performance combining both methods

### Generated Peptides
- **Novel Sequences**: Computationally designed peptides with predicted high activity
- **Multi-Target Profile**: Optimized for simultaneous activity across multiple receptors
- **Experimental Validation**: Selected candidates for wet-lab validation

### Computational Efficiency
- **Training Time**: GAT ~2-3 hours, CNN ~30-45 minutes
- **Prediction Speed**: Real-time inference for large sequence libraries
- **Resource Requirements**: GPU recommended for training, CPU sufficient for prediction

## Key Dependencies

```
torch>=1.9.0
torch-geometric>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
biopython>=1.79
rdkit>=2021.09.1
```

## File Structure

```
peptide-design-project/
├── multi-task-GNN.py              # Graph Neural Network training
├── peptide_comparison.py          # GNN vs CNN comparison
├── peptide_genetic_algorithm.py   # Genetic algorithm optimization
├── peptide_analysis.py            # Comprehensive analysis pipeline
├── training_data.csv          # Training sequences
├── validation_sequences.csv   # Validation set
└── all_sequences_aligned.csv  # Complete dataset
├── models/                        # Trained model checkpoints
├── results/                       # Output results and plots
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and collaborations, please contact awong16@illinois.edu.

---

**Keywords**: Peptide Design, Graph Neural Networks, Multi-Task Learning, Genetic Algorithms, Drug Discovery, GCGR, GLP-1R, Machine Learning
