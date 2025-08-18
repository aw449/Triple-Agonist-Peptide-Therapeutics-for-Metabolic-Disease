#!/usr/bin/env python3
"""
Main Execution Script for Peptide Genetic Algorithm
Complete pipeline for evolving novel peptide sequences with high binding affinity.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import all our modules
from peptide_sequence_handler import PeptideSequenceHandler
from gat_model_wrapper import GATModelWrapper
from genetic_operations import Individual, SelectionOperator, CrossoverOperator, MutationOperator
from peptide_genetic_algorithm import PeptideGeneticAlgorithm, GAConfig
from ga_visualization import GAVisualizer


def create_default_config() -> GAConfig:
    """Create default GA configuration with optimized parameters."""
    return GAConfig(
        population_size=100,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_rate=0.1,
        selection_strategy="tournament",
        crossover_method="single_point",
        mutation_method="adaptive",
        tournament_size=3,
        target_threshold=0,
        convergence_threshold=0.1,  # Stop when improvement < 0.1 fitness/generation
        convergence_window=3,       # Evaluate over 3 generations
        min_generations_for_convergence=3,
        min_sequence_length=25,
        max_sequence_length=40,
        diversity_threshold=0.1,
        stagnation_limit=10,
        parallel_evaluation=True,
        num_processes=None,
        random_seed=50
    )


def create_fast_config() -> GAConfig:
    """Create configuration for fast testing/debugging."""
    return GAConfig(
        population_size=30,
        max_generations=15,
        crossover_rate=0.7,
        mutation_rate=0.3,
        elitism_rate=0.15,
        selection_strategy="tournament",
        crossover_method="uniform",
        mutation_method="adaptive",
        tournament_size=2,
        target_threshold=0,
        convergence_threshold=0.1,  # Stop when improvement < 0.1 fitness/generation
        convergence_window=3,       # Evaluate over 5 generations
        min_generations_for_convergence=3,
        min_sequence_length=25,
        max_sequence_length=40,
        diversity_threshold=0.15,
        stagnation_limit=8,
        parallel_evaluation=False,
        num_processes=1,
        random_seed=42
    )


def create_intensive_config() -> GAConfig:
    """Create configuration for intensive optimization."""
    return GAConfig(
        population_size=200,
        max_generations=5,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_rate=0.05,
        selection_strategy="tournament",
        crossover_method="two_point",
        mutation_method="adaptive",
        tournament_size=4,
        target_threshold=0,
        convergence_threshold= 0.1,  # Stop when improvement < 0.1 fitness/generation
        convergence_window= 5,       # Evaluate over 3 generations
        min_generations_for_convergence= 5,
        min_sequence_length=25,
        max_sequence_length=40,
        diversity_threshold=0.08,
        stagnation_limit=15,
        parallel_evaluation=True,
        num_processes=None,
        random_seed=42
    )


def validate_environment(model_dir: str, data_file: str) -> bool:
    """
    Validate that all required files and directories exist.
    
    Args:
        model_dir: Directory containing GAT model
        data_file: CSV file with seed sequences
        
    Returns:
        True if environment is valid
    """
    print("🔍 Validating environment...")
    
    # Check model directory
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Check for model files
    model_files = list(model_path.glob("*_model*.pth"))
    if not model_files:
        print(f"❌ No GAT model file found in {model_dir}")
        return False
    
    print(f"✅ Found GAT model: {model_files[0].name}")
    
    # Check data file
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_file}")
        return False
    
    print(f"✅ Found data file: {data_file}")
    
    # Test model loading
    try:
        print("🧪 Testing model loading...")
        model_wrapper = GATModelWrapper(model_dir)
        print("✅ Model loaded successfully")
        
        # Test with a simple sequence
        test_result = model_wrapper.predict_single_sequence("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR")
        print(f"✅ Model prediction test successful: {test_result}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    print("✅ Environment validation completed successfully")
    return True


def run_parameter_sweep(base_config: GAConfig, 
                       data_file: str,
                       model_dir: str,
                       output_dir: str) -> None:
    """
    Run parameter sweep to find optimal GA settings.
    
    Args:
        base_config: Base configuration to modify
        data_file: CSV file with seed sequences
        model_dir: Directory containing GAT model
        output_dir: Output directory for results
    """
    print("\n🔬 Running parameter sweep...")
    
    # Parameter variations to test
    parameter_sets = [
        {"population_size": 50, "mutation_rate": 0.1, "name": "small_pop_low_mut"},
        {"population_size": 50, "mutation_rate": 0.3, "name": "small_pop_high_mut"},
        {"population_size": 150, "mutation_rate": 0.1, "name": "large_pop_low_mut"},
        {"population_size": 150, "mutation_rate": 0.3, "name": "large_pop_high_mut"},
        {"crossover_method": "uniform", "selection_strategy": "roulette", "name": "uniform_roulette"},
        {"crossover_method": "two_point", "selection_strategy": "rank", "name": "two_point_rank"},
    ]
    
    sweep_results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"\n--- Parameter Set {i+1}/{len(parameter_sets)}: {params['name']} ---")
        
        # Create modified config
        config = GAConfig(**{**base_config.__dict__, **{k: v for k, v in params.items() if k != 'name'}})
        config.max_generations = 20  # Shorter runs for sweep
        
        # Run GA
        sweep_output_dir = Path(output_dir) / f"sweep_{params['name']}"
        sweep_output_dir.mkdir(exist_ok=True, parents=True)
        
        ga = PeptideGeneticAlgorithm(config, model_dir, str(sweep_output_dir))
        
        try:
            results = ga.run_from_csv(data_file)
            
            # Extract key metrics
            sweep_result = {
                "parameters": params,
                "final_fitness": results['evolution_summary']['final_best_fitness'],
                "high_affinity_found": results['evolution_summary']['high_affinity_found'],
                "generations": results['evolution_summary']['generations_completed'],
                "time": results['evolution_summary']['total_time']
            }
            sweep_results.append(sweep_result)
            
            print(f"✅ Completed: Final fitness = {sweep_result['final_fitness']:.3f}, "
                  f"High affinity = {sweep_result['high_affinity_found']}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            sweep_results.append({
                "parameters": params,
                "error": str(e)
            })
    
    # Save sweep results
    sweep_file = Path(output_dir) / "parameter_sweep_results.json"
    with open(sweep_file, 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)
    
    print(f"\n📊 Parameter sweep completed. Results saved to {sweep_file}")
    
    # Print summary
    print("\nSWEEP SUMMARY:")
    print("-" * 60)
    for result in sweep_results:
        if 'error' not in result:
            print(f"{result['parameters']['name']:20} | "
                  f"Fitness: {result['final_fitness']:6.3f} | "
                  f"High affinity: {result['high_affinity_found']:2d} | "
                  f"Time: {result['time']:6.1f}s")
        else:
            print(f"{result['parameters']['name']:20} | ERROR: {result['error']}")


def analyze_best_sequences(results: Dict[str, Any]) -> None:
    """
    Perform detailed analysis of the best sequences found.
    
    Args:
        results: GA results dictionary
    """
    print("\n🧬 ANALYZING BEST SEQUENCES")
    print("=" * 60)
    
    best_sequences = results.get('best_sequences', [])
    high_affinity = results.get('high_affinity_candidates', [])
    
    if not best_sequences:
        print("No sequences to analyze")
        return
    
    exact_matches = 0
    for seq_info in best_sequences:
        if seq_info.get('exact_match', False):
            exact_matches += 1
    
    print(f"✅ Novel sequences generated: {len(best_sequences) - exact_matches}/{len(best_sequences)}")
    if exact_matches > 0:
        print(f"⚠️  Exact matches found: {exact_matches} (these should be filtered out)")
    
    # Sequence handler for analysis
    handler = PeptideSequenceHandler()
    
    print(f"\nTOP 10 SEQUENCES:")
    print("-" * 60)
    
    for i, seq_info in enumerate(best_sequences[:10]):
        sequence = seq_info['sequence']
        fitness = seq_info['fitness_score']
        predictions = seq_info.get('predictions', {})
        
        # Calculate properties
        tokens = handler.tokenize_sequence(sequence)
        properties = handler.calculate_properties(tokens)
        
        # NEW: Get plausibility scores if available
        plausibility_scores = handler.evaluate_biological_plausibility(tokens)
        
        print(f"\nRank {i+1}:")
        print(f"  Sequence: {sequence}")
        print(f"  Fitness: {fitness:.3f}")
        print(f"  Length: {properties.length} AA")
        print(f"  Charge: {properties.charge:+.1f}")
        print(f"  Hydrophobicity: {properties.hydrophobicity:.2f}")
        print(f"  Modifications: {properties.modification_count}")
        
        # NEW: Plausibility breakdown
        print(f"  Plausibility: {plausibility_scores['overall']:.3f} "
              f"(Chem: {plausibility_scores['chemical']:.2f}, "
              f"Motifs: {plausibility_scores['motifs']:.2f}, "
              f"Stab: {plausibility_scores['proteolysis']:.2f})")
        
        if predictions:
            print(f"  Binding probabilities:")
            for receptor, prob in predictions.items():
                status = "HIGH" if prob >= 0.5 else "low"
                print(f"    {receptor}: {prob:.3f} ({status})")
    
    if high_affinity:
        print(f"\n🎯 HIGH AFFINITY CANDIDATES (All 3 receptors ≥ 0.5):")
        print("-" * 60)
        
        for i, seq_info in enumerate(high_affinity):
            sequence = seq_info['sequence']
            fitness = seq_info['fitness_score']
            predictions = seq_info['predictions']
            
            tokens = handler.tokenize_sequence(sequence)
            properties = handler.calculate_properties(tokens)
            
            print(f"\nCandidate {i+1}:")
            print(f"  Sequence: {sequence}")
            print(f"  Fitness: {fitness:.3f}")
            print(f"  Length: {properties.length} AA")
            print(f"  Binding: GCGR={predictions['GCGR']:.3f}, "
                  f"GLP1R={predictions['GLP1R']:.3f}, "
                  f"GIPR={predictions['GIPR']:.3f}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("-" * 60)
    
    all_fitness = [seq['fitness_score'] for seq in best_sequences]
    all_lengths = [len(handler.tokenize_sequence(seq['sequence'])) for seq in best_sequences]
    
    print(f"Best sequences analyzed: {len(best_sequences)}")
    print(f"Average fitness: {sum(all_fitness)/len(all_fitness):.3f}")
    print(f"Best fitness: {max(all_fitness):.3f}")
    print(f"Average length: {sum(all_lengths)/len(all_lengths):.1f} AA")
    print(f"High affinity candidates: {len(high_affinity)}")
    
    if high_affinity:
        ha_fitness = [seq['fitness_score'] for seq in high_affinity]
        print(f"High affinity avg fitness: {sum(ha_fitness)/len(ha_fitness):.3f}")


def export_sequences_for_synthesis(results: Dict[str, Any], output_file: str) -> None:
    """
    Export top sequences in format suitable for peptide synthesis.
    
    Args:
        results: GA results dictionary
        output_file: Output file path
    """
    print(f"\n💾 Exporting sequences for synthesis to {output_file}")
    
    high_affinity = results.get('high_affinity_candidates', [])
    best_sequences = results.get('best_sequences', [])
    
    # Combine and deduplicate
    all_candidates = high_affinity.copy()
    
    # Add best sequences that aren't already in high affinity
    ha_sequences = {seq['sequence'] for seq in high_affinity}
    for seq in best_sequences[:20]:
        if seq['sequence'] not in ha_sequences:
            all_candidates.append(seq)
    
    # Sort by fitness
    all_candidates.sort(key=lambda x: x['fitness_score'], reverse=True)
    
    # Export to CSV
    import pandas as pd
    
    export_data = []
    handler = PeptideSequenceHandler()
    
    for i, seq_info in enumerate(all_candidates[:30]):  # Top 30 for synthesis
        sequence = seq_info['sequence']
        tokens = handler.tokenize_sequence(sequence)
        properties = handler.calculate_properties(tokens)
        predictions = seq_info.get('predictions', {})
        
        export_data.append({
            'rank': i + 1,
            'sequence': sequence,
            'fitness_score': seq_info['fitness_score'],
            'length': properties.length,
            'net_charge': properties.charge,
            'hydrophobicity': properties.hydrophobicity,
            'molecular_weight': properties.molecular_weight,
            'modifications': properties.modification_count,
            'GCGR_probability': predictions.get('GCGR', 0),
            'GLP1R_probability': predictions.get('GLP1R', 0),
            'GIPR_probability': predictions.get('GIPR', 0),
            'all_high_affinity': all(prob >= 0.5 for prob in predictions.values()) if predictions else False,
            'generation': seq_info.get('generation', 0),
            'synthesis_priority': 'HIGH' if i < 10 else 'MEDIUM' if i < 20 else 'LOW'
        })
    
    df = pd.DataFrame(export_data)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Exported {len(export_data)} sequences for synthesis")
    print(f"   High priority: {sum(1 for x in export_data if x['synthesis_priority'] == 'HIGH')}")
    print(f"   Medium priority: {sum(1 for x in export_data if x['synthesis_priority'] == 'MEDIUM')}")
    print(f"   Low priority: {sum(1 for x in export_data if x['synthesis_priority'] == 'LOW')}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Peptide Genetic Algorithm for Multi-Receptor Binding Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default parameters
  python main_ga_execution.py --data all_sequences_aligned.csv
  
  # Fast test run
  python main_ga_execution.py --data all_sequences_aligned.csv --config fast
  
  # Intensive optimization
  python main_ga_execution.py --data all_sequences_aligned.csv --config intensive
  
  # Custom parameters
  python main_ga_execution.py --data all_sequences_aligned.csv --population 150 --generations 75
  
  # Parameter sweep
  python main_ga_execution.py --data all_sequences_aligned.csv --parameter-sweep
  
  # Just visualization of existing results
  python main_ga_execution.py --visualize-only results/ga_results_20231201_123456.json
        """
    )
    
    # Main arguments
    parser.add_argument('--data', type=str, default='all_sequences_aligned.csv',
                       help='CSV file containing seed sequences')
    parser.add_argument('--model-dir', type=str, default='kfold_transfer_learning_results_GAT',
                       help='Directory containing trained GAT model')
    parser.add_argument('--output-dir', type=str, default='peptide_ga_results',
                       help='Output directory for results')
    
    # Configuration presets
    parser.add_argument('--config', choices=['default', 'fast', 'intensive'], default='default',
                       help='Configuration preset')
    
    # Custom parameters
    parser.add_argument('--population', type=int, help='Population size')
    parser.add_argument('--generations', type=int, help='Maximum generations')
    parser.add_argument('--mutation-rate', type=float, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, help='Crossover rate')
    parser.add_argument('--convergence-threshold', type=float, help='Convergence threshold (improvement rate/generation)')
    parser.add_argument('--convergence-window', type=int, help='Number of generations for convergence evaluation')
    parser.add_argument('--min-convergence-generations', type=int, help='Minimum generations before checking convergence')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    # Analysis options
    parser.add_argument('--parameter-sweep', action='store_true',
                       help='Run parameter sweep instead of single optimization')
    parser.add_argument('--visualize-only', type=str,
                       help='Only create visualizations from existing results file')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--export-synthesis', action='store_true',
                       help='Export sequences for synthesis')
    
    # System options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel evaluation')
    parser.add_argument('--num-processes', type=int,
                       help='Number of processes for parallel evaluation')
    
    args = parser.parse_args()
    
    # Handle visualization-only mode
    if args.visualize_only:
        print(f"🎨 Creating visualizations for {args.visualize_only}")
        visualizer = GAVisualizer(args.output_dir)
        
        with open(args.visualize_only, 'r') as f:
            results = json.load(f)
        
        visualizer.create_comprehensive_report(results)
        return
    
    # Validate environment
    if not validate_environment(args.model_dir, args.data):
        print("❌ Environment validation failed. Please check your setup.")
        sys.exit(1)
    
    # Create configuration
    if args.config == 'fast':
        config = create_fast_config()
    elif args.config == 'intensive':
        config = create_intensive_config()
    else:
        config = create_default_config()
    
    # Apply custom parameters
    if args.population:
        config.population_size = args.population
    if args.generations:
        config.max_generations = args.generations
    if args.mutation_rate:
        config.mutation_rate = args.mutation_rate
    if args.crossover_rate:
        config.crossover_rate = args.crossover_rate
    if args.convergence_threshold:
        config.convergence_threshold = args.convergence_threshold
    if args.convergence_window:
        config.convergence_window = args.convergence_window
    if args.min_convergence_generations:
        config.min_generations_for_convergence = args.min_convergence_generations
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.no_parallel:
        config.parallel_evaluation = False
    if args.num_processes:
        config.num_processes = args.num_processes
    
    print("\n🧬 PEPTIDE GENETIC ALGORITHM")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Population size: {config.population_size}")
    print(f"Max generations: {config.max_generations}")
    print(f"Data file: {args.data}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Run parameter sweep or single optimization
    if args.parameter_sweep:
        run_parameter_sweep(config, args.data, args.model_dir, args.output_dir)
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize and run GA
    print("\n🚀 Starting genetic algorithm...")
    start_time = time.time()
    
    ga = PeptideGeneticAlgorithm(config, args.model_dir, args.output_dir)
    results = ga.run_from_csv(args.data)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Genetic algorithm completed in {total_time:.1f} seconds")
    
    # Analyze results
    analyze_best_sequences(results)
    
    # Export for synthesis
    if args.export_synthesis:
        synthesis_file = Path(args.output_dir) / "sequences_for_synthesis.csv"
        export_sequences_for_synthesis(results, str(synthesis_file))
    
    # Create visualizations
    if not args.no_visualization:
        print("\n🎨 Creating visualizations...")
        visualizer = GAVisualizer(args.output_dir)
        visualizer.create_comprehensive_report(results)
    
    # Final summary
    print(f"\n🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    summary = results['evolution_summary']
    best_sequences = results.get('best_sequences', [])
    high_affinity = results.get('high_affinity_candidates', [])
    
    print(f"🏆 Best fitness achieved: {summary['final_best_fitness']:.3f}")
    print(f"🎯 High affinity candidates found: {len(high_affinity)}")
    print(f"⚡ Total evaluations: {summary['total_evaluations']:,}")
    print(f"⏱️  Total time: {summary['total_time']:.1f} seconds")
    
    if best_sequences:
        print(f"\n🥇 BEST SEQUENCE:")
        print(f"   {best_sequences[0]['sequence']}")
        print(f"   Fitness: {best_sequences[0]['fitness_score']:.3f}")
        
        predictions = best_sequences[0].get('predictions', {})
        if predictions:
            print(f"   Binding: GCGR={predictions.get('GCGR', 0):.3f}, "
                  f"GLP1R={predictions.get('GLP1R', 0):.3f}, "
                  f"GIPR={predictions.get('GIPR', 0):.3f}")
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    print(f"🔬 Ready for experimental validation!")


if __name__ == "__main__":
    main()