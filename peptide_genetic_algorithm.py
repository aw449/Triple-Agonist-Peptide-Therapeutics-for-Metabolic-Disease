"""
Main Genetic Algorithm Framework for Peptide Evolution
Orchestrates the entire genetic algorithm process for peptide design.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import copy
import time
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from peptide_sequence_handler import PeptideSequenceHandler
from gat_model_wrapper import GATModelWrapper
from genetic_operations import (
    Individual, SelectionOperator, CrossoverOperator, 
    MutationOperator, PopulationManager
)


@dataclass 
class GAConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 100
    max_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_rate: float = 0.1
    selection_strategy: str = "tournament"
    crossover_method: str = "single_point"
    mutation_method: str = "adaptive"
    tournament_size: int = 3
    target_threshold: float = 0.5  # Threshold for high affinity
    min_sequence_length: int = 25
    max_sequence_length: int = 40
    diversity_threshold: float = 0.1  # Maintain this much diversity
    stagnation_limit: int = 10  # Stop if no improvement for this many generations
    convergence_threshold: float = 0.1  # Stop when improvement rate < this value per generation
    convergence_window: int = 3  # Number of generations to evaluate convergence over
    min_generations_for_convergence: int = 3
    parallel_evaluation: bool = True
    num_processes: int = None  # None = auto-detect
    random_seed: int = 42


class PeptideGeneticAlgorithm:
    """
    Main Genetic Algorithm class for peptide design optimization.
    """
    
    def __init__(self, 
                 config: GAConfig,
                 model_dir: str = "kfold_transfer_learning_results_GAT",
                 output_dir: str = "peptide_ga_results"):
        """
        Initialize the genetic algorithm.
        
        Args:
            config: GA configuration
            model_dir: Directory containing the trained GAT model
            output_dir: Directory to save results
        """
        self.config = config
        self.model_dir = model_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seed
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize components
        print("Initializing Peptide Genetic Algorithm...")
        self.sequence_handler = PeptideSequenceHandler()
        self.model_wrapper = GATModelWrapper(model_dir)
        
        # Initialize operators
        self.selector = SelectionOperator(config.selection_strategy)
        self.crossover_op = CrossoverOperator(self.sequence_handler)
        self.mutation_op = MutationOperator(self.sequence_handler)
        self.population_manager = PopulationManager()
        
        # Evolution tracking
        self.current_population = []
        self.best_individuals = []
        self.evolution_history = []
        self.generation_stats = []

        self.all_unique_peptides = {}  # sequence -> {'individual': Individual, 'first_gen': int, 'best_fitness': float}
        self.generation_unique_counts = []  # Track unique peptides per generation
        
        # Performance tracking
        self.start_time = None
        self.total_evaluations = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []

        self.improvement_rates = []  # Track improvement rates over generations
        self.convergence_data = []  # Track convergence metrics for visualization
        
        print(f"GA initialized with {config.population_size} individuals, {config.max_generations} generations")
        print(f"Target: EC50 < {1000 * (1 - config.target_threshold):.0f} pM across all receptors")
    
    def load_seed_sequences(self, csv_file: str, sequence_column: str = "sequence") -> List[str]:
        """
        Load seed sequences from CSV file.
        
        Args:
            csv_file: Path to CSV file containing sequences
            sequence_column: Name of column containing sequences
            
        Returns:
            List of seed sequences
        """
        print(f"Loading seed sequences from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            if sequence_column not in df.columns:
                raise ValueError(f"Column '{sequence_column}' not found in CSV")
            
            # Filter out empty sequences
            sequences = df[sequence_column].dropna().tolist()
            sequences = [seq.strip() for seq in sequences if seq.strip()]

            self.model_wrapper.set_reference_sequences(sequences)
            
            print(f"Loaded {len(sequences)} seed sequences")
            return sequences
            
        except Exception as e:
            print(f"Error loading seed sequences: {e}")
            print("Generating random seed sequences instead...")
            return self._generate_random_seeds(50)
    
    def _generate_random_seeds(self, count: int) -> List[str]:
        """Generate random seed sequences."""
        seeds = []
        for _ in range(count):
            length = random.randint(self.config.min_sequence_length, self.config.max_sequence_length)
            tokens = self.sequence_handler.generate_random_sequence(length)
            sequence = self.sequence_handler.tokens_to_sequence(tokens)
            seeds.append(sequence)
        return seeds
    
    def initialize_population(self, seed_sequences: List[str]) -> List[Individual]:
        """
        Initialize the population from seed sequences.
        
        Args:
            seed_sequences: List of seed sequences
            
        Returns:
            Initial population
        """
        print("Initializing population...")
        population = []
        
        # Use seed sequences
        for i, seq in enumerate(seed_sequences[:self.config.population_size]):
            tokens = self.sequence_handler.tokenize_sequence(seq)
            
            # Ensure valid length and chemical plausibility
            if not self.sequence_handler.is_valid_length(tokens, 
                                                        self.config.min_sequence_length, 
                                                        self.config.max_sequence_length):
                tokens = self.sequence_handler.repair_sequence(tokens, 
                                                              self.config.min_sequence_length,
                                                              self.config.max_sequence_length)
            
            if not self.sequence_handler.is_chemically_plausible(tokens):
                tokens = self.sequence_handler.repair_sequence(tokens)
            
            if not self.sequence_handler.is_chemically_plausible(tokens):
                tokens = self.sequence_handler.repair_sequence(tokens)
            
            individual = Individual(
                tokens=tokens,
                sequence=self.sequence_handler.tokens_to_sequence(tokens),
                fitness_score=0.0,
                predictions={},
                generation=0
            )
            population.append(individual)
        
        # Fill remaining slots with random sequences if needed
        while len(population) < self.config.population_size:
            length = random.randint(self.config.min_sequence_length, self.config.max_sequence_length)
            tokens = self.sequence_handler.generate_random_sequence(length)
            sequence_str = self.sequence_handler.tokens_to_sequence(tokens)
            
            max_attempts = 10
            attempts = 0
            while self.model_wrapper.is_exact_match(sequence_str) and attempts < max_attempts:
                tokens = self.sequence_handler.generate_random_sequence(length)
                sequence_str = self.sequence_handler.tokens_to_sequence(tokens)
                attempts += 1
            
            # Only add if not an exact match
            if not self.model_wrapper.is_exact_match(sequence_str):
                individual = Individual(
                    tokens=tokens,
                    sequence=sequence_str,
                    fitness_score=0.0,
                    predictions={},
                    generation=0
                )
                population.append(individual)
        
        print(f"Population initialized with {len(population)} individuals")
        return population
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """
        Evaluate fitness of a single individual with enhanced plausibility scoring.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Individual with updated fitness and predictions
        """
        try:
            # Get comprehensive fitness evaluation from model
            fitness_result = self.model_wrapper.evaluate_fitness(
                individual.sequence, 
                self.config.target_threshold
            )
            
            # Update individual with all metrics
            individual.fitness_score = fitness_result['fitness_score']
            individual.predictions = fitness_result['predictions']
            
            # Store additional plausibility information
            individual.plausibility_scores = fitness_result.get('plausibility_scores', {})
            individual.filter_status = fitness_result.get('filter_status', 'UNKNOWN')
            individual.base_activity_score = fitness_result.get('base_activity_score', 0.0)
            
            return individual
            
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            # Return individual with zero fitness
            individual.fitness_score = 0.0
            individual.predictions = {'GCGR': 0.0, 'GLP1R': 0.0, 'GIPR': 0.0}
            individual.plausibility_scores = {'overall': 0.0, 'chemical': 0.0, 'motifs': 0.0, 'proteolysis': 0.0, 'composition': 0.0}
            individual.filter_status = 'ERROR'
            return individual

    def _update_unique_peptides_tracking(self, population: List[Individual], generation: int):
            """
            Update tracking of all unique peptides across generations.
            
            Args:
                population: Current generation population
                generation: Current generation number
            """
            generation_unique_count = 0
            
            for individual in population:
                sequence = individual.sequence
                
                if sequence not in self.all_unique_peptides:
                    # New unique peptide found
                    self.all_unique_peptides[sequence] = {
                        'individual': copy.deepcopy(individual),
                        'first_generation': generation,
                        'best_fitness': individual.fitness_score,
                        'generation_history': [generation]
                    }
                    generation_unique_count += 1
                else:
                    # Peptide seen before, update if fitness is better
                    existing_entry = self.all_unique_peptides[sequence]
                    if individual.fitness_score > existing_entry['best_fitness']:
                        existing_entry['individual'] = copy.deepcopy(individual)
                        existing_entry['best_fitness'] = individual.fitness_score
                    
                    # Track generation history
                    if generation not in existing_entry['generation_history']:
                        existing_entry['generation_history'].append(generation)
            
            self.generation_unique_counts.append({
                'generation': generation,
                'new_unique_peptides': generation_unique_count,
                'total_unique_peptides': len(self.all_unique_peptides)
            })
            
            print(f"  New unique peptides this generation: {generation_unique_count}")
            print(f"  Total unique peptides discovered: {len(self.all_unique_peptides)}")
    
    
    def evaluate_population_parallel(self, population: List[Individual]) -> List[Individual]:
        """
        Evaluate population fitness in parallel.
        
        Args:
            population: Population to evaluate
            
        Returns:
            Population with updated fitness scores
        """
        if not self.config.parallel_evaluation:
            # Sequential evaluation
            evaluated = []
            for individual in population:
                evaluated.append(self.evaluate_individual(individual))
            return evaluated
        
        # Parallel evaluation
        num_processes = self.config.num_processes or min(mp.cpu_count(), len(population))
        
        # Note: Due to the complexity of the GAT model, we'll use ThreadPoolExecutor
        # instead of ProcessPoolExecutor to avoid serialization issues
        with ThreadPoolExecutor(max_workers=min(4, len(population))) as executor:
            evaluated = list(executor.map(self.evaluate_individual, population))
        
        self.total_evaluations += len(population)
        return evaluated
    
    def evaluate_population_sequential(self, population: List[Individual]) -> List[Individual]:
        """Evaluate population sequentially (backup method)."""
        print(f"Evaluating {len(population)} individuals...")
        evaluated = []
        
        for i, individual in enumerate(population):
            if i % 20 == 0:
                print(f"  Evaluated {i}/{len(population)} individuals")
            evaluated.append(self.evaluate_individual(individual))
        
        self.total_evaluations += len(population)
        return evaluated
    
    def create_next_generation(self, 
                          population: List[Individual], 
                          generation: int) -> List[Individual]:
        """
        Create the next generation through selection, crossover, and mutation.
        """
        next_generation = []
        exact_matches_rejected = 0  # Track rejections
        
        # Elitism: keep best individuals (but track if they're exact matches)
        elite_count = int(self.config.elitism_rate * self.config.population_size)
        if elite_count > 0:
            elites = self.population_manager.get_best_individuals(population, elite_count)
            for elite in elites:
                if self.model_wrapper.is_exact_match(elite.sequence):
                    print(f"  Warning: Elite sequence is exact match (fitness: {elite.fitness_score:.3f})")
                next_generation.append(elite)  # Keep elites even if exact matches
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Selection
            parent1 = self.selector.select(population, 1, 
                                        tournament_size=self.config.tournament_size)[0]
            parent2 = self.selector.select(population, 1, 
                                        tournament_size=self.config.tournament_size)[0]
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring1, offspring2 = self.crossover_op.crossover(
                    parent1, parent2, self.config.crossover_method
                )
            else:
                # No crossover, copy parents
                offspring1 = copy.deepcopy(parent1)
                offspring2 = copy.deepcopy(parent2)
                offspring1.generation = generation
                offspring2.generation = generation
            
            # Mutation - ALWAYS mutate if parent was exact match
            should_mutate1 = (random.random() < self.config.mutation_rate or 
                            self.model_wrapper.is_exact_match(parent1.sequence))
            should_mutate2 = (random.random() < self.config.mutation_rate or 
                            self.model_wrapper.is_exact_match(parent2.sequence))
            
            if should_mutate1:
                offspring1 = self.mutation_op.mutate(
                    offspring1, 
                    method=self.config.mutation_method,
                    generation=generation,
                    max_generations=self.config.max_generations
                )
            
            if should_mutate2:
                offspring2 = self.mutation_op.mutate(
                    offspring2, 
                    method=self.config.mutation_method,
                    generation=generation,
                    max_generations=self.config.max_generations
                )
            
            # ADD THESE LINES: Only reject if genetic operations FAILED to create novelty
            # (i.e., mutation/crossover didn't change an exact match)
            if (self.model_wrapper.is_exact_match(offspring1.sequence) and 
                generation > 0):  # Allow exact matches in generation 0
                exact_matches_rejected += 1
                # Try harder mutation
                offspring1 = self.mutation_op.mutate(
                    offspring1, 
                    method="point",
                    mutation_rate=0.3  # Higher rate to force change
                )
            
            if (self.model_wrapper.is_exact_match(offspring2.sequence) and 
                generation > 0):
                exact_matches_rejected += 1
                # Try harder mutation
                offspring2 = self.mutation_op.mutate(
                    offspring2, 
                    method="point", 
                    mutation_rate=0.3
                )
            
            # Add offspring to next generation
            next_generation.append(offspring1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(offspring2)
        
        if exact_matches_rejected > 0:
            print(f"  Applied extra mutations to {exact_matches_rejected} unchanged sequences")
        
        # Remove duplicates and maintain diversity
        next_generation = self.population_manager.remove_duplicates(next_generation)
        
        # If population too small, add random individuals
        while len(next_generation) < self.config.population_size:
            length = random.randint(self.config.min_sequence_length, self.config.max_sequence_length)
            tokens = self.sequence_handler.generate_random_sequence(length)
            
            individual = Individual(
                tokens=tokens,
                sequence=self.sequence_handler.tokens_to_sequence(tokens),
                fitness_score=0.0,
                predictions={},
                generation=generation
            )
            next_generation.append(individual)
        
        return next_generation[:self.config.population_size]
    
    def check_termination_criteria(self, generation: int, population: List[Individual]) -> Tuple[bool, str]:
        """
        Check if termination criteria are met.
        
        Args:
            generation: Current generation
            population: Current population
            
        Returns:
            Tuple of (should_terminate, reason)
        """
        # Max generations reached
        if generation >= self.config.max_generations:
            return True, f"Maximum generations ({self.config.max_generations}) reached"
        
        
        best_fitness = max(ind.fitness_score for ind in population)

        # Fitness threshold reached
        if len(self.best_fitness_history) >= 2:
            current_fitness = self.best_fitness_history[-1]
            previous_fitness = self.best_fitness_history[-2]
            improvement_rate = current_fitness - previous_fitness
            self.improvement_rates.append(improvement_rate)
            
            # Store convergence data for visualization
            self.convergence_data.append({
                'generation': generation,
                'best_fitness': current_fitness,
                'improvement_rate': improvement_rate,
                'cumulative_improvement': current_fitness - self.best_fitness_history[0] if self.best_fitness_history else 0
            })
        
        # MODIFICATION: Convergence-based termination check
        if (generation >= self.config.min_generations_for_convergence and 
            len(self.improvement_rates) >= self.config.convergence_window):
            
            # Calculate average improvement rate over the convergence window
            recent_improvements = self.improvement_rates[-self.config.convergence_window:]
            avg_improvement_rate = np.mean(recent_improvements)
            
            # Check if improvement rate has fallen below threshold
            if avg_improvement_rate < self.config.convergence_threshold:
                return True, (f"Convergence achieved: average improvement rate "
                            f"({avg_improvement_rate:.4f}) below threshold "
                            f"({self.config.convergence_threshold}) over last "
                            f"{self.config.convergence_window} generations")
        
        # Stagnation check
        if self.best_fitness_history:
            recent_best = max(self.best_fitness_history[-min(self.config.stagnation_limit, len(self.best_fitness_history)):])
            if abs(best_fitness - recent_best) < 0.01:  # No significant improvement
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.config.stagnation_limit:
                    return True, f"No improvement for {self.config.stagnation_limit} generations"
            else:
                self.stagnation_counter = 0
        
        return False, ""
    
    def generate_convergence_plot(self, save_path: Optional[str] = None):
        """
        Generate convergence plot showing fitness progression and improvement rates.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime, timedelta
        except ImportError:
            print("matplotlib not available for convergence plotting")
            return
        
        if not self.convergence_data:
            print("No convergence data available for plotting")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Genetic Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
        
        generations = [data['generation'] for data in self.convergence_data]
        fitness_values = [data['best_fitness'] for data in self.convergence_data]
        improvement_rates = [data['improvement_rate'] for data in self.convergence_data]
        cumulative_improvements = [data['cumulative_improvement'] for data in self.convergence_data]
        
        # Plot 1: Best fitness over generations
        ax1.plot(generations, fitness_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Best Fitness Over Generations', fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement rate over generations
        ax2.plot(generations, improvement_rates, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.axhline(y=self.config.convergence_threshold, color='r', linestyle='--', 
                label=f'Convergence Threshold ({self.config.convergence_threshold})')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Fitness Improvement Rate Per Generation', fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Improvement Rate (Fitness/Generation)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add moving average of improvement rate
        if len(improvement_rates) >= self.config.convergence_window:
            moving_avg = []
            for i in range(len(improvement_rates)):
                if i >= self.config.convergence_window - 1:
                    window_start = i - self.config.convergence_window + 1
                    avg = np.mean(improvement_rates[window_start:i+1])
                    moving_avg.append(avg)
                else:
                    moving_avg.append(np.nan)
            
            ax2.plot(generations, moving_avg, 'orange', linewidth=3, alpha=0.7,
                    label=f'{self.config.convergence_window}-Gen Moving Average')
            ax2.legend()
        
        # Plot 3: Cumulative improvement from start
        ax3.plot(generations, cumulative_improvements, 'purple', linewidth=2, marker='^', markersize=4)
        ax3.set_title('Cumulative Fitness Improvement from Start', fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Cumulative Improvement')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        
        plt.show()
        
        # Print convergence summary
        print(f"\n{'='*60}")
        print("CONVERGENCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        if self.convergence_data:
            total_improvement = cumulative_improvements[-1] if cumulative_improvements else 0
            avg_improvement_rate = np.mean(improvement_rates) if improvement_rates else 0
            final_improvement_rate = np.mean(improvement_rates[-self.config.convergence_window:]) if len(improvement_rates) >= self.config.convergence_window else 0
            
            print(f"Total fitness improvement: {total_improvement:.4f}")
            print(f"Average improvement rate: {avg_improvement_rate:.4f} fitness/generation")
            print(f"Final {self.config.convergence_window}-generation average rate: {final_improvement_rate:.4f} fitness/generation")
            print(f"Convergence threshold: {self.config.convergence_threshold:.4f} fitness/generation")
            
            if final_improvement_rate < self.config.convergence_threshold:
                print("✅ Algorithm converged (improvement rate below threshold)")
            else:
                print("⚠️  Algorithm did not converge (improvement rate above threshold)")
        
    def run_evolution(self, seed_sequences: List[str]) -> Dict:
        """
        Run the complete genetic algorithm evolution process.
        
        Args:
            seed_sequences: List of seed sequences to start with
            
        Returns:
            Dictionary with evolution results
        """
        print(f"\n{'='*80}")
        print("STARTING PEPTIDE EVOLUTION")
        print(f"{'='*80}")
        
        self.start_time = time.time()
        
        # Initialize population
        self.current_population = self.initialize_population(seed_sequences)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            gen_start_time = time.time()
            
            print(f"\n--- Generation {generation + 1}/{self.config.max_generations} ---")

            # ADD THESE LINES: Update population for diversity calculation
            current_sequences = [ind.sequence for ind in self.current_population]
            self.model_wrapper.update_current_population(current_sequences)
            
            # Evaluate population
            if generation == 0 or any(ind.fitness_score == 0 for ind in self.current_population):
                try:
                    self.current_population = self.evaluate_population_parallel(self.current_population)
                except Exception as e:
                    print(f"Parallel evaluation failed: {e}")
                    print("Falling back to sequential evaluation...")
                    self.current_population = self.evaluate_population_sequential(self.current_population)
            

            self._update_unique_peptides_tracking(self.current_population, generation)

            
            # Calculate statistics
            stats = self.population_manager.get_population_stats(self.current_population, generation + 1)
            self.generation_stats.append(stats)
            
            # Track best fitness
            best_fitness = stats['fitness']['max']
            self.best_fitness_history.append(best_fitness)
            
            # Update best individuals
            current_best = self.population_manager.get_best_individuals(self.current_population, 10)
            self.best_individuals.extend(current_best)
            
            # Print generation summary
            gen_time = time.time() - gen_start_time
            print(f"  Best fitness: {best_fitness:.3f}")
            print(f"  Mean fitness: {stats['fitness']['mean']:.3f}")
            print(f"  High affinity (all 3): {stats['high_affinity']['all_three_count']}/{stats['population_size']}")
            print(f"  Diversity: {stats['diversity']['average_similarity']:.3f}")
            print(f"  Generation time: {gen_time:.1f}s")
            
            # Check termination criteria
            should_terminate, reason = self.check_termination_criteria(generation + 1, self.current_population)
            if should_terminate:
                print(f"\nTermination criteria met: {reason}")
                break
            
            # Create next generation
            if generation < self.config.max_generations - 1:
                self.current_population = self.create_next_generation(self.current_population, generation + 1)
        
        # Final evaluation and results
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("EVOLUTION COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Total evaluations: {self.total_evaluations}")
        print(f"Final generation: {len(self.generation_stats)}")
        
        # Compile results
        results = self.compile_results()
        
        # Save results
        self.save_results(results)
        
        return results
    
    def compile_results(self) -> Dict:
        """
        Compile comprehensive results from the evolution process.
        
        Returns:
            Dictionary with complete results
        """
            # Get final best individuals
        all_final = self.population_manager.get_best_individuals(
            self.current_population, 
            len(self.current_population)  # Get all, then filter
        )
        
        # ADD THESE LINES: Filter out exact matches
        # Separate exact matches for reporting
        exact_matches = [ind for ind in all_final if self.model_wrapper.is_exact_match(ind.sequence)]
        novel_sequences = [ind for ind in all_final if not self.model_wrapper.is_exact_match(ind.sequence)]
        
        print(f"  Final population: {len(all_final)} total sequences")
        print(f"  Novel sequences: {len(novel_sequences)}")
        print(f"  Exact matches: {len(exact_matches)} (excluded from results)")
    
    # Use novel sequences for results
        final_best = novel_sequences[:min(50, len(novel_sequences))]

        # All-time best individuals
        all_time_candidates = self.population_manager.get_best_individuals(
            self.best_individuals, 
            200  # Get more, then filter
        )
        
        # ADD THESE LINES: Filter out exact matches from all-time best
        novel_all_time = [ind for ind in all_time_candidates if not self.model_wrapper.is_exact_match(ind.sequence)]
        all_time_best = novel_all_time[:100]
        
        # High affinity candidates (all three receptors)
        high_affinity_candidates = [
            ind for ind in all_time_best 
            if hasattr(ind, 'predictions') and ind.predictions and
            all(prob >= self.config.target_threshold for prob in ind.predictions.values())
        ]

        top_20_overall = self._get_top_unique_peptides_overall(20)

        final_convergence_metrics = {}
        if self.convergence_data:
            recent_improvements = self.improvement_rates[-self.config.convergence_window:] if len(self.improvement_rates) >= self.config.convergence_window else self.improvement_rates
            final_convergence_metrics = {
                'total_improvement': self.convergence_data[-1]['cumulative_improvement'] if self.convergence_data else 0,
                'average_improvement_rate': np.mean(self.improvement_rates) if self.improvement_rates else 0,
                'final_window_improvement_rate': np.mean(recent_improvements) if recent_improvements else 0,
                'converged': np.mean(recent_improvements) < self.config.convergence_threshold if recent_improvements else False,
                'convergence_threshold': self.config.convergence_threshold,
                'convergence_window': self.config.convergence_window
            }
        
        results = {
            'config': self.config.__dict__,
            'evolution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'total_evaluations': self.total_evaluations,
                'generations_completed': len(self.generation_stats),
                'final_best_fitness': max(ind.fitness_score for ind in self.current_population),
                'high_affinity_found': len(high_affinity_candidates)
            },
            'best_sequences': [
                {
                    'sequence': ind.sequence,
                    'fitness_score': ind.fitness_score,
                    'predictions': ind.predictions,
                    'generation': ind.generation,
                    'length': len(ind.tokens),
                    'rank': i + 1
                }
                for i, ind in enumerate(final_best[:20])
            ],
            'high_affinity_candidates': [
                {
                    'sequence': ind.sequence,
                    'fitness_score': ind.fitness_score,
                    'predictions': ind.predictions,
                    'generation': ind.generation,
                    'length': len(ind.tokens)
                }
                for ind in high_affinity_candidates[:20]
            ],
            # MODIFICATION 5: Add top 20 overall unique peptides section
            'top_20_overall_unique_peptides': top_20_overall,
            'unique_peptides_per_generation': self.generation_unique_counts,  # MODIFICATION: Add generation tracking
            'generation_stats': self.generation_stats,
            'fitness_history': self.best_fitness_history,
            'generation_stats': self.generation_stats,
            'fitness_history': self.best_fitness_history,
            'convergence_data': self.convergence_data
        }
        
        return results
    
    def _get_top_unique_peptides_overall(self, top_n: int = 20) -> List[Dict]:
        """
        Get top N unique peptides overall from all generations based on fitness scores.
        
        Args:
            top_n: Number of top peptides to return
            
        Returns:
            List of dictionaries containing top unique peptides information
        """
        # Sort all unique peptides by their best fitness scores (descending)
        sorted_unique_peptides = sorted(
            self.all_unique_peptides.items(),
            key=lambda x: x[1]['best_fitness'],
            reverse=True
        )
        
        top_peptides = []
        for i, (sequence, peptide_data) in enumerate(sorted_unique_peptides[:top_n]):
            individual = peptide_data['individual']
            
            peptide_info = {
                'rank': i + 1,
                'sequence': sequence,
                'fitness_score': peptide_data['best_fitness'],
                'predictions': individual.predictions,
                'first_discovered_generation': peptide_data['first_generation'],
                'generation_history': peptide_data['generation_history'],
                'total_generations_present': len(peptide_data['generation_history']),
                'length': len(individual.tokens),
                'plausibility_scores': getattr(individual, 'plausibility_scores', {}),
                'filter_status': getattr(individual, 'filter_status', 'UNKNOWN')
            }
            
            top_peptides.append(peptide_info)
        
        return top_peptides
    
    def save_results(self, results: Dict):
        """
        Save results to files.
        
        Args:
            results: Results dictionary to save
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"ga_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best sequences as CSV
        if results['best_sequences']:
            df_best = pd.DataFrame(results['best_sequences'])
            csv_path = self.output_dir / f"best_sequences_{timestamp}.csv"
            df_best.to_csv(csv_path, index=False)
        
        # Save high affinity candidates as CSV
        if results['high_affinity_candidates']:
            df_candidates = pd.DataFrame(results['high_affinity_candidates'])
            candidates_path = self.output_dir / f"high_affinity_candidates_{timestamp}.csv"
            df_candidates.to_csv(candidates_path, index=False)
        
        if results['top_20_overall_unique_peptides']:
            df_top20 = pd.DataFrame(results['top_20_overall_unique_peptides'])
            top20_path = self.output_dir / f"top_20_overall_unique_peptides_{timestamp}.csv"
            df_top20.to_csv(top20_path, index=False)
            print(f"  Top 20 overall unique peptides: {top20_path.name}")
        
        # MODIFICATION 8: Save unique peptides per generation tracking
        if results['unique_peptides_per_generation']:
            df_unique_gen = pd.DataFrame(results['unique_peptides_per_generation'])
            unique_gen_path = self.output_dir / f"unique_peptides_per_generation_{timestamp}.csv"
            df_unique_gen.to_csv(unique_gen_path, index=False)
            print(f"  Unique peptides per generation: {unique_gen_path.name}")

        if self.convergence_data:
            convergence_plot_path = self.output_dir / f"convergence_plot_{timestamp}.png"
            self.generate_convergence_plot(str(convergence_plot_path))
            
            # Save convergence data as CSV for further analysis
            df_convergence = pd.DataFrame(self.convergence_data)
            convergence_csv_path = self.output_dir / f"convergence_data_{timestamp}.csv"
            df_convergence.to_csv(convergence_csv_path, index=False)
            print(f"  Convergence data: {convergence_csv_path.name}")
        
        print(f"Results saved to {self.output_dir}")
        print(f"  Main results: {json_path.name}")
        if results['best_sequences']:
            print(f"  Best sequences: {csv_path.name}")
        if results['high_affinity_candidates']:
            print(f"  High affinity candidates: {candidates_path.name}")
    
    def run_from_csv(self, csv_file: str, sequence_column: str = "sequence") -> Dict:
        """
        Convenience method to run GA from CSV file.
        
        Args:
            csv_file: Path to CSV file with seed sequences
            sequence_column: Column name containing sequences
            
        Returns:
            Evolution results dictionary
        """
        seed_sequences = self.load_seed_sequences(csv_file, sequence_column)
        return self.run_evolution(seed_sequences)