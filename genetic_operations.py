"""
Genetic Operations for Peptide Evolution
Implements selection, crossover, and mutation operations for the genetic algorithm.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import copy


@dataclass
class Individual:
    """Represents an individual peptide in the population."""
    tokens: List[str]  # Tokenized sequence
    sequence: str      # String representation
    fitness_score: float
    predictions: Dict[str, float]
    generation: int
    parent_ids: List[int] = None
    mutation_history: List[str] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.mutation_history is None:
            self.mutation_history = []


class SelectionOperator:
    """Implements various selection strategies for genetic algorithm."""
    
    def __init__(self, strategy: str = "tournament"):
        """
        Initialize selection operator.
        
        Args:
            strategy: Selection strategy ('tournament', 'roulette', 'rank', 'elitist')
        """
        self.strategy = strategy
    
    def tournament_selection(self, 
                           population: List[Individual], 
                           tournament_size: int = 3) -> Individual:
        """
        Tournament selection: select best individual from random tournament.
        
        Args:
            population: Current population
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness_score)
    
    def roulette_selection(self, population: List[Individual]) -> Individual:
        """
        Roulette wheel selection based on fitness scores.
        
        Args:
            population: Current population
            
        Returns:
            Selected individual
        """
        # Handle negative fitness scores by shifting
        min_fitness = min(ind.fitness_score for ind in population)
        if min_fitness < 0:
            adjusted_scores = [ind.fitness_score - min_fitness + 1 for ind in population]
        else:
            adjusted_scores = [ind.fitness_score for ind in population]
        
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            return random.choice(population)
        
        # Generate random number
        r = random.uniform(0, total_fitness)
        
        # Find selected individual
        cumulative = 0
        for i, score in enumerate(adjusted_scores):
            cumulative += score
            if cumulative >= r:
                return population[i]
        
        return population[-1]  # Fallback
    
    def rank_selection(self, population: List[Individual]) -> Individual:
        """
        Rank-based selection: select based on rank rather than raw fitness.
        
        Args:
            population: Current population
            
        Returns:
            Selected individual
        """
        # Sort by fitness (ascending order)
        sorted_pop = sorted(population, key=lambda ind: ind.fitness_score)
        
        # Assign ranks (1 to n)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        
        # Roulette selection on ranks
        r = random.uniform(0, total_rank)
        cumulative = 0
        for i, rank in enumerate(ranks):
            cumulative += rank
            if cumulative >= r:
                return sorted_pop[i]
        
        return sorted_pop[-1]  # Fallback
    
    def elitist_selection(self, 
                         population: List[Individual], 
                         elite_size: int = 1) -> List[Individual]:
        """
        Elitist selection: select top individuals.
        
        Args:
            population: Current population
            elite_size: Number of elite individuals to select
            
        Returns:
            List of elite individuals
        """
        sorted_pop = sorted(population, key=lambda ind: ind.fitness_score, reverse=True)
        return sorted_pop[:elite_size]
    
    def select(self, 
              population: List[Individual], 
              num_selections: int = 1,
              **kwargs) -> List[Individual]:
        """
        Perform selection based on configured strategy.
        
        Args:
            population: Current population
            num_selections: Number of individuals to select
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Selected individuals
        """
        selected = []
        
        if self.strategy == "tournament":
            tournament_size = kwargs.get('tournament_size', 3)
            for _ in range(num_selections):
                selected.append(self.tournament_selection(population, tournament_size))
        
        elif self.strategy == "roulette":
            for _ in range(num_selections):
                selected.append(self.roulette_selection(population))
        
        elif self.strategy == "rank":
            for _ in range(num_selections):
                selected.append(self.rank_selection(population))
        
        elif self.strategy == "elitist":
            elite_size = kwargs.get('elite_size', num_selections)
            selected = self.elitist_selection(population, elite_size)
        
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
        
        return selected


class CrossoverOperator:
    """Implements crossover operations for peptide sequences."""
    
    def __init__(self, sequence_handler):
        """
        Initialize crossover operator.
        
        Args:
            sequence_handler: PeptideSequenceHandler instance
        """
        self.sequence_handler = sequence_handler
    
    def single_point_crossover(self, 
                              parent1: Individual, 
                              parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Single-point crossover between two peptide sequences.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        tokens1 = parent1.tokens.copy()
        tokens2 = parent2.tokens.copy()
        
        # Determine crossover point
        min_len = min(len(tokens1), len(tokens2))
        if min_len <= 1:
            # Cannot perform crossover, return copies
            offspring1 = Individual(
                tokens=tokens1,
                sequence=self.sequence_handler.tokens_to_sequence(tokens1),
                fitness_score=0.0,
                predictions={},
                generation=parent1.generation + 1,
                parent_ids=[id(parent1), id(parent2)],
                mutation_history=["crossover_single_point"]
            )
            offspring2 = Individual(
                tokens=tokens2,
                sequence=self.sequence_handler.tokens_to_sequence(tokens2),
                fitness_score=0.0,
                predictions={},
                generation=parent1.generation + 1,
                parent_ids=[id(parent1), id(parent2)],
                mutation_history=["crossover_single_point"]
            )
            return offspring1, offspring2
        
        crossover_point = random.randint(1, min_len - 1)
        
        # Create offspring
        offspring1_tokens = tokens1[:crossover_point] + tokens2[crossover_point:]
        offspring2_tokens = tokens2[:crossover_point] + tokens1[crossover_point:]
        
        # Repair sequences if needed
        offspring1_tokens = self.sequence_handler.repair_sequence(offspring1_tokens)
        offspring2_tokens = self.sequence_handler.repair_sequence(offspring2_tokens)
        
        offspring1 = Individual(
            tokens=offspring1_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring1_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_single_point"]
        )
        
        offspring2 = Individual(
            tokens=offspring2_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring2_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_single_point"]
        )
        
        return offspring1, offspring2
    
    def two_point_crossover(self, 
                           parent1: Individual, 
                           parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Two-point crossover between two peptide sequences.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        tokens1 = parent1.tokens.copy()
        tokens2 = parent2.tokens.copy()
        
        min_len = min(len(tokens1), len(tokens2))
        if min_len <= 2:
            return self.single_point_crossover(parent1, parent2)
        
        # Two crossover points
        point1 = random.randint(0, min_len - 2)
        point2 = random.randint(point1 + 1, min_len)
        
        # Create offspring by swapping middle section
        offspring1_tokens = tokens1[:point1] + tokens2[point1:point2] + tokens1[point2:]
        offspring2_tokens = tokens2[:point1] + tokens1[point1:point2] + tokens2[point2:]
        
        # Repair sequences
        offspring1_tokens = self.sequence_handler.repair_sequence(offspring1_tokens)
        offspring2_tokens = self.sequence_handler.repair_sequence(offspring2_tokens)
        
        offspring1 = Individual(
            tokens=offspring1_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring1_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_two_point"]
        )
        
        offspring2 = Individual(
            tokens=offspring2_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring2_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_two_point"]
        )
        
        return offspring1, offspring2
    
    def uniform_crossover(self, 
                         parent1: Individual, 
                         parent2: Individual, 
                         swap_probability: float = 0.5) -> Tuple[Individual, Individual]:
        """
        Uniform crossover: swap each position with given probability.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            swap_probability: Probability of swapping each position
            
        Returns:
            Tuple of two offspring individuals
        """
        tokens1 = parent1.tokens.copy()
        tokens2 = parent2.tokens.copy()
        
        # Make sequences same length for uniform crossover
        max_len = max(len(tokens1), len(tokens2))
        while len(tokens1) < max_len:
            tokens1.append(self.sequence_handler.get_weighted_random_residue())
        while len(tokens2) < max_len:
            tokens2.append(self.sequence_handler.get_weighted_random_residue())
        
        offspring1_tokens = []
        offspring2_tokens = []
        
        for i in range(max_len):
            if random.random() < swap_probability:
                # Swap
                offspring1_tokens.append(tokens2[i])
                offspring2_tokens.append(tokens1[i])
            else:
                # No swap
                offspring1_tokens.append(tokens1[i])
                offspring2_tokens.append(tokens2[i])
        
        # Repair sequences
        offspring1_tokens = self.sequence_handler.repair_sequence(offspring1_tokens)
        offspring2_tokens = self.sequence_handler.repair_sequence(offspring2_tokens)
        
        offspring1 = Individual(
            tokens=offspring1_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring1_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_uniform"]
        )
        
        offspring2 = Individual(
            tokens=offspring2_tokens,
            sequence=self.sequence_handler.tokens_to_sequence(offspring2_tokens),
            fitness_score=0.0,
            predictions={},
            generation=parent1.generation + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_history=["crossover_uniform"]
        )
        
        return offspring1, offspring2
    
    def crossover(self, 
                 parent1: Individual, 
                 parent2: Individual, 
                 method: str = "single_point") -> Tuple[Individual, Individual]:
        """
        Perform crossover using specified method.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            method: Crossover method ('single_point', 'two_point', 'uniform')
            
        Returns:
            Tuple of two offspring individuals
        """
        if method == "single_point":
            return self.single_point_crossover(parent1, parent2)
        elif method == "two_point":
            return self.two_point_crossover(parent1, parent2)
        elif method == "uniform":
            return self.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {method}")


class MutationOperator:
    """Implements mutation operations for peptide sequences."""
    
    def __init__(self, sequence_handler):
        """
        Initialize mutation operator.
        
        Args:
            sequence_handler: PeptideSequenceHandler instance
        """
        self.sequence_handler = sequence_handler
    
    def point_mutation(self, 
                      individual: Individual, 
                      mutation_rate: float = 0.1) -> Individual:
        """
        Point mutation: randomly change individual amino acids.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each position
            
        Returns:
            Mutated individual
        """
        tokens = individual.tokens.copy()
        mutations_made = []
        
        for i in range(len(tokens)):
            if random.random() < mutation_rate:
                old_token = tokens[i]
                tokens = self.sequence_handler.mutate_single_position(tokens, i)
                new_token = tokens[i]
                mutations_made.append(f"pos_{i}:{old_token}->{new_token}")
        
        # Ensure sequence is still valid
        tokens = self.sequence_handler.repair_sequence(tokens)
        
        mutated = Individual(
            tokens=tokens,
            sequence=self.sequence_handler.tokens_to_sequence(tokens),
            fitness_score=0.0,
            predictions={},
            generation=individual.generation,
            parent_ids=[id(individual)],
            mutation_history=individual.mutation_history + [f"point_mutation:{','.join(mutations_made)}"]
        )
        
        return mutated
    
    def insertion_mutation(self, individual: Individual) -> Individual:
        """
        Insertion mutation: insert a random amino acid.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        tokens = individual.tokens.copy()
        
        if len(tokens) < 35:  # Only insert if below max length
            position = random.randint(0, len(tokens))
            tokens = self.sequence_handler.insert_residue(tokens, position)
            mutation_record = f"insertion:pos_{position}"
        else:
            # If at max length, do point mutation instead
            return self.point_mutation(individual, mutation_rate=0.1)
        
        # Ensure sequence is still valid
        tokens = self.sequence_handler.repair_sequence(tokens)
        
        mutated = Individual(
            tokens=tokens,
            sequence=self.sequence_handler.tokens_to_sequence(tokens),
            fitness_score=0.0,
            predictions={},
            generation=individual.generation,
            parent_ids=[id(individual)],
            mutation_history=individual.mutation_history + [mutation_record]
        )
        
        return mutated
    
    def deletion_mutation(self, individual: Individual) -> Individual:
        """
        Deletion mutation: delete a random amino acid.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        tokens = individual.tokens.copy()
        
        if len(tokens) > 25:  # Only delete if above min length
            position = random.randint(0, len(tokens) - 1)
            deleted_token = tokens[position]
            tokens = self.sequence_handler.delete_residue(tokens, position)
            mutation_record = f"deletion:pos_{position}:{deleted_token}"
        else:
            # If at min length, do point mutation instead
            return self.point_mutation(individual, mutation_rate=0.1)
        
        # Ensure sequence is still valid
        tokens = self.sequence_handler.repair_sequence(tokens)
        
        mutated = Individual(
            tokens=tokens,
            sequence=self.sequence_handler.tokens_to_sequence(tokens),
            fitness_score=0.0,
            predictions={},
            generation=individual.generation,
            parent_ids=[id(individual)],
            mutation_history=individual.mutation_history + [mutation_record]
        )
        
        return mutated
    
    def block_mutation(self, 
                      individual: Individual, 
                      block_size: int = 3) -> Individual:
        """
        Block mutation: mutate a contiguous block of amino acids.
        
        Args:
            individual: Individual to mutate
            block_size: Size of block to mutate
            
        Returns:
            Mutated individual
        """
        tokens = individual.tokens.copy()
        
        if len(tokens) < block_size:
            return self.point_mutation(individual, mutation_rate=0.3)
        
        # Select random block
        start_pos = random.randint(0, len(tokens) - block_size)
        mutations_made = []
        
        for i in range(start_pos, min(start_pos + block_size, len(tokens))):
            old_token = tokens[i]
            tokens = self.sequence_handler.mutate_single_position(tokens, i)
            new_token = tokens[i]
            mutations_made.append(f"pos_{i}:{old_token}->{new_token}")
        
        # Ensure sequence is still valid
        tokens = self.sequence_handler.repair_sequence(tokens)
        
        mutated = Individual(
            tokens=tokens,
            sequence=self.sequence_handler.tokens_to_sequence(tokens),
            fitness_score=0.0,
            predictions={},
            generation=individual.generation,
            parent_ids=[id(individual)],
            mutation_history=individual.mutation_history + [f"block_mutation:{','.join(mutations_made)}"]
        )
        
        return mutated
    
    def adaptive_mutation(self, 
                         individual: Individual, 
                         generation: int, 
                         max_generations: int) -> Individual:
        """
        Adaptive mutation: mutation rate and type change based on generation.
        
        Args:
            individual: Individual to mutate
            generation: Current generation
            max_generations: Maximum number of generations
            
        Returns:
            Mutated individual
        """
        # Adaptive mutation rate: start high, decrease over time
        progress = generation / max_generations
        base_rate = 0.3 * (1 - progress) + 0.05 * progress  # 0.3 -> 0.05
        
        # Choose mutation type based on generation
        if progress < 0.3:
            # Early generations: more exploration (insertions/deletions)
            mutation_type = random.choices(
                ['point', 'insertion', 'deletion', 'block'],
                weights=[0.4, 0.25, 0.25, 0.1]
            )[0]
        elif progress < 0.7:
            # Middle generations: balanced
            mutation_type = random.choices(
                ['point', 'insertion', 'deletion', 'block'],
                weights=[0.6, 0.15, 0.15, 0.1]
            )[0]
        else:
            # Late generations: more exploitation (point mutations)
            mutation_type = random.choices(
                ['point', 'insertion', 'deletion', 'block'],
                weights=[0.8, 0.05, 0.05, 0.1]
            )[0]
        
        if mutation_type == 'point':
            return self.point_mutation(individual, base_rate)
        elif mutation_type == 'insertion':
            return self.insertion_mutation(individual)
        elif mutation_type == 'deletion':
            return self.deletion_mutation(individual)
        elif mutation_type == 'block':
            return self.block_mutation(individual)
        else:
            return self.point_mutation(individual, base_rate)
    
    def mutate(self, 
              individual: Individual, 
              method: str = "point", 
              **kwargs) -> Individual:
        """
        Perform mutation using specified method.
        
        Args:
            individual: Individual to mutate
            method: Mutation method ('point', 'insertion', 'deletion', 'block', 'adaptive')
            **kwargs: Additional method-specific parameters
            
        Returns:
            Mutated individual
        """
        if method == "point":
            mutation_rate = kwargs.get('mutation_rate', 0.1)
            return self.point_mutation(individual, mutation_rate)
        elif method == "insertion":
            return self.insertion_mutation(individual)
        elif method == "deletion":
            return self.deletion_mutation(individual)
        elif method == "block":
            block_size = kwargs.get('block_size', 3)
            return self.block_mutation(individual, block_size)
        elif method == "adaptive":
            generation = kwargs.get('generation', 0)
            max_generations = kwargs.get('max_generations', 100)
            return self.adaptive_mutation(individual, generation, max_generations)
        else:
            raise ValueError(f"Unknown mutation method: {method}")


class PopulationManager:
    """Manages population-level operations and statistics."""
    
    def __init__(self):
        """Initialize population manager."""
        self.generation_stats = []
    
    def filter_exact_matches(self, population: List[Individual], model_wrapper) -> List[Individual]:
        """Filter out exact matches from population."""
        return [ind for ind in population if not model_wrapper.is_exact_match(ind.sequence)]
    

    def calculate_diversity(self, population: List[Individual]) -> Dict[str, float]:
        """
        Calculate population diversity metrics.
        
        Args:
            population: Current population
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(population) < 2:
            return {'average_similarity': 0.0, 'max_similarity': 0.0, 'min_similarity': 0.0}
        
        from peptide_sequence_handler import PeptideSequenceHandler
        handler = PeptideSequenceHandler()
        
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                sim = handler.calculate_sequence_similarity(
                    population[i].tokens, 
                    population[j].tokens
                )
                similarities.append(sim)
        
        return {
            'average_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'num_comparisons': len(similarities)
        }
    
    def get_population_stats(self, 
                           population: List[Individual], 
                           generation: int) -> Dict[str, any]:
        """
        Calculate comprehensive population statistics including plausibility metrics.
        
        Args:
            population: Current population
            generation: Current generation number
            
        Returns:
            Dictionary with population statistics
        """
        if not population:
            return {}
        
        fitness_scores = [ind.fitness_score for ind in population]
        sequence_lengths = [len(ind.tokens) for ind in population]
        
        # High affinity counts
        high_affinity_counts = []
        all_high_affinity = 0
        
        for ind in population:
            if hasattr(ind, 'predictions') and ind.predictions:
                # Count high affinity predictions (>= threshold)
                high_count = sum(1 for prob in ind.predictions.values() if prob >= 0.5)
                high_affinity_counts.append(high_count)
                if high_count == 3:  # All three receptors
                    all_high_affinity += 1
        
        # Diversity metrics
        diversity = self.calculate_diversity(population)
        
        # ENHANCED: Plausibility tracking
        plausibility_stats = self._calculate_plausibility_stats(population)
        filter_distribution = self._calculate_filter_distribution(population)
        
        stats = {
            'generation': generation,
            'population_size': len(population),
            'fitness': {
                'mean': np.mean(fitness_scores),
                'std': np.std(fitness_scores),
                'min': np.min(fitness_scores),
                'max': np.max(fitness_scores),
                'median': np.median(fitness_scores)
            },
            'sequence_length': {
                'mean': np.mean(sequence_lengths),
                'std': np.std(sequence_lengths),
                'min': np.min(sequence_lengths),
                'max': np.max(sequence_lengths)
            },
            'high_affinity': {
                'counts': high_affinity_counts,
                'mean_count': np.mean(high_affinity_counts) if high_affinity_counts else 0,
                'all_three_count': all_high_affinity,
                'all_three_percentage': (all_high_affinity / len(population)) * 100
            },
            'diversity': diversity,
            'plausibility': plausibility_stats,
            'filter_distribution': filter_distribution
        }
        
        self.generation_stats.append(stats)
        return stats
    
    def _calculate_plausibility_stats(self, population: List[Individual]) -> Dict[str, float]:
        """Calculate plausibility statistics for the population."""
        plausibility_data = {
            'overall': [],
            'chemical': [],
            'motifs': [],
            'proteolysis': [],
            'composition': []
        }
        
        for ind in population:
            if hasattr(ind, 'plausibility_scores') and ind.plausibility_scores:
                for component, scores_list in plausibility_data.items():
                    score = ind.plausibility_scores.get(component, 0.0)
                    scores_list.append(score)
        
        # Calculate statistics for each component
        stats = {}
        for component, scores in plausibility_data.items():
            if scores:
                stats[f'{component}_mean'] = np.mean(scores)
                stats[f'{component}_std'] = np.std(scores)
                stats[f'{component}_min'] = np.min(scores)
                stats[f'{component}_max'] = np.max(scores)
            else:
                stats[f'{component}_mean'] = 0.0
                stats[f'{component}_std'] = 0.0
                stats[f'{component}_min'] = 0.0
                stats[f'{component}_max'] = 0.0
        
        return stats
    
    def _calculate_filter_distribution(self, population: List[Individual]) -> Dict[str, int]:
        """Calculate distribution of filter statuses in population."""
        filter_counts = {
            'HIGHLY_PLAUSIBLE': 0,
            'PLAUSIBLE': 0,
            'SOFT_PENALTY': 0,
            'MAJOR_PENALTY': 0,
            'HARD_REJECTED': 0,
            'ERROR': 0,
            'UNKNOWN': 0
        }
        
        for ind in population:
            status = getattr(ind, 'filter_status', 'UNKNOWN')
            if status in filter_counts:
                filter_counts[status] += 1
            else:
                filter_counts['UNKNOWN'] += 1
        
        return filter_counts
    
    def get_best_individuals(self, 
                           population: List[Individual], 
                           n: int = 10,
                           model_wrapper=None) -> List[Individual]:
        """
        Get the best n individuals from population.
        
        Args:
            population: Current population
            n: Number of best individuals to return
            
        Returns:
            List of best individuals
        """
        if model_wrapper:
            filtered_population = self.filter_exact_matches(population, model_wrapper)
        else:
            filtered_population = population
        
        sorted_pop = sorted(filtered_population, key=lambda ind: ind.fitness_score, reverse=True)
        return sorted_pop[:n]
    
    def remove_duplicates(self, population: List[Individual]) -> List[Individual]:
        """
        Remove duplicate sequences from population.
        
        Args:
            population: Current population
            
        Returns:
            Population without duplicates
        """
        seen_sequences = set()
        unique_population = []
        
        for individual in population:
            if individual.sequence not in seen_sequences:
                seen_sequences.add(individual.sequence)
                unique_population.append(individual)
        
        return unique_population