import logging
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from guacamol.goal_directed_score_contributions import ScoreContributionSpecification, compute_global_score
from guacamol.scoring_function import ScoringFunction, ScoringFunctionWrapper
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.chemistry import canonicalize_list, remove_duplicates, calculate_internal_pairwise_similarities

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GoalDirectedBenchmarkResult:
    """
    Contains the results of a goal-directed benchmark.
    """

    def __init__(self, benchmark_name: str, score: float, optimized_molecules: List[Tuple[str, float]],
                 execution_time: float, number_scoring_function_calls: int, metadata: Dict[str, Any]) -> None:
        """
        Args:
            benchmark_name: name of the goal-directed benchmark
            score: benchmark score
            optimized_molecules: generated molecules, given as a list of (SMILES string, molecule score) tuples
            execution_time: execution time for the benchmark in seconds
            number_scoring_function_calls: number of calls to the scoring function
            metadata: benchmark-specific information
        """
        self.benchmark_name = benchmark_name
        self.score = score
        self.optimized_molecules = optimized_molecules
        self.execution_time = execution_time
        self.number_scoring_function_calls = number_scoring_function_calls
        self.metadata = metadata


class GoalDirectedBenchmark:
    """
    This class assesses how well a model is able to generate molecules satisfying a given objective.
    """

    def __init__(self, name: str, objective: ScoringFunction,
                 contribution_specification: ScoreContributionSpecification,
                 starting_population: Optional[List[str]] = None) -> None:
        """
        Args:
            name: Benchmark name
            objective: Objective for the goal-directed optimization
            contribution_specification: Specifies how to calculate the global benchmark score
        """
        self.name = name
        self.objective = objective
        self.wrapped_objective = ScoringFunctionWrapper(scoring_function=objective)
        self.contribution_specification = contribution_specification
        self.starting_population = starting_population

    def assess_model(self, model: GoalDirectedGenerator) -> GoalDirectedBenchmarkResult:
        """
        Assess the given model by asking it to generate molecules optimizing a scoring function.
        The number of molecules to generate is determined automatically from the score contribution specification.

        Args:
            model: model to assess
        """
        number_molecules_to_generate = max(self.contribution_specification.top_counts)
        start_time = time.time()
        molecules = model.generate_optimized_molecules(scoring_function=self.wrapped_objective,
                                                       number_molecules=number_molecules_to_generate,
                                                       starting_population=self.starting_population
                                                       )
        end_time = time.time()

        canonicalized_molecules = canonicalize_list(molecules, include_stereocenters=False)
        unique_molecules = remove_duplicates(canonicalized_molecules)
        scores = self.objective.score_list(unique_molecules)

        if len(unique_molecules) != number_molecules_to_generate:
            number_missing = number_molecules_to_generate - len(unique_molecules)
            logger.warning(f'An incorrect number of distinct molecules was generated: '
                           f'{len(unique_molecules)} instead of {number_molecules_to_generate}. '
                           f'Padding scores with {number_missing} zeros...')
            scores.extend([0.0] * number_missing)

        global_score, top_x_dict = compute_global_score(self.contribution_specification, scores)

        scored_molecules = zip(unique_molecules, scores)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], x[0]), reverse=True)

        internal_similarities = calculate_internal_pairwise_similarities(unique_molecules)

        # accumulate internal_similarities in metadata
        int_simi_histogram = np.histogram(internal_similarities, bins=10, range=(0, 1), density=True)

        metadata: Dict[str, Any] = {}
        metadata.update(top_x_dict)
        metadata['internal_similarity_max'] = internal_similarities.max()
        metadata['internal_similarity_mean'] = internal_similarities.mean()
        metadata["internal_similarity_histogram_density"] = int_simi_histogram[0].tolist(),
        metadata["internal_similarity_histogram_bins"] = int_simi_histogram[1].tolist(),

        return GoalDirectedBenchmarkResult(benchmark_name=self.name,
                                           score=global_score,
                                           optimized_molecules=sorted_scored_molecules,
                                           execution_time=end_time - start_time,
                                           number_scoring_function_calls=self.wrapped_objective.evaluations,
                                           metadata=metadata)
