import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Any, Dict

import guacamol
from guacamol.goal_directed_benchmark import (
    GoalDirectedBenchmark,
    GoalDirectedBenchmarkResult,
)
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.data import get_time_string

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def assess_goal_directed_generation(
    goal_directed_molecule_generator: GoalDirectedGenerator,
    json_output_file="output_goal_directed.json",
    benchmark_version="v3",
) -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    logger.info(
        f"Benchmarking goal-directed molecule generation, version {benchmark_version}"
    )
    benchmarks = goal_directed_benchmark_suite(version_name=benchmark_version)

    results = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmarks=benchmarks,
    )

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results["guacamol_version"] = guacamol.__version__
    benchmark_results["benchmark_suite_version"] = benchmark_version
    benchmark_results["timestamp"] = get_time_string()
    benchmark_results["results"] = [vars(result) for result in results]

    logger.info(f"Save results to file {json_output_file}")
    with open(json_output_file, "wt") as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_goal_directed_benchmarks(
    goal_directed_molecule_generator: GoalDirectedGenerator,
    benchmarks: List[GoalDirectedBenchmark],
) -> List[GoalDirectedBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        goal_directed_molecule_generator: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    logger.info(f"Number of benchmarks: {len(benchmarks)}")

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f"Running benchmark {i}/{len(benchmarks)}: {benchmark.name}")
        result = benchmark.assess_model(goal_directed_molecule_generator)
        logger.info(f'Results for the benchmark "{result.benchmark_name}":')
        logger.info(f"  Score: {result.score:.6f}")
        logger.info(
            f"  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}"
        )
        logger.info(f"  Metadata: {result.metadata}")
        results.append(result)

    logger.info("Finished execution of the benchmarks")

    return results
