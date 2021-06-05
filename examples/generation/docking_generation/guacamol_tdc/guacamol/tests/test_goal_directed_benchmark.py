from typing import Optional, List

import pytest
from rdkit import Chem

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import ScoringFunctionBasedOnRdkitMol, ScoringFunction


class MockScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    For testing purposes: scoring function that returns 0.1 * (number of atoms)
    """

    def score_mol(self, mol: Chem.Mol) -> float:
        return 0.1 * mol.GetNumAtoms()


class MockGenerator(GoalDirectedGenerator):
    """
    Mock generator that returns pre-defined molecules
    """
    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        assert number_molecules == len(self.molecules)
        return self.molecules


def test_removes_duplicates():
    """
    Assert that duplicated molecules (even with different SMILES strings) are considered only once.
    """
    top3 = uniform_specification(3)
    benchmark = GoalDirectedBenchmark('benchmark', MockScoringFunction(), top3)
    generator = MockGenerator(['OCC', 'CCO', 'C(O)C'])

    individual_mock_score = 0.3

    assert benchmark.assess_model(generator).score == pytest.approx(individual_mock_score / 3)


def test_removes_invalid_molecules():
    top3 = uniform_specification(3)
    benchmark = GoalDirectedBenchmark('benchmark', MockScoringFunction(), top3)
    generator = MockGenerator(['OCC', 'invalid', 'invalid2'])

    individual_mock_score = 0.3

    assert benchmark.assess_model(generator).score == pytest.approx(individual_mock_score / 3)


def test_correct_score_averaging():
    top3 = uniform_specification(3)
    benchmark = GoalDirectedBenchmark('benchmark', MockScoringFunction(), top3)
    generator = MockGenerator(['OCC', 'CCCCOCCCC', 'C'])

    expected_score = (0.3 + 0.9 + 0.1) / 3

    assert benchmark.assess_model(generator).score == pytest.approx(expected_score)


def test_correct_score_with_multiple_contributions():
    """
    Verify that 0.5 * (top1 + top3) delivers the correct result
    """
    specification = uniform_specification(1, 3)
    benchmark = GoalDirectedBenchmark('benchmark', MockScoringFunction(), specification)
    generator = MockGenerator(['OCC', 'CCCCOCCCC', 'C'])

    top3 = (0.3 + 0.9 + 0.1) / 3
    top1 = 0.9
    expected_score = (top1 + top3) / 2

    assert benchmark.assess_model(generator).score == pytest.approx(expected_score)
