import argparse
import json
import logging
import os
from typing import List, Optional

import joblib
import numpy as np
from guacamol.scoring_function import MoleculewiseScoringFunction
from joblib import delayed
from rdkit import Chem, rdBase

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger

from graph_mcts.goal_directed_generation import find_molecule, State, Node
from graph_mcts.stats import Stats, get_stats_from_pickle, get_stats_from_smiles

rdBase.DisableLog("rdApp.error")


class DummyScoringFunction(MoleculewiseScoringFunction):
    """
    Dummy scoring function to store in State instances (no scoring needed for distribution learning)
    """

    def raw_score(self, smiles: str) -> float:
        return 0.0


def gen_search(state: State) -> State:
    """
    Starts from a state and updates it (more exactly, the internal molecule) until the state is terminal.
    """
    while not state.terminal():
        state = state.next_state()
    return state


def sample_molecule(mol, smiles, max_atoms, max_children, stats):
    """
    Sample one molecule.
    """
    root_state = State(
        scoring_function=DummyScoringFunction(),
        mol=mol,
        smiles=smiles,
        max_atoms=max_atoms,
        max_children=max_children,
        stats=stats,
        seed=0,
    )
    random_state = gen_search(root_state)

    return random_state.smiles


class GB_MCTS_Sampler(DistributionMatchingGenerator):
    def __init__(
        self,
        pickle_directory: str,
        population_size,
        generations,
        num_sims,
        max_children,
        init_smiles,
        max_atoms,
        n_jobs=-1,
        random_start=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.pickle_directory = pickle_directory
        self.population_size = population_size
        self.generations = generations
        self.random_start = random_start
        self.num_sims = num_sims
        self.max_children = max_children
        self.init_smiles = init_smiles
        self.init_mol = Chem.MolFromSmiles(init_smiles)
        self.max_atoms = max_atoms

        self.stats = get_stats_from_pickle(self.pickle_directory)

    @staticmethod
    def sanitize(population):
        new_population = []
        smiles_set = set()
        for smiles in population:
            if smiles is not None and smiles not in smiles_set:
                smiles_set.add(smiles)
                new_population.append(smiles)
        return new_population

    def generate(self, number_samples: int) -> List[str]:
        population = []

        while len(population) != number_samples:
            remaining_samples = number_samples - len(population)

            job = delayed(sample_molecule)(
                self.init_mol,
                self.init_smiles,
                self.max_atoms,
                self.max_children,
                self.stats,
            )

            new_mols = self.pool(job for _ in range(remaining_samples))
            new_mols = self.sanitize(new_mols)

            population += new_mols

        return population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_file",
        help="Location of the ChEMBL dataset to use for the distribution benchmarks.",
        default="data/guacamol_v1_all.smiles",
    )
    parser.add_argument(
        "--pickle_directory",
        help="Directory containing pickle files with the distribution statistics",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--num_sims", type=int, default=40)
    parser.add_argument("--max_children", type=int, default=25)
    parser.add_argument("--max_atoms", type=int, default=60)
    parser.add_argument("--init_smiles", type=str, default="CC")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--suite", default="v2")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.pickle_directory is None:
        args.pickle_directory = os.path.dirname(os.path.realpath(__file__))

    np.random.seed(args.seed)

    setup_default_logger()

    # save command line args
    with open(
        os.path.join(args.output_dir, "distribution_learning_params.json"), "w"
    ) as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    sampler = GB_MCTS_Sampler(
        pickle_directory=args.pickle_directory,
        n_jobs=args.n_jobs,
        random_start=args.random_start,
        num_sims=args.num_sims,
        max_children=args.max_children,
        init_smiles=args.init_smiles,
        max_atoms=args.max_atoms,
        generations=args.generations,
        population_size=args.population_size,
    )

    json_file_path = os.path.join(args.output_dir, "distribution_learning_results.json")
    assess_distribution_learning(
        sampler,
        json_output_file=json_file_path,
        chembl_training_file=args.smiles_file,
        benchmark_version=args.suite,
    )


if __name__ == "__main__":
    main()
