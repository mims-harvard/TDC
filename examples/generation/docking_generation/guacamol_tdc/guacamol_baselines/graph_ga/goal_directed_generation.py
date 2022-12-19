from __future__ import print_function

import argparse
import heapq
import json
import os
import random
from time import time
from typing import List, Optional

from tqdm import tqdm
import pickle
import joblib
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from guacamol.utils.helpers import setup_default_logger
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from . import crossover as co, mutate as mu


def read_in_num(filename):
    with open(filename, "r") as fin:
        line = fin.readline()
    return int(line)


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs

    mol_list, score_list = [], []
    for mol, score in zip(population_mol, population_scores):
        # if 0<= score <=1:
        mol_list.append(mol)
        score_list.append(score)

    # print("score_list", score_list)

    population_mol, population_scores = mol_list, score_list
    sum_scores = sum(population_scores)
    if sum_scores == 0:
        population_probs = [
            1.0 / len(population_scores) for i in range(len(population_scores))
        ]
    else:
        population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol, p=population_probs, size=offspring_size, replace=True
    )
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print("bad smiles")
    return new_population


class GB_GA_Generator(GoalDirectedGenerator):
    def __init__(
        self,
        smi_file,
        population_size,
        offspring_size,
        generations,
        mutation_rate,
        n_jobs=-1,
        random_start=False,
        patience=5,
    ):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience

        self.docking_num_file = (
            "/project/molecular_data/graphnn/pyscreener/docking_num.txt"
        )
        self.smiles2score = {}  ### save the result up to now.
        self.oracle_num = 0
        self.population_size = 25

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:

        # if number_molecules > self.population_size:
        #     self.population_size = number_molecules
        #     print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')
        from random import shuffle

        shuffle(self.all_smiles)

        # fetch initial population?
        # if starting_population is None:
        #     print('selecting initial population...')
        #     # if self.random_start:
        #     # if True: ### random start for limit oracle call
        #     # else:
        #         # starting_population = self.top_k(self.all_smiles[:self.population_size*10], scoring_function, self.population_size)
        starting_population = np.random.choice(self.all_smiles, self.population_size)

        print(
            "---------- before init",
            self.oracle_num,
            "population_size",
            self.population_size,
        )
        # select initial population
        population_smiles = heapq.nlargest(
            self.population_size, starting_population, key=scoring_function.score
        )
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = self.pool(
            delayed(score_mol)(m, scoring_function.score) for m in population_mol
        )
        for smiles, score in zip(population_smiles, population_scores):
            print("===docking score", score)
            self.smiles2score[smiles] = score
        self.oracle_num += len(population_mol) * 2
        print("---------- before ga", self.oracle_num)
        # evolution: go go go!!
        t0 = time()

        patience = 0

        for generation in tqdm(range(self.generations)):  ### 1k

            pickle.dump(
                self.smiles2score,
                open(
                    "/project/molecular_data/graphnn/pyscreener/graph_ga/"
                    + str(self.oracle_num)
                    + ".pkl",
                    "wb",
                ),
            )
            print("-------save", self.oracle_num)
            if self.oracle_num > 5000:
                exit()

            # new_population
            print(len(population_mol), len(population_scores), self.offspring_size)
            mating_pool = make_mating_pool(
                population_mol, population_scores, self.offspring_size
            )
            offspring_mol = self.pool(
                delayed(reproduce)(mating_pool, self.mutation_rate)
                for _ in range(self.population_size)
            )

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            population_scores = self.pool(
                delayed(score_mol)(m, scoring_function.score) for m in population_mol
            )
            self.oracle_num += len(population_mol)
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(
                population_tuples, key=lambda x: x[0], reverse=True
            )[: self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            for mol, score in zip(population_mol, population_scores):
                smiles = Chem.MolToSmiles(mol)
                print("===docking score", score)
                self.smiles2score[smiles] = score

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f"Failed to progress: {patience}")
                if patience >= self.patience:
                    print(f"No more patience, bailing...")
                    break
            else:
                patience = 0

            print(
                f"{generation} | "
                f"max: {np.max(population_scores):.3f} | "
                f"avg: {np.mean(population_scores):.3f} | "
                f"min: {np.min(population_scores):.3f} | "
                f"std: {np.std(population_scores):.3f} | "
                f"sum: {np.sum(population_scores):.3f} | "
                f"{gen_time:.2f} sec/gen | "
                f"{mol_sec:.2f} mol/sec"
            )

        # finally
        return [Chem.MolToSmiles(m) for m in population_mol][:number_molecules]


def main():
    population_size = 3
    offspring_size = 100
    max_oracle_num = 20000
    generations_num = int(max_oracle_num / offspring_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population_size", type=int, default=population_size)
    parser.add_argument("--offspring_size", type=int, default=offspring_size)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument(
        "--generations", type=int, default=generations_num
    )  ## 1k -> 10 limit oracle   50*200 = max_oracle_num=10k
    parser.add_argument("--n_jobs", type=int, default=-1)
    # parser.add_argument('--random_start', action='store_true')
    parser.add_argument("--random_start", default=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--suite", default="v3")

    args = parser.parse_args()

    np.random.seed(args.seed)

    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # save command line args
    with open(os.path.join(args.output_dir, "goal_directed_params.json"), "w") as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    optimiser = GB_GA_Generator(
        smi_file=args.smiles_file,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        n_jobs=args.n_jobs,
        random_start=args.random_start,
        patience=args.patience,
    )

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")
    assess_goal_directed_generation(
        optimiser, json_output_file=json_file_path, benchmark_version=args.suite
    )


if __name__ == "__main__":
    main()
