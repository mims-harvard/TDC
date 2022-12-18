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

# from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
# from guacamol.goal_directed_generator import GoalDirectedGenerator
# from guacamol.scoring_function import ScoringFunction
# from guacamol.utils.helpers import setup_default_logger
# from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from graph_ga import crossover as co, mutate as mu

result_folder = "/project/molecular_data/graphnn/pyscreener/graph_ga/results.3/"


from tdc import Oracle

drd3_oracle = Oracle(
    name="Docking_Score",
    software="vina",
    pyscreener_path="/project/molecular_data/graphnn/pyscreener",
    receptors=["/project/molecular_data/graphnn/pyscreener/testing_inputs/DRD3.pdb"],
    center=(9, 22.5, 26),
    size=(15, 15, 15),
    buffer=10,
    path="/project/molecular_data/graphnn/pyscreener/my_test/",
    num_worker=1,
    ncpu=10,
)

qed_oracle = Oracle(name="qed")

# def drd3_docking_oracle(smiles):
#     return max(-drd3_oracle(smiles),0)


def drd3_docking_oracle(smiles):
    qed = qed_oracle(smiles)
    if qed < 0.2:
        return 0.0
    return max(-drd3_oracle(smiles), 0)


# def drd3_docking_oracle(smiles):
#     return random.random()


def canonicalize(smiles: str, include_stereocenters=True):
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


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
    # print(population_probs, 'population_probs')
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


class GB_GA_Generator:
    def __init__(
        self, smi_file, population_size, offspring_size, generations, mutation_rate
    ):
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.smiles2score = {}  ### save the result up to now.
        self.oracle_num = 0

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def generate_optimized_molecules(self):
        from random import shuffle

        shuffle(self.all_smiles)
        starting_population = self.all_smiles[: self.population_size]
        starting_population = [canonicalize(smiles) for smiles in starting_population]
        print(
            "---------- before init, oracle_num",
            self.oracle_num,
            "population_size",
            self.population_size,
        )
        # select initial population
        # population_smiles = heapq.nlargest(self.population_size, starting_population, key=oracle)
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        # population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
        population_scores = []
        for smiles in population_smiles:
            smiles = canonicalize(smiles)
            if smiles in self.smiles2score:
                score = self.smiles2score[smiles]
            else:
                try:
                    score = drd3_docking_oracle(smiles)
                except:
                    score = 0
                self.oracle_num += 1
                print("docking number", self.oracle_num)
                if self.oracle_num % 100 == 0:
                    pickle.dump(
                        self.smiles2score,
                        open(result_folder + str(self.oracle_num) + ".pkl", "wb"),
                    )
                self.smiles2score[smiles] = score
            population_scores.append(score)

        for generation in tqdm(range(self.generations)):  ### 1k

            # new_population
            mating_pool = make_mating_pool(
                population_mol, population_scores, self.offspring_size
            )
            # offspring_mol = self.pool(delayed(reproduce)(mating_pool, self.mutation_rate) for _ in range(self.population_size))
            offspring_mol = []
            for _ in range(self.population_size):
                offspring = reproduce(mating_pool, self.mutation_rate)
                offspring_mol.append(offspring)

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            old_scores = population_scores
            # population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
            population_scores = []
            for m in population_mol:
                smiles = Chem.MolToSmiles(m)
                smiles = canonicalize(smiles)
                if smiles in self.smiles2score:
                    score = self.smiles2score[smiles]
                else:
                    try:
                        score = drd3_docking_oracle(smiles)
                    except:
                        score = 0
                    self.oracle_num += 1
                    print("docking number", self.oracle_num)
                    if self.oracle_num % 100 == 0:
                        pickle.dump(
                            self.smiles2score,
                            open(result_folder + str(self.oracle_num) + ".pkl", "wb"),
                        )
                    self.smiles2score[smiles] = score
                population_scores.append(drd3_docking_oracle(Chem.MolToSmiles(m)))

            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(
                population_tuples, key=lambda x: x[0], reverse=True
            )[: self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

        return [Chem.MolToSmiles(m) for m in population_mol][:]


ga = GB_GA_Generator(
    smi_file="data/guacamol_v1_all.smiles",
    population_size=100,
    offspring_size=200,
    generations=100000,
    mutation_rate=0.01,
)

ga.generate_optimized_molecules()


num2score = {}

for file in os.listdir(result_folder):
    num = int(file.split(".")[0])
    smiles2score = pickle.load(open(result_folder + file, "rb"))
    smiles_score_lst = [(smiles, score) for smiles, score in smiles2score.items()]
    smiles_score_lst.sort(key=lambda x: x[1], reverse=True)
    score_lst = [i[1] for i in smiles_score_lst]
    mean_score = np.mean(score_lst[:10])
    num2score[num] = mean_score


num_score_lst = [(num, score) for num, score in num2score.items()]
num_score_lst.sort(key=lambda x: x[0])
print(num_score_lst)
