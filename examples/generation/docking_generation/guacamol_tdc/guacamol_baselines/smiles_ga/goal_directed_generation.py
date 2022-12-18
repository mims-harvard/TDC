from __future__ import print_function

import argparse
import copy
import json
import os
from collections import namedtuple
from time import time
from typing import List, Optional

import joblib
import nltk
import numpy as np
from joblib import delayed
from rdkit import rdBase

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from guacamol.utils.helpers import setup_default_logger
from . import cfg_util, smiles_grammar

rdBase.DisableLog("rdApp.error")
GCFG = smiles_grammar.GCFG

Molecule = namedtuple("Molecule", ["score", "smiles", "genes"])

from guacamol.scoring_function import max_oracle_num  ### e.g., 10k

max_oracle_num = 10


def cfg_to_gene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [
            idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs
        ]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [
                np.random.randint(0, 256) for _ in range(max_len - len(gene))
            ]
    return gene


def gene_to_cfg(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [
            idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs
        ]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(
            lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != "None"),
            smiles_grammar.GCFG.productions()[rule].rhs(),
        )
        stack.extend(list(rhs)[::-1])
    return prod_rules


def select_parent(population, tournament_size=3):
    idx = np.random.randint(len(population), size=tournament_size)
    best = population[idx[0]]
    for i in idx[1:]:
        if population[i][0] > best[0]:
            best = population[i]
    return best


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def deduplicate(population):
    unique_smiles = set()
    unique_population = []
    for item in population:
        score, smiles, gene = item
        if smiles not in unique_smiles:
            unique_population.append(item)
        unique_smiles.add(smiles)
    return unique_population


def mutate(p_gene, scoring_function):
    c_gene = mutation(p_gene)
    c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    c_score = scoring_function.score(c_smiles)
    return Molecule(c_score, c_smiles, c_gene)


class ChemGEGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        smi_file,
        population_size,
        n_mutations,
        gene_size,
        generations,
        n_jobs=-1,
        random_start=False,
        patience=5,
    ):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.gene_size = gene_size
        self.generations = generations
        self.random_start = random_start
        self.patience = patience

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

        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(
                f"Benchmark requested more molecules than expected: new population is {number_molecules}"
            )

        # fetch initial population?
        if starting_population is None:
            print("selecting initial population...")
            init_size = self.population_size + self.n_mutations
            all_smiles = copy.deepcopy(self.all_smiles)
            # if self.random_start:
            starting_population = np.random.choice(all_smiles, init_size)
            # else:
            #     starting_population = self.top_k(all_smiles, scoring_function, init_size)

        # ### VS top100 as warm start
        # docking_start_file = "/project/molecular_data/graphnn/pyscreener/smiles_ga/docking_zinc_drd3_top100"
        # with open(docking_start_file, 'r') as fin:
        #     lines = fin.readlines()
        # starting_population = [line.strip() for line in lines]
        # starting_population = starting_population[:init_size]
        # ### VS top100 as warm start
        # with open("/project/molecular_data/graphnn/pyscreener/smiles_ga/clean_zinc.txt", 'r') as fin:
        #     lines = fin.readlines()
        # zinc_lst = [line.strip() for line in lines]
        # import random
        # random.seed(1)
        # random.shuffle(zinc_lst)

        # The smiles GA cannot deal with '%' in SMILES strings (used for two-digit ring numbers).
        starting_population = [
            smiles for smiles in starting_population if "%" not in smiles
        ]

        # calculate initial genes
        initial_genes = [
            cfg_to_gene(cfg_util.encode(s), max_len=self.gene_size)
            for s in starting_population
        ]

        # score initial population
        initial_scores = scoring_function.score_list(starting_population)
        population = [
            Molecule(*m)
            for m in zip(initial_scores, starting_population, initial_genes)
        ]
        population = sorted(population, key=lambda x: x.score, reverse=True)[
            : self.population_size
        ]
        population_scores = [p.score for p in population]

        # evolution: go go go!!
        t0 = time()

        patience = 0

        for generation in range(self.generations):

            old_scores = population_scores
            # select random genes
            all_genes = [molecule.genes for molecule in population]
            choice_indices = np.random.choice(
                len(all_genes), self.n_mutations, replace=True
            )
            genes_to_mutate = [all_genes[i] for i in choice_indices]

            # evolve genes
            joblist = (delayed(mutate)(g, scoring_function) for g in genes_to_mutate)
            new_population = self.pool(joblist)

            # join and dedup
            population += new_population
            population = deduplicate(population)

            # survival of the fittest
            population = sorted(population, key=lambda x: x.score, reverse=True)[
                : self.population_size
            ]

            # stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()

            population_scores = [p.score for p in population]

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
                f"{gen_time:.2f} sec/gen | "
                f"{mol_sec:.2f} mol/sec"
            )

        # finally
        return [molecule.smiles for molecule in population[:number_molecules]]


def main():
    population_size = 5
    generations_num = int(max_oracle_num / population_size)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population_size", type=int, default=population_size)
    parser.add_argument("--n_mutations", type=int, default=200)
    parser.add_argument("--gene_size", type=int, default=300)
    parser.add_argument("--generations", type=int, default=generations_num)
    parser.add_argument("--n_jobs", type=int, default=-1)
    # parser.add_argument('--random_start', action='store_true')
    parser.add_argument("--random_start", default=True)  ## limit oracle
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

    optimiser = ChemGEGenerator(
        smi_file=args.smiles_file,
        population_size=args.population_size,
        n_mutations=args.n_mutations,
        gene_size=args.gene_size,
        generations=args.generations,
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
