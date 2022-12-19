import heapq
from typing import List, Optional, Tuple

import joblib
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import delayed

from guacamol.scoring_function import max_oracle_num

from .chembl_file_reader import ChemblFileReader


class BestFromChemblOptimizer(GoalDirectedGenerator):
    """
    Goal-directed molecule generator that will simply look for the most adequate molecules present in a file.
    """

    def __init__(self, smiles_reader: ChemblFileReader, n_jobs: int) -> None:
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        # get a list of all the smiles
        self.smiles = [s for s in smiles_reader]

        ### limit oracle calls.
        from random import shuffle

        shuffle(self.smiles)
        self.smiles = self.smiles[:max_oracle_num]

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
        """
        Will iterate through the reference set of SMILES strings and select the best molecules.
        """
        return self.top_k(self.smiles, scoring_function, number_molecules)
