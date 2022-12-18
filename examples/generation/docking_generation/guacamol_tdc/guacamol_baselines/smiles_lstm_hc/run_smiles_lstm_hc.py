from pathlib import Path
from typing import List, Optional

import joblib
import torch
from random import random
from guacamol.goal_directed_generator import GoalDirectedGenerator

# from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize_list, canonicalize
from joblib import delayed

from smiles_lstm_hc.rnn_generator import SmilesRnnMoleculeGenerator
from smiles_lstm_hc.rnn_utils import load_rnn_model

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

global oracle_num
oracle_num = 0


def drd3_docking_oracle(smiles):
    # oracle_num += 1
    # print('Docking call', oracle_num)
    return min(max(-drd3_oracle(smiles) / 15.0, 0), 1)


# def drd3_docking_oracle(smiles):
#     return random()


class SmilesRnnDirectedGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        pretrained_model_path: str,
        n_epochs=4,
        mols_to_sample=100,
        keep_top=512,
        optimize_n_epochs=2,
        max_len=100,
        optimize_batch_size=64,
        number_final_samples=1028,
        sample_final_model_only=False,
        random_start=False,
        smi_file=None,
        n_jobs=-1,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_batch_size = optimize_batch_size
        self.optimize_n_epochs = optimize_n_epochs
        self.pretrain_n_epochs = 0
        self.max_len = max_len
        self.number_final_samples = number_final_samples
        self.sample_final_model_only = sample_final_model_only
        self.random_start = random_start
        self.smi_file = smi_file
        self.pool = joblib.Parallel(n_jobs=n_jobs)

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
        scoring_function,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:

        # fetch initial population?
        # if starting_population is None:
        #     print('selecting initial population...')
        #     if self.random_start:
        #         starting_population = []
        #     else:
        #         all_smiles = self.load_smiles_from_file(self.smi_file)
        #         starting_population = self.top_k(all_smiles, scoring_function, self.mols_to_sample)

        starting_population = []

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix(".json")

        model = load_rnn_model(
            model_def, self.pretrained_model_path, device, copy_to_cpu=True
        )

        generator = SmilesRnnMoleculeGenerator(
            model=model, max_len=self.max_len, device=device
        )

        molecules = generator.optimise(
            objective=scoring_function,
            start_population=starting_population,
            n_epochs=self.n_epochs,
            mols_to_sample=self.mols_to_sample,
            keep_top=self.keep_top,
            optimize_batch_size=self.optimize_batch_size,
            optimize_n_epochs=self.optimize_n_epochs,
            pretrain_n_epochs=self.pretrain_n_epochs,
        )

        # take the molecules seen during the hill-climbing, and also sample from the final model
        samples = [m.smiles for m in molecules]
        if self.sample_final_model_only:
            samples.clear()
        samples += generator.sample(max(number_molecules, self.number_final_samples))

        # calculate the scores and return the best ones
        samples = canonicalize_list(samples)
        # scores = scoring_function.score_list(samples)
        scores = []
        for smiles in samples:
            try:
                score = scoring_function(smiles)
            except:
                score = 0.0
            scores.append(score)

        scored_molecules = zip(samples, scores)
        sorted_scored_molecules = sorted(
            scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True
        )

        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]


# model = SmilesRnnDirectedGenerator(pretrained_model_path = 'smiles_lstm_hc/pretrained_model/model_final_0.473.json')
model = SmilesRnnDirectedGenerator(
    pretrained_model_path="smiles_lstm_hc/pretrained_model/model_final_0.473.json",
    n_epochs=5000,
    mols_to_sample=100,
)
model.generate_optimized_molecules(
    scoring_function=drd3_docking_oracle, number_molecules=100
)
