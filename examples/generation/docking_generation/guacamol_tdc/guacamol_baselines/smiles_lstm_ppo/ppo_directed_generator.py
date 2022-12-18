from pathlib import Path
from typing import List, Optional
import torch

from smiles_lstm_ppo.ppo_generator import PPOMoleculeGenerator
from smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic
from smiles_lstm_hc.rnn_utils import load_rnn_model
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize_list
from guacamol.goal_directed_generator import GoalDirectedGenerator


class PPODirectedGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        pretrained_model_path: str,
        num_epochs=4,
        episode_size=1028,
        batch_size=512,
        entropy_weight=1,
        kl_div_weight=5,
        clip_param=0.2,
        number_final_samples=1028,
        sample_final_model_only=False,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.number_final_samples = number_final_samples
        self.sample_final_model_only = sample_final_model_only
        self.model_args = {
            "num_epochs": num_epochs,
            "optimize_batch_size": batch_size,
            "optimize_episode_size": episode_size,
            "entropy_weight": entropy_weight,
            "kl_div_weight": kl_div_weight,
            "clip_param": clip_param,
        }
        self.max_seq_len = 100

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix(".json")

        smiles_rnn = load_rnn_model(
            model_def, self.pretrained_model_path, device, copy_to_cpu=True
        )
        model = SmilesRnnActorCritic(smiles_rnn=smiles_rnn).to(device)

        generator = PPOMoleculeGenerator(
            model=model, max_seq_length=self.max_seq_len, device=device
        )

        molecules = generator.optimise(
            objective=scoring_function, start_population=[], **self.model_args
        )

        # take the molecules seen during the hill-climbing, and also sample from the final model
        samples = [m.smiles for m in molecules]
        if self.sample_final_model_only:
            samples.clear()
        samples += generator.sample(max(number_molecules, self.number_final_samples))

        # calculate the scores and return the best ones
        samples = canonicalize_list(samples)
        scores = scoring_function.score_list(samples)

        scored_molecules = zip(samples, scores)
        sorted_scored_molecules = sorted(
            scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True
        )

        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]
