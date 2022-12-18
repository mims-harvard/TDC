# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/aae/sample.py

import argparse
import os
from typing import List

import torch
import tqdm
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger

from moses.aae import AAE
from moses.script_utils import add_sample_args, set_seed


class AaeGenerator(DistributionMatchingGenerator):
    def __init__(self, config):
        model_config = torch.load(config.config_load)
        model_vocab = torch.load(config.vocab_load)
        model_state = torch.load(config.model_load)
        self.config = config

        device = torch.device(config.device)

        self.model = AAE(model_vocab, model_config)
        self.model.load_state_dict(model_state)
        self.model = self.model.to(device)
        self.model.eval()

    def generate(self, number_samples: int) -> List[str]:
        samples = []
        n = number_samples
        with tqdm.tqdm(total=number_samples, desc="Generating samples") as T:
            while n > 0:
                current_samples = self.model.sample(
                    min(n, self.config.n_batch), self.config.max_len
                )
                samples.extend(current_samples)

                n -= len(current_samples)
                T.update(len(current_samples))
        return samples


def get_parser():
    parser = add_sample_args(argparse.ArgumentParser())
    parser.add_argument("--dist_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--suite", default="v2")
    return parser


def main(config):
    setup_default_logger()

    set_seed(config.seed)
    generator = AaeGenerator(config)

    if config.output_dir is None:
        config.output_dir = os.path.dirname(os.path.realpath(__file__))

    json_file_path = os.path.join(
        config.output_dir, "distribution_learning_results.json"
    )
    assess_distribution_learning(
        generator,
        chembl_training_file=config.dist_file,
        json_output_file=json_file_path,
        benchmark_version=config.suite,
    )

    samples = generator.generate(number_samples=10)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
