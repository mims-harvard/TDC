# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/vae/sample.py

import argparse
import os
from typing import List

import torch
import tqdm
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger

from moses.script_utils import add_sample_args, set_seed
from moses.vae.model import VAE


def get_parser():
    parser = add_sample_args(argparse.ArgumentParser())

    parser.add_argument("--dist_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--suite", default="v2")

    return parser


class VaeGenerator(DistributionMatchingGenerator):
    def __init__(self, config):
        model_config = torch.load(config.config_load)
        map_location = "cpu" if config.device == "cpu" else None
        model_state = torch.load(config.model_load, map_location=map_location)
        self.model_vocab = torch.load(config.vocab_load)
        self.config = config

        device = torch.device(config.device)

        # For CUDNN to work properly:
        if device.type.startswith("cuda"):
            torch.cuda.set_device(device.index or 0)

        self.model = VAE(self.model_vocab, model_config)
        self.model.load_state_dict(model_state)
        self.model = self.model.to(device)
        self.model.eval()

    def generate(self, number_samples: int) -> List[str]:
        gen, n = [], number_samples
        T = tqdm.tqdm(range(number_samples), desc="Generating mols")
        while n > 0:
            x = self.model.sample(min(n, self.config.n_batch), self.config.max_len)[-1]
            mols = [self.model_vocab.ids2string(i_x.tolist()) for i_x in x]
            n -= len(mols)
            T.update(len(mols))
            T.refresh()
            gen.extend(mols)
        return gen


def main(config):
    setup_default_logger()

    set_seed(config.seed)

    if config.output_dir is None:
        config.output_dir = os.path.dirname(os.path.realpath(__file__))

    generator = VaeGenerator(config)

    json_file_path = os.path.join(
        config.output_dir, "distribution_learning_results.json"
    )
    assess_distribution_learning(
        generator,
        chembl_training_file=config.dist_file,
        json_output_file=json_file_path,
        benchmark_version=config.suite,
    )


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
