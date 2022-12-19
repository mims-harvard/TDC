import argparse
import os

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.helpers import setup_default_logger

from .generator import RandomSmilesSampler
from .optimizer import RandomSamplingOptimizer

if __name__ == "__main__":
    setup_default_logger()

    parser = argparse.ArgumentParser(
        description="Molecule distribution learning benchmark for random smiles sampler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(args.smiles_file, "r") as smiles_file:
        smiles_list = smiles_file.readlines()

    sampler = RandomSmilesSampler(molecules=smiles_list)

    optimizer = RandomSamplingOptimizer(sampler=sampler)

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")

    assess_goal_directed_generation(optimizer, json_output_file=json_file_path)
