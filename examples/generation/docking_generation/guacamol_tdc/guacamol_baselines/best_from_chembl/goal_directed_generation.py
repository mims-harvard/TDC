import argparse
import os

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.helpers import setup_default_logger

from .chembl_file_reader import ChemblFileReader
from .optimizer import BestFromChemblOptimizer

if __name__ == "__main__":
    setup_default_logger()

    parser = argparse.ArgumentParser(
        description="Goal-directed benchmark for best molecules from SMILES file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--suite", default="v3")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    smiles_reader = ChemblFileReader(args.smiles_file)

    optimizer = BestFromChemblOptimizer(smiles_reader=smiles_reader, n_jobs=args.n_jobs)

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")

    assess_goal_directed_generation(
        optimizer, json_output_file=json_file_path, benchmark_version=args.suite
    )
