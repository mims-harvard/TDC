import argparse
import logging
import os
from pathlib import Path

import torch

from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger

from .rnn_utils import load_rnn_model, set_random_seed
from .smiles_rnn_generator import SmilesRnnGenerator

if __name__ == '__main__':
    setup_default_logger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--model_path', default=None, help='Full path to SMILES RNN model')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--dist_file', default='data/guacamol_v1_all.smiles', help='Distribution file')
    parser.add_argument('--suite', default='v2')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'device:\t{device}')

    set_random_seed(args.seed, device)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.model_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.model_path = os.path.join(dir_path, 'pretrained_model', 'model_final_0.473.pt')

    model_def = Path(args.model_path).with_suffix('.json')
    model = load_rnn_model(model_def, args.model_path, device, copy_to_cpu=True)
    generator = SmilesRnnGenerator(model=model, device=device)

    json_file_path = os.path.join(args.output_dir, 'distribution_learning_results.json')
    assess_distribution_learning(generator,
                                 chembl_training_file=args.dist_file,
                                 json_output_file=json_file_path,
                                 benchmark_version=args.suite)
