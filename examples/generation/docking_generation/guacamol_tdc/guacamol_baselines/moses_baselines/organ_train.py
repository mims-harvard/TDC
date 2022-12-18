# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/organ/train.py

import torch
import rdkit

from moses.organ import ORGAN, ORGANTrainer, get_parser as organ_parser
from moses.script_utils import add_train_args, set_seed, MetricsReward
from moses.utils import CharVocab
from multiprocessing import Pool

from moses_baselines.common import read_smiles

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_parser():
    parser = add_train_args(organ_parser())

    parser.add_argument(
        "--n_ref_subsample",
        type=int,
        default=500,
        help="Number of reference molecules (sampling from training data)",
    )
    parser.add_argument(
        "--addition_rewards",
        nargs="+",
        type=str,
        choices=MetricsReward.supported_metrics,
        default=[],
        help="Adding of addition rewards",
    )

    return parser


def main(config):
    set_seed(config.seed)

    train = read_smiles(config.train_load)
    vocab = CharVocab.from_data(train)
    device = torch.device(config.device)

    with Pool(config.n_jobs) as pool:
        reward_func = MetricsReward(
            train,
            config.n_ref_subsample,
            config.rollouts,
            pool,
            config.addition_rewards,
        )
        model = ORGAN(vocab, config, reward_func)
        model = model.to(device)

        trainer = ORGANTrainer(config)
        trainer.fit(model, train)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
