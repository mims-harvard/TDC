# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/aae/train.py

import torch
from guacamol.utils.helpers import setup_default_logger

from moses.aae import AAE, AAETrainer, get_parser as aae_parser
from moses.script_utils import add_train_args, read_smiles_csv, set_seed
from moses.utils import CharVocab

from moses_baselines.common import read_smiles


def get_parser():
    return add_train_args(aae_parser())


def main(config):
    setup_default_logger()

    set_seed(config.seed)

    train = read_smiles(config.train_load)

    vocab = CharVocab.from_data(train)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)

    device = torch.device(config.device)

    model = AAE(vocab, config)
    model = model.to(device)

    trainer = AAETrainer(config)
    trainer.fit(model, train)

    model.to("cpu")
    torch.save(model.state_dict(), config.model_save)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
