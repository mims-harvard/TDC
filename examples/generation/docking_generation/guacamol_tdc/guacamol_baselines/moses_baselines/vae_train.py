# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/vae/train.py

import torch

from moses.script_utils import add_train_args, set_seed
from moses.vae.config import get_parser as vae_parser
from moses.vae.corpus import OneHotCorpus
from moses.vae.model import VAE
from moses.vae.trainer import VAETrainer

from moses_baselines.common import read_smiles


def get_parser():
    return add_train_args(vae_parser())


def main(config):
    set_seed(config.seed)

    train = read_smiles(config.train_load)

    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)

    corpus = OneHotCorpus(config.n_batch, device)
    train = corpus.fit(train).transform(train)

    model = VAE(corpus.vocab, config).to(device)

    trainer = VAETrainer(config)

    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)
    trainer.fit(model, train)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
