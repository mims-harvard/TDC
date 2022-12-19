import argparse
import os

from guacamol.utils.helpers import setup_default_logger

from .smiles_rnn_distribution_learner import SmilesRnnDistributionLearner

if __name__ == "__main__":
    setup_default_logger()

    parser = argparse.ArgumentParser(
        description="Distribution learning benchmark for SMILES RNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_data",
        default="data/guacamol_v1_train.smiles",
        help="Full path to SMILES file containing training data",
    )
    parser.add_argument(
        "--valid_data",
        default="data/guacamol_v1_valid.smiles",
        help="Full path to SMILES file containing validation data",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Size of a mini-batch for gradient descent",
    )
    parser.add_argument(
        "--valid_every", default=1000, type=int, help="Validate every so many batches"
    )
    parser.add_argument(
        "--print_every", default=10, type=int, help="Report every so many batches"
    )
    parser.add_argument(
        "--n_epochs", default=10, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_len", default=100, type=int, help="Max length of a SMILES string"
    )
    parser.add_argument(
        "--hidden_size", default=512, type=int, help="Size of hidden layer"
    )
    parser.add_argument(
        "--n_layers", default=3, type=int, help="Number of layers for training"
    )
    parser.add_argument(
        "--rnn_dropout", default=0.2, type=float, help="Dropout value for RNN"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="RNN learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--output_dir", default=None, help="Output directory")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    trainer = SmilesRnnDistributionLearner(
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_len=args.max_len,
        batch_size=args.batch_size,
        rnn_dropout=args.rnn_dropout,
        lr=args.lr,
        valid_every=args.valid_every,
    )

    training_set_file = args.train_data
    validation_set_file = args.valid_data

    with open(training_set_file) as f:
        train_list = f.readlines()

    with open(validation_set_file) as f:
        valid_list = f.readlines()

    trainer.train(training_set=train_list, validation_set=valid_list)

    print(f"All done, your trained model is in {args.output_dir}")
