"""
Script for training and evaluating an MLP for DrugCombo benchmark

Some code adapted from https://github.com/yejinjkim/synergy-transfer

"""

import copy
import tdc
from model_classes import *

import torch
import numpy as np
import pandas as pd
import time
import pickle
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader


def arg_parse():
    """
    Commandline arguments parser
    """

    parser = argparse.ArgumentParser(description="Arguments.")
    parser.add_argument("--device", type=int, help="CPU / GPU device.")
    parser.add_argument(
        "--drug_feats", type=str, help="Path to drug features data file."
    )
    parser.add_argument(
        "--data_df", type=str, help="Path to complete data frame for drugcomb"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--cuda", type=bool, help="CUDA")

    parser.set_defaults(
        device="0",
        drug_feats="./data/drug_features.pkl",
        data_df="./data/drugcomb_nci60.pkl",
        epochs=50,
        batch_size=128,
        cuda=True,
    )
    return parser.parse_args()


def set_up_features(drug_feat_file, data_df):
    """
    Reads in drug and cell line features

    ----
    drug_feat_file: dataframe containing drug features
    data_df: complete dataset
    """

    # Create drug feature dataframe
    with open(drug_feat_file, "rb") as handle:
        drug_features = pickle.load(handle)
    drug_features = pd.DataFrame(drug_features).T
    df = pickle.load(open(data_df, "rb"))

    X = df.loc[:, ["Drug1_ID", "Drug1"]].merge(
        drug_features, left_on="Drug1", right_index=True
    )
    X = X.drop_duplicates("Drug1_ID").reset_index(drop=True)
    X = X.loc[:, ["Drug1_ID", 0]].rename(columns={0: "fps", "Drug1_ID": "Drug_ID"})

    Y = df.loc[:, ["Drug2_ID", "Drug2"]].merge(
        drug_features, left_on="Drug2", right_index=True
    )
    Y = Y.drop_duplicates("Drug2_ID").reset_index(drop=True)
    Y = Y.loc[:, ["Drug2_ID", 0]].rename(columns={0: "fps", "Drug2_ID": "Drug_ID"})

    drug_features = pd.concat([X, Y], sort=False).drop_duplicates("Drug_ID")
    drug_features = drug_features.set_index("Drug_ID")

    # Create cell line features dataframe
    cell_features = df.loc[:, ["Cell_Line_ID", "CellLine"]].drop_duplicates(
        "Cell_Line_ID"
    )
    cell_features["CellLine"] = cell_features["CellLine"].apply(
        lambda x: np.concatenate(x)
    )
    return drug_features, cell_features


def train(model, data_loader, loss_fn, batch_size, cuda=True, log=100, lr=0.001):

    """
    Train given pytorch model using the loss function and dataset provided

    ----
    model: pytorch model file
    dataloader: pytorch dataloader
    loss_fn: loss function for training
    batch_size: batch size for dataset
    cuda: Flag for using CUDA (GPUs) during training
    log: Log results every log iterations
    lr: learning rate
    """

    model.train()
    total_loss = 0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()

    for iteration, sample in enumerate(data_loader):
        d1_fp = Variable(sample["d1_fp"].float())
        d2_fp = Variable(sample["d2_fp"].float())
        c_gn = Variable(sample["c_gn"].float())

        out_metric = Variable(sample["out_metric"])

        if cuda:
            d1_fp = d1_fp.cuda()
            d2_fp = d2_fp.cuda()
            c_gn = c_gn.cuda()
            out_metric = out_metric.cuda()

        optimizer.zero_grad()
        out = model(d1_fp, d2_fp, c_gn)
        loss = loss_fn(out_metric, out)

        loss.backward()
        optimizer.step()
        total_loss += loss.data

        # Display output at every log iteration
        if iteration % log == 0 and iteration > 0:
            cur_loss = total_loss.item() / log
            elapsed = time.time() - start_time
            print(
                "| {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                "loss {:8.5f}".format(
                    iteration,
                    int(len(data_loader) / batch_size),
                    elapsed * 1000 / log,
                    cur_loss,
                )
            )

            total_loss = 0
            start_time = time.time()


def evaluate(model, data_loader, loss_fn, cuda=True):
    """
    Evaluate model performance

    ----
    model: trained pytorch model file
    dataloader: pytorch dataloader
    loss_fn: loss function for training
    cuda: Flag for using CUDA (GPUs) during training
    """
    model.eval()
    total_loss = 0
    all_outs = []

    # loss
    with torch.no_grad():
        for iteration, sample in enumerate(data_loader):
            d1_fp = Variable(sample["d1_fp"].float())
            d2_fp = Variable(sample["d2_fp"].float())
            c_gn = Variable(sample["c_gn"].float())

            out_metric = Variable(sample["out_metric"])

            if cuda:
                d1_fp = d1_fp.cuda()
                d2_fp = d2_fp.cuda()
                c_gn = c_gn.cuda()
                out_metric = out_metric.cuda()

            out = model(d1_fp, d2_fp, c_gn)
            loss = loss_fn(out_metric, out)
            total_loss += loss.data
            all_outs.extend(out.view(-1).cpu().numpy())

        mean_loss = total_loss.item() / (iteration + 1)
        print("Validation MAE:", mean_loss)
    return all_outs, mean_loss


def main():

    args = arg_parse()
    loss_fn = nn.L1Loss()

    # Set up training and validation dataframes
    group = tdc.BenchmarkGroup("drugcombo_group", path="data/", file_format="pkl")

    from tdc.multi_pred import DrugSyn

    data_df = DrugSyn(name="DrugComb")

    # Create dataframe for drug features
    # drug to fingerprint mapping
    def get_fp(x):
        return (
            list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)))
            if Chem.MolFromSmiles(x) is not None
            else ""
        )

    df = pickle.load(open("./data/drugcomb.pkl", "rb"))

    fp_map = {}
    all_drugs = set(df["Drug1"].unique()).union(set(df["Drug2"].unique()))
    for drug in all_drugs:
        fp_map[drug] = [get_fp(drug)]

    with open("./data/drug_features.pkl", "wb") as handle:
        pickle.dump(fp_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Set paths
    drug_feat_file = "./data/drug_features.pkl"
    data_df = "./data/drugcomb.pkl"
    drug_features, cell_features = set_up_features(
        drug_feat_file=drug_feat_file, data_df=data_df
    )

    results_all_seeds = {}

    for seed in [1, 2, 3, 4, 5]:
        predictions = {}
        for benchmark in group:
            train, valid = group.get_train_valid_split(
                seed=seed, benchmark=benchmark["name"]
            )

            name = benchmark["name"]
            train_df, valid_df, test_df = train, valid, benchmark["test"]

            train_ = DrugCombDataset(train_df, drug_features, cell_features)
            train_loader = DataLoader(train_, batch_size=args.batch_size, shuffle=True)
            valid_ = DrugCombDataset(valid_df, drug_features, cell_features)
            valid_loader = DataLoader(valid_, batch_size=args.batch_size, shuffle=True)
            test_ = DrugCombDataset(test_df, drug_features, cell_features)
            test_loader = DataLoader(test_, batch_size=args.batch_size, shuffle=True)

            num_genes = len(cell_features.loc[0, "CellLine"])

            model = Comb(num_genes=num_genes)
            if args.cuda:
                model.cuda()

            # Training
            best_val_loss = np.inf
            try:
                for epoch in range(1, args.epochs + 1):
                    epoch_start_time = time.time()
                    train(model, train_loader, loss_fn, args.batch_size, args.cuda)
                    _, val_loss = evaluate(model, valid_loader, loss_fn)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model)
                    print("-" * 89)
            except KeyboardInterrupt:
                print("-" * 89)
                print("Exiting from training early")
                print("Best model MAE:", best_val_loss)

            print("Best model MAE:", best_val_loss)

            predictions[name], _ = evaluate(best_model, test_loader, loss_fn, args.cuda)

        # Evaluate performance
        out = group.evaluate(predictions)
        # print(out)
        results_all_seeds["seed " + str(seed)] = out

    def to_submission_format(results):
        import pandas as pd

        df = pd.DataFrame(results)

        def get_metric(x):
            metric = []
            for i in x:
                metric.append(list(i.values())[0])
            return [round(np.mean(metric), 3), round(np.std(metric), 3)]

        return dict(df.apply(get_metric, axis=1))

    print(to_submission_format(results_all_seeds))


if __name__ == "__main__":
    main()
