import os
import rdkit
import torch
import random
import pathlib
import argparse
import numpy as np
import logging as log
from tqdm import tqdm
from rdkit import Chem, RDLogger

from .estimator.estimator import Estimator
from .estimator.models import Discriminator
from .proposal.models.editor_basic import BasicEditor
from .proposal.proposal import Proposal_Random, Proposal_Editor, Proposal_Mix
from .sampler import Sampler_SA, Sampler_MH, Sampler_Recursive
from .datasets.utils import load_mols

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--run_exist", action="store_true")
    parser.add_argument("--root_dir", type=str, default="MARS")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--run_dir", type=str, default="runs/debug")
    parser.add_argument("--editor_dir", type=str, default=None)
    parser.add_argument("--mols_ref", type=str, default=None)
    parser.add_argument("--mols_init", type=str, default=None)
    parser.add_argument("--vocab", type=str, default="chembl")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--max_size", type=int, default=40)
    parser.add_argument("--num_mols", type=int, default=50)
    parser.add_argument("--num_step", type=int, default=200)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)

    parser.add_argument(
        "--sampler", type=str, default="sa", help="mcmc sampling algorithm"
    )
    parser.add_argument(
        "--proposal", type=str, default="mix", help="how to pose proposals"
    )
    parser.add_argument(
        "--rand_ratio", type=float, default=0.1, help="random ratio in mix proposal"
    )
    # parser.add_argument('--objectives', type=str,   default='gsk3b,jnk3,qed,sa,div')
    parser.add_argument("--objectives", type=str, default="docking")
    parser.add_argument("--score_wght", type=str, default="  1.0")
    parser.add_argument("--score_succ", type=str, default="  0.5")
    parser.add_argument("--score_clip", type=str, default="  0.6")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_atom_feat", type=int, default=17)
    parser.add_argument("--n_bond_feat", type=int, default=5)
    parser.add_argument("--n_node_hidden", type=int, default=64)
    parser.add_argument("--n_edge_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    args = parser.parse_args()
    if args.debug:
        args.num_runs, args.num_mols, args.num_step = 1, 100, 10
    if args.run_dir == "runs/debug":
        args.run_exist = True

    config = vars(args)
    config["device"] = torch.device(config["device"])
    config["objectives"] = config["objectives"].split(",")
    config["score_wght"] = [float(_) for _ in config["score_wght"].split(",")]
    config["score_succ"] = [float(_) for _ in config["score_succ"].split(",")]
    config["score_clip"] = [float(_) for _ in config["score_clip"].split(",")]
    assert len(config["score_wght"]) == len(config["objectives"])
    assert len(config["score_succ"]) == len(config["objectives"])
    assert len(config["score_clip"]) == len(config["objectives"])
    config["run_dir"] = os.path.join(config["root_dir"], config["run_dir"])
    config["data_dir"] = os.path.join(config["root_dir"], config["data_dir"])
    os.makedirs(config["run_dir"], exist_ok=config["run_exist"])
    log.basicConfig(
        format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
    )
    log.getLogger().addHandler(
        log.FileHandler(os.path.join(config["run_dir"], "log.txt"), mode="w")
    )
    log.info(str(config))

    # random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # np.random.seed(0)

    ### estimator
    if config["mols_ref"]:
        config["mols_ref"] = load_mols(config["data_dir"], config["mols_ref"])
    discriminator = (
        Discriminator(config).to(config["device"])
        if "adv" in config["objectives"]
        else None
    )
    estimator = Estimator(config, discriminator)

    for run in range(config["num_runs"]):
        run_dir = os.path.join(config["run_dir"], "run_%02d" % run)
        log.info("Run %02d: ======================" % run)

        ### proposal
        editor = (
            BasicEditor(config).to(config["device"])
            if not config["proposal"] == "random"
            else None
        )
        if config["editor_dir"] is not None:  # load pre-trained editor
            path = os.path.join(
                config["root_dir"], config["editor_dir"], "model_best.pt"
            )
            editor.load_state_dict(
                torch.load(path, map_location=torch.device(config["device"]))
            )
            print("successfully loaded editor model from %s" % path)
        if config["proposal"] == "random":
            proposal = Proposal_Random(config)
        elif config["proposal"] == "editor":
            proposal = Proposal_Editor(config, editor)
        elif config["proposal"] == "mix":
            proposal = Proposal_Mix(config, editor)
        else:
            raise NotImplementedError

        ### sampler
        if config["sampler"] == "re":
            sampler = Sampler_Recursive(config, proposal, estimator)
        elif config["sampler"] == "sa":
            sampler = Sampler_SA(config, proposal, estimator)
        elif config["sampler"] == "mh":
            sampler = Sampler_MH(config, proposal, estimator)
        else:
            raise NotImplementedError

        ### sampling
        if config["mols_init"]:
            mols = load_mols(config["data_dir"], config["mols_init"])
            mols = random.choices(mols, k=config["num_mols"])
            mols_init = mols[: config["num_mols"]]
        else:
            mols_init = [Chem.MolFromSmiles("CC") for _ in range(config["num_mols"])]

        sampler.sample(run_dir, mols_init)
