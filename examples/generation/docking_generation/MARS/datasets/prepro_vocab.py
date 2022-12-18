import os
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem, RDLogger

from .utils import load_mols
from ..common.chem import break_bond, Arm, Skeleton

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="MARS/data")
    parser.add_argument("--mols_file", type=str, default="zinc.txt")
    parser.add_argument(
        "--vocab_name",
        type=str,
        default="zinc",
    )
    parser.add_argument("--max_size", type=int, default=10, help="max size of arm")
    args = parser.parse_args()

    ### load data
    mols = load_mols(args.data_dir, args.mols_file)

    ### drop arms
    arms, cnts, smiles2idx = [], [], {}
    for mol in tqdm(mols):
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if not bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                continue
            try:
                skeleton, arm = break_bond(mol, u, v)
            except ValueError:
                continue

            for reverse in [False, True]:
                if reverse is True:
                    tmp = arm
                    arm = Arm(skeleton.mol, skeleton.u, skeleton.bond_type)
                    skeleton = Skeleton(tmp.mol, tmp.v, tmp.bond_type)

                # functional group check
                mark = False
                if not skeleton.mol.GetAtomWithIdx(skeleton.u).GetAtomicNum() == 6:
                    continue
                if arm.mol.GetNumAtoms() > args.max_size:
                    continue
                for atom in arm.mol.GetAtoms():
                    if not atom.GetAtomicNum() == 6:
                        mark = True
                        break
                for bond in arm.mol.GetBonds():
                    if mark:
                        break
                    if (
                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                        or bond.GetBondType() == Chem.rdchem.BondType.TRIPLE
                    ):
                        mark = True
                        break
                if not mark:
                    continue

                smiles = Chem.MolToSmiles(arm.mol, rootedAtAtom=arm.v)
                if smiles.startswith("CC"):
                    continue
                if smiles2idx.get(smiles) is None:
                    smiles2idx[smiles] = len(arms)
                    arms.append(arm)
                    cnts.append(1)
                else:
                    cnts[smiles2idx[smiles]] += 1

    ### save arms
    idx2smiles = {idx: smiles for smiles, idx in smiles2idx.items()}
    indices = sorted(range(len(cnts)), key=lambda i: cnts[i], reverse=True)
    arms = [arms[i] for i in indices]
    cnts = [cnts[i] for i in indices]
    vocab_dir = os.path.join(args.data_dir, "vocab_%s" % args.vocab_name)
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "arms.pkl"), "wb") as f:
        pickle.dump(arms, f)
    with open(os.path.join(vocab_dir, "arms.smiles"), "w") as f:
        for i, cnt in zip(indices, cnts):
            smiles = idx2smiles[i]
            f.write("%i\t%s\n" % (cnt, smiles))
