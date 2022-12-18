import os
import sys
import rdkit
import torch
import argparse
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

from .datasets.utils import load_mols

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

ROOT_DIR = "MARS/"
parser = argparse.ArgumentParser()
parser.add_argument("--unique", action="store_true")
parser.add_argument("--num_mols", type=int, default=5000)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--run_dir", type=str, default=None)
parser.add_argument("--mols_ref", type=str, default=None)
args = parser.parse_args()

"""
evaluate generated molecules (in the format of .tsv)

success standards:
    qed >= 0.6
    sa <= 4 (sa >= 0.67)
    predicted property >= 0.5
"""

### reference molecules
if args.mols_ref:
    true_mols = load_mols(os.path.join(ROOT_DIR, "data"), args.mols_ref)
    true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]
else:
    true_fps = None

### read molecules
if args.run_dir:
    mol_lines = []
    for path in os.listdir(args.run_dir):
        if not path.startswith("run"):
            continue
        run = int(path.split("_")[-1])
        if args.num_runs and run >= args.num_runs:
            continue
        with open(os.path.join(args.run_dir, path, "mols.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split("\t") for line in lines]
            titles = lines[1]
            mol_lines += lines[-args.num_mols :]
else:
    lines = [line.strip().split("\t") for line in sys.stdin]
    titles = lines[0]
    mol_lines = lines[1:]

if args.unique:
    smiles_set = set()
    mol_lines, tmp_lines = [], mol_lines
    for line in tmp_lines:
        smiles = line[-1]
        if smiles in smiles_set:
            continue
        smiles_set.add(smiles)
        mol_lines.append(line)

pred_mols = []
for line in mol_lines:  # lines[0] for table titles
    smiles = line[-1]
    scores = [float(v) for v in line[:-1]]

    success = True
    for obj, score in zip(titles[:-1], scores):
        if obj == "score":
            continue
        elif obj == "qed":
            success = success and score >= 0.6
        elif obj == "sa":
            success = success and score >= 0.67
        else:
            success = success and score >= 0.5
    if success:
        pred_mols.append(Chem.MolFromSmiles(smiles))

success = 1.0 * len(pred_mols) / len(mol_lines)
print("success: %.4f (%i)" % (success, len(mol_lines)))
pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

### novelty
if true_fps:
    n_novel = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) >= 0.4:
            n_novel += 1
    novelty = 1.0 - 1.0 * n_novel / (len(pred_mols) + 1e-6)
else:
    novelty = 1.0
novelty = min(novelty, 1.0)
print("novelty: %.4f" % novelty)

### diversity
similarity = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
    similarity += sum(sims)
n = len(pred_fps)
n_pairs = n * (n - 1) / 2
diversity = 1 - similarity / (n_pairs + 1e-6)
diversity = min(diversity, 1.0)
print("diversity: %.4f" % diversity)

prod = success * novelty * diversity
print("prod: %.4f" % prod)
