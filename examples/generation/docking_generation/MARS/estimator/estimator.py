import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader

# from .scorer import chemprop_scorer
from .scorer.scorer import get_scores
from ..common.chem import mol_to_dgl
from ..datasets.datasets import GraphDataset


class Estimator:
    def __init__(self, config, discriminator=None, mols_ref=None):
        """
        @params:
            config (dict): configurations
            discriminator (nn.Module): adversarial discriminator
        """
        # chemprop_scorer.device = config['device']
        self.discriminator = discriminator
        self.batch_size = config["batch_size"]
        self.objectives = config["objectives"]
        self.fps_ref = (
            [
                AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048)
                for x in config["mols_ref"]
            ]
            if config["mols_ref"]
            else None
        )

    def get_scores(self, mols, smiles2score=None):
        """
        @params:
            mols: molecules to estimate score
        @return:
            dicts (list): list of score dictionaries
        """
        if "nov" in self.objectives or "div" in self.objectives:
            fps_mols = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]

        dicts = [{} for _ in mols]

        for obj in self.objectives:
            if obj == "adv":
                continue
            if obj == "nov":
                for i, fp in enumerate(fps_mols):
                    sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_ref)
                    dicts[i][obj] = 1.0 - max(sims)
                continue
            if obj == "div":
                for i, fp in enumerate(fps_mols):
                    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_mols)
                    dicts[i][obj] = 1.0 - 1.0 * sum(sims) / len(fps_mols)
                continue
            scores = get_scores(obj, mols, smiles2score)
            for i, mol in enumerate(mols):
                dicts[i][obj] = scores[i]

        if "adv" in self.objectives:
            graphs = [mol_to_dgl(mol) for mol in mols]
            dataset = GraphDataset(graphs)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, collate_fn=GraphDataset.collate_fn
            )

            preds = []
            for batch in loader:
                with torch.no_grad():
                    pred = self.discriminator(batch)  # (batch_size, 2)
                    pred = F.softmax(pred, dim=1)  # (batch_size, 2)
                preds.append(pred[:, 1])  # (batch_size,)
            preds = torch.cat(preds, dim=0).tolist()  # (num_mols,)
            for i, pred in enumerate(preds):
                dicts[i]["adv"] = pred
        return dicts
