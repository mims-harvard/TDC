import os
import numpy as np
from rdkit import Chem
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers


models = {}
device = None
ROOT_DIR = 'MARS/estimator/scorer'

class chemprop_model():
    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    self.scaler, self.features_scaler = load_scalers(fname)
                    self.train_args = load_args(fname)
                    model = load_checkpoint(fname, device=device)
                    self.checkpoints.append(model)

    def __call__(self, smiles, batch_size=200):
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        # if self.train_args.features_scaling:
        #     test_data.normalize_features(self.features_scaler)
        
        sum_preds = np.zeros((len(test_data), 1))
        data_loader = MoleculeDataLoader(test_data, batch_size=batch_size)
        for model in self.checkpoints:
            model_preds = predict(
                model=model,
                data_loader=data_loader,
                scaler=self.scaler
            )
            sum_preds += np.array(model_preds)

        # Ensemble predictions
        avg_preds = sum_preds / len(self.checkpoints)
        avg_preds = avg_preds.squeeze(-1).tolist()

        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]
        return np.array(full_preds, dtype=np.float32)


def get_scores(task, mols):
    model = models.get(task)
    if model is None:
        if task == 'chemprop_ecoli':
            model = chemprop_model(os.path.join(ROOT_DIR, 'chemprop_ckpt/ecoli'))
        elif task == 'chemprop_sars':
            model = chemprop_model(os.path.join(ROOT_DIR, 'chemprop_ckpt/sars_balanced'))
        elif task == 'chemprop_sars_cov_2':
            model = chemprop_model(os.path.join(ROOT_DIR, 'chemprop_ckpt/sars_cov_2'))
        else: raise NotImplementedError
        models[task] = model

    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    scores = model(smiles).tolist()
    return scores