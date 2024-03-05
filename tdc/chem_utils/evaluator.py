# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import numpy as np
import os

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
except:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! ")


def single_molecule_validity(smiles):
    """Evaluate the chemical validity of a single molecule in terms of SMILES string

    Args:
      smiles: str, SMILES string.

    Returns:
      Boolean: if the SMILES string is a valid molecule

    """
    if smiles.strip() == "":
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    return True


def validity(list_of_smiles):
    valid_list_smiles = list(filter(single_molecule_validity, list_of_smiles))
    return 1.0 * len(valid_list_smiles) / len(list_of_smiles)


def canonicalize(smiles):
    """Convert SMILES into canonical form.

    Args:
      smiles: str, SMILES string

    Returns:
      smiles: str, canonical SMILES string.

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def unique_lst_of_smiles(list_of_smiles):
    canonical_smiles_lst = list(map(canonicalize, list_of_smiles))
    canonical_smiles_lst = list(
        filter(lambda x: x is not None, canonical_smiles_lst))
    canonical_smiles_lst = list(set(canonical_smiles_lst))
    return canonical_smiles_lst


def uniqueness(list_of_smiles):
    """Evaluate the uniqueness of a list of SMILES string, i.e., the fraction of unique molecules among a given list.

    Args:
      list_of_smiles: list (of SMILES string)

    Returns:
      uniqueness: float
    """
    canonical_smiles_lst = unique_lst_of_smiles(list_of_smiles)
    return 1.0 * len(canonical_smiles_lst) / len(list_of_smiles)


def novelty(generated_smiles_lst, training_smiles_lst):
    """Evaluate the novelty of set of generated smiles using list of training smiles as reference.
    Novelty is defined as the fraction of generated molecules that doesn't appear in the training set.

    Args:
      generated_smiles_lst: list (of SMILES string), which are generated.
      training_smiles_lst: list (of SMILES string), which are used for training.

    Returns:
      novelty: float
    """
    generated_smiles_lst = unique_lst_of_smiles(generated_smiles_lst)
    training_smiles_lst = unique_lst_of_smiles(training_smiles_lst)
    novel_ratio = (sum(
        [1 if i in training_smiles_lst else 0 for i in generated_smiles_lst]) *
                   1.0 / len(generated_smiles_lst))
    return 1 - novel_ratio


def diversity(list_of_smiles):
    """Evaluate the internal diversity of a set of molecules. The internbal diversity is defined as the average pairwise
      Tanimoto distance between the Morgan fingerprints.

    Args:
      list_of_smiles: list of SMILES strings

    Returns:
      div: float
    """
    list_of_unique_smiles = unique_lst_of_smiles(list_of_smiles)
    list_of_mol = [
        Chem.MolFromSmiles(smiles) for smiles in list_of_unique_smiles
    ]
    list_of_fp = [
        AllChem.GetMorganFingerprintAsBitVect(mol,
                                              2,
                                              nBits=2048,
                                              useChirality=False)
        for mol in list_of_mol
    ]
    avg_lst = []
    for idx, fp in enumerate(list_of_fp):
        for fp2 in list_of_fp[idx + 1:]:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            ### option I
            distance = 1 - sim
            ### option II
            # distance = -np.log2(sim)
            avg_lst.append(distance)
    div = np.mean(avg_lst)
    return div


########################################
######## KL divergence ########


def _calculate_pc_descriptors(smiles, pc_descriptors):
    """Calculate Physical Chemical descriptors of a single SMILES (internal function).

    Args:
      list_of_smiles: SMILES strings
      pc_descriptors: list of strings, names of descriptors to calculate

    Returns:
      descriptros: list of float
    """
    from rdkit.ML.Descriptors import MoleculeDescriptors

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(pc_descriptors)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    _fp = calc.CalcDescriptors(mol)
    _fp = np.array(_fp)
    mask = np.isfinite(_fp)
    if (mask == 0).sum() > 0:
        logger.warning(f"{smiles} contains an NAN physchem descriptor")
        _fp[~mask] = 0

    return _fp


def calculate_pc_descriptors(smiles, pc_descriptors):
    """Calculate Physical Chemical descriptors of a list of molecules.

    Args:
      list_of_smiles: list of SMILES strings
      pc_descriptors: list of strings, names of descriptors to calculate

    Returns:
      descriptros: list of float
    """
    output = []

    for i in smiles:
        d = _calculate_pc_descriptors(i, pc_descriptors)
        if d is not None:
            output.append(d)

    return np.array(output)


def continuous_kldiv(X_baseline: np.array, X_sampled: np.array) -> float:
    """calculate KL divergence for two numpy arrays, conitnuous version.

    Args:
      X_baseline: numpy array
      X_sampled: numpy array

    Returns:
      KL divergence: float
    """
    X_baseline += 1e-5
    X_sampled += 1e-5
    from scipy.stats import entropy, gaussian_kde

    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(
        np.hstack([X_baseline, X_sampled]).min(),
        np.hstack([X_baseline, X_sampled]).max(),
        num=1000,
    )
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)


def discrete_kldiv(X_baseline: np.array, X_sampled: np.array) -> float:
    """calculate KL divergence for two numpy arrays, discrete version.

    Args:
      X_baseline: numpy array
      X_sampled: numpy array

    Returns:
      KL divergence: float
    """
    from scipy.stats import entropy
    from scipy import histogram

    P, bins = histogram(X_baseline, bins=10, density=True)
    P += 1e-10
    Q, _ = histogram(X_sampled, bins=bins, density=True)
    Q += 1e-10

    return entropy(P, Q)


def get_fingerprints(mols, radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols
    ]


def get_mols(smiles_list):
    """Convert SMILES strings to RDKit RDMol objects.

    Args:
      list_of_smiles: list of SMILES strings

    Returns:
      mols: list of RDKit RDMol objects
    """
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)


def calculate_internal_pairwise_similarities(smiles_list):
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Args:
        smiles_list: list of str

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(f"Calculating internal similarity on large set of "
                       f"SMILES strings ({len(smiles_list)})")

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities


def kl_divergence(generated_smiles_lst, training_smiles_lst):
    """Evaluate the KL divergence of set of generated smiles using list of training smiles as reference.
    KL divergence is defined as the averaged KL divergence of a set of physical chemical descriptors
    between a set of generated molecules and a set of training molecules.

    Args:
      generated_smiles_lst: list (of SMILES string), which are generated.
      training_smiles_lst: list (of SMILES string), which are used for training.

    Returns:
      KL divergence: float
    """
    pc_descriptor_subset = [
        "BertzCT",
        "MolLogP",
        "MolWt",
        "TPSA",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "NumAliphaticRings",
        "NumAromaticRings",
    ]

    def canonical(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol,
                                    isomericSmiles=True)  ### todo double check
        else:
            return None

    generated_lst_mol = list(map(canonical, generated_smiles_lst))
    training_lst_mol = list(map(canonical, training_smiles_lst))
    filter_out_func = lambda x: x is not None
    generated_lst_mol = list(filter(filter_out_func, generated_lst_mol))
    training_lst_mol = list(filter(filter_out_func, training_lst_mol))

    d_sampled = calculate_pc_descriptors(generated_lst_mol,
                                         pc_descriptor_subset)
    d_chembl = calculate_pc_descriptors(training_lst_mol, pc_descriptor_subset)

    kldivs = {}
    for i in range(4):
        kldiv = continuous_kldiv(X_baseline=d_chembl[:, i],
                                 X_sampled=d_sampled[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    # ... and for the int valued ones.
    for i in range(4, 9):
        kldiv = discrete_kldiv(X_baseline=d_chembl[:, i],
                               X_sampled=d_sampled[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    # pairwise similarity

    chembl_sim = calculate_internal_pairwise_similarities(training_lst_mol)
    chembl_sim = chembl_sim.max(axis=1)

    sampled_sim = calculate_internal_pairwise_similarities(generated_lst_mol)
    sampled_sim = sampled_sim.max(axis=1)

    kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim,
                                     X_sampled=sampled_sim)
    kldivs["internal_similarity"] = kldiv_int_int
    """
        # for some reason, this runs into problems when both sets are identical.
        # cross_set_sim = calculate_pairwise_similarities(self.training_set_molecules, unique_molecules)
        # cross_set_sim = cross_set_sim.max(axis=1)
        #
        # kldiv_ext = discrete_kldiv(chembl_sim, cross_set_sim)
        # kldivs['external_similarity'] = kldiv_ext
        # kldiv_sum += kldiv_ext
  """

    # Each KL divergence value is transformed to be in [0, 1].
    # Then their average delivers the final score.
    partial_scores = [np.exp(-score) for score in kldivs.values()]
    score = sum(partial_scores) / len(partial_scores)
    return score


def fcd_distance_tf(generated_smiles_lst, training_smiles_lst):
    """Evaluate FCD distance between generated smiles set and training smiles set using tensorflow.

    Args:
      generated_smiles_lst: list (of SMILES string), which are generated.
      training_smiles_lst: list (of SMILES string), which are used for training.

    Returns:
      fcd_distance: float
    """
    import pkgutil, tempfile, os

    if "chemnet" not in globals().keys():
        global chemnet
        ### _load_chemnet
        chemnet_model_filename = "ChemNet_v0.13_pretrained.h5"
        model_bytes = pkgutil.get_data("fcd", chemnet_model_filename)
        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, chemnet_model_filename)
        with open(model_path, "wb") as f:
            f.write(model_bytes)
        chemnet = fcd.load_ref_model(model_path)
        # _load_chemnet

    def _calculate_distribution_statistics(chemnet, molecules):
        sample_std = fcd.canonical_smiles(molecules)
        gen_mol_act = fcd.get_predictions(chemnet, sample_std)

        mu = np.mean(gen_mol_act, axis=0)
        cov = np.cov(gen_mol_act.T)
        return mu, cov

    mu_ref, cov_ref = _calculate_distribution_statistics(
        chemnet, training_smiles_lst)
    mu, cov = _calculate_distribution_statistics(chemnet, generated_smiles_lst)

    FCD = fcd.calculate_frechet_distance(mu1=mu_ref,
                                         mu2=mu,
                                         sigma1=cov_ref,
                                         sigma2=cov)
    fcd_distance = np.exp(-0.2 * FCD)
    return fcd_distance


def fcd_distance_torch(generated_smiles_lst, training_smiles_lst):
    """Evaluate FCD distance between generated smiles set and training smiles set using PyTorch.

    Args:
      generated_smiles_lst: list (of SMILES string), which are generated.
      training_smiles_lst: list (of SMILES string), which are used for training.

    Returns:
      fcd_distance: float
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    from fcd_torch import FCD

    fcd = FCD(device="cpu", n_jobs=8)
    fcd_distance = fcd(generated_smiles_lst, training_smiles_lst)
    return fcd_distance


def fcd_distance(generated_smiles_lst, training_smiles_lst):
    """Evaluate FCD distance between generated smiles set and training smiles set.

    Args:
      generated_smiles_lst: list (of SMILES string), which are generated.
      training_smiles_lst: list (of SMILES string), which are used for training.

    Returns:
      fcd_distance: float
    """
    try:
        import tensorflow, fcd

        global fcd
    except:
        try:
            import torch, fcd_torch

            return fcd_distance_torch(generated_smiles_lst, training_smiles_lst)
        except:
            raise ImportError(
                "Please install fcd by 'pip install FCD' (for Tensorflow backend) \
                                            or 'pip install fcd_torch' (for PyTorch backend)!"
            )
    return fcd_distance_tf(generated_smiles_lst, training_smiles_lst)
