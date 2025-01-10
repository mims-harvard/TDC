import numpy as np
from typing import List, Tuple

from ...utils.load import pd_load, download_wrapper


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: str = "<cls>",
) -> List[Tuple]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_names, counts) of non zero gene expressions.
    """
    download_wrapper("scgpt_vocab", "./data", ["scgpt_vocab"])
    vocab_map = pd_load("scgpt_vocab", "./data")
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)}).")

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
        if return_pt:
            import torch
            genes = torch.tensor([vocab_map.get(x, 0) for x in genes],
                                 dtype=torch.int64)
            values = torch.from_numpy(values).float()
        tokenized_data.append((genes, values))
    return tokenized_data


class scGPTTokenizer:

    def __init__(self):
        pass

    @classmethod
    def tokenize_cell_vectors(cls, data, gene_names):
        """
        Tokenizing single-cell gene expression vectors formatted as anndata types
        """
        return tokenize_batch(data, gene_names)
