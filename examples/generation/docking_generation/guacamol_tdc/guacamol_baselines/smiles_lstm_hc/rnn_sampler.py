import torch

from .action_sampler import ActionSampler
from .rnn_model import SmilesRnn
from .smiles_char_dict import SmilesCharDictionary


class SmilesRnnSampler:
    """
    Samples molecules from an RNN smiles language model
    """

    def __init__(self, device: str, batch_size=64) -> None:
        """
        Args:
            device: cpu | cuda
            batch_size: number of concurrent samples to generate
        """
        self.device = device
        self.batch_size = batch_size
        self.sd = SmilesCharDictionary()

    def sample(self, model: SmilesRnn, num_to_sample: int, max_seq_len=100):
        """

        Args:
            model: RNN to sample from
            num_to_sample: number of samples to produce
            max_seq_len: maximum length of the samples
            batch_size: number of concurrent samples to generate

        Returns: a list of SMILES string, with no beginning nor end symbols

        """
        sampler = ActionSampler(
            max_batch_size=self.batch_size,
            max_seq_length=max_seq_len,
            device=self.device,
        )

        model.eval()
        with torch.no_grad():
            indices = sampler.sample(model, num_samples=num_to_sample)
            return self.sd.matrix_to_smiles(indices)
