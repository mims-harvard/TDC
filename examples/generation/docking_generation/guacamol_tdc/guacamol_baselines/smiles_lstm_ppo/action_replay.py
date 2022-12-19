from typing import Type, Tuple

import torch
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic
from smiles_lstm_hc.rnn_utils import rnn_start_token_vector


class ActionReplay(object):
    """
    Action replay for policy-based RL algorithms.
    Given some actions sampled from a RNN model, will calculate the log probabilities and entropy.
    """

    def __init__(
        self, max_batch_size, device, distribution_cls: Type[Distribution] = None
    ) -> None:
        """
        Args:
            max_batch_size: Max. batch size
            device: cuda | cpu
            distribution_cls: distribution type to sample from. If None, will be a multinomial distribution.
        """
        self.max_batch_size = max_batch_size
        self.device = device

        self.distribution_cls = (
            Categorical if distribution_cls is None else distribution_cls
        )

    def replay(
        self,
        model: SmilesRnnActorCritic,
        prior: SmilesRnnActorCritic,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an array of action sequences, calculate the corresponding log probabilities and entropy.

        Args:
            model: Smiles RNN model
            prior: Initial Smiles RNN model pretrained on ChEMBL
            actions: characters to feed into the model (batch_size x max_seq_length)

        Returns:
            tuple:
            - log probabilities (batch_size x max_seq_length)
            - entropy (batch_size x max_seq_length)
            - kl div between model and prior (batch_size x max_seq_length)
        """

        num_samples, max_seq_length = actions.size()

        # Round up division to get the number of batches that are necessary:
        number_batches = (num_samples + self.max_batch_size - 1) // self.max_batch_size
        remaining_samples = num_samples

        log_probs = torch.zeros(num_samples, max_seq_length).to(self.device)
        values = torch.zeros(num_samples, max_seq_length).to(self.device)
        entropies = torch.zeros(num_samples, max_seq_length).to(self.device)
        kl_divs = torch.zeros(num_samples, max_seq_length).to(self.device)

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            action_batch = actions[batch_start:batch_end, :]

            lp, v, e, kl_div = self._replay_batch(model, prior, action_batch)

            log_probs[batch_start:batch_end, :] = lp
            values[batch_start:batch_end, :] = v
            entropies[batch_start:batch_end, :] = e
            kl_divs[batch_start:batch_end, :] = kl_div

            batch_start += batch_size
            remaining_samples -= batch_size

        return log_probs, values, entropies, kl_divs

    def _replay_batch(
        self,
        model: SmilesRnnActorCritic,
        prior: SmilesRnnActorCritic,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of action sequences, calculate the corresponding log probabilities, values, entropy, kl_div

        Returns:
            tuple:
            - log probabilities (batch_size x max_seq_length)
            - entropies (batch_size x max_seq_length)
            - kl divergences between model and prior (batch_size x max_seq_length)
            - values from critic (batch_size x max_seq_length)
        """
        batch_size = action_batch.size()[0]
        max_seq_length = action_batch.size()[1]

        hidden = model.smiles_rnn.init_hidden(batch_size, self.device)
        inp = rnn_start_token_vector(batch_size, self.device)

        hidden_prior = prior.smiles_rnn.init_hidden(batch_size, self.device)
        inp_prior = rnn_start_token_vector(batch_size, self.device)

        log_probs = torch.zeros(size=action_batch.size(), device=self.device)
        values = torch.zeros(size=action_batch.size(), device=self.device)
        entropies = torch.zeros(size=action_batch.size(), device=self.device)
        kl_divs = torch.zeros(size=action_batch.size(), device=self.device)

        for char in range(max_seq_length):
            actor_output, critic_output, hidden = model(inp, hidden)
            prob = F.softmax(actor_output, dim=2)

            distribution = self.distribution_cls(probs=prob)
            action = action_batch[:, char].unsqueeze(-1)

            inp = action

            with torch.no_grad():
                actor_output_prior, _, hidden_prior = prior(inp_prior, hidden_prior)
                prob_prior = F.softmax(actor_output_prior, dim=2)
                inp_prior = action

            kl_div = prob.mul(prob.log().sub(prob_prior.log())).mean(dim=2)

            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            value = critic_output

            log_probs[:, char] = log_prob.squeeze()
            entropies[:, char] = entropy.squeeze()
            kl_divs[:, char] = kl_div.squeeze()
            values[:, char] = value.squeeze()

        return log_probs, values, entropies, kl_divs
