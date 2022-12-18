from typing import List

import numpy as np
import logging
import torch
import copy
import torch.optim as optim
from functools import total_ordering
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from smiles_lstm_hc.action_sampler import ActionSampler
from smiles_lstm_ppo.action_replay import ActionReplay
from smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic
from guacamol.scoring_function import ScoringFunction
from smiles_lstm_ppo.molecule_batch import MoleculeBatch
from smiles_lstm_ppo.running_reward import RunningReward
from smiles_lstm_hc.smiles_char_dict import SmilesCharDictionary

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score

    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)

    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)


class PPOTrainer(object):
    """
    Class to train a SMILES-generating model with proximal policy optimization.

    Updates are done after calculating scores for batches of molecules, by giving each action leading
    to a molecule the same reward (= molecule score).

    An entropy term is added to promote diversification of the generated molecules.

    A KL divergence term between pretrained and current policy is added to keep the policy close to the
    ChEMBL chemical space

    A critic / value function is added and is subtracted from the reward as a baseline.

    Important arguments:
        self.episode_size: The number of molecules sampled by the policy at the start of a series of ppo updates
        self.batch_size: The number of molecules used for each ppo update. Ideally self.episode_size should be a
                         multiple of self.batch_size
        self.ppo_epochs: The number of updates in one series of ppo updates
        self.clip_param: The parameter which determines how far the new policy is from the old one
    """

    def __init__(
        self,
        model: SmilesRnnActorCritic,
        optimization_objective: ScoringFunction,
        max_seq_length,
        device,
        num_epochs,
        clip_param,
        batch_size,
        episode_size,
        entropy_weight=1.0,
        kl_div_weight=5.0,
    ) -> None:
        self.model = model
        self.prior = copy.deepcopy(model).to(device)
        self.optimization_objective = optimization_objective
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.episode_size = episode_size
        self.sd = SmilesCharDictionary()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.print_every = 10
        self.entropy_weight = entropy_weight
        self.kl_div_weight = kl_div_weight
        self.sampler = ActionSampler(
            max_seq_length=max_seq_length, device=device, max_batch_size=batch_size
        )
        self.action_replay = ActionReplay(device=device, max_batch_size=batch_size)
        self.running_reward = RunningReward(keep_factor=0.99)
        self.normalize_advantages = True
        self.smiles_history: List[
            OptResult
        ] = []  # Necessary because MoleculeGenerator keeps history, may need
        # refactoring later on.
        self.ppo_epochs = self.episode_size // self.batch_size
        self.clip_param = clip_param

    def train(self):
        """
        self.train calls self.train_ppo_epochs N times where N = self.num_epochs.
        Each time self.train_ppo_epochs is called it does M ppo updates where M = self.ppo_epochs.
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            self.train_ppo_epochs(epoch)

    def train_ppo_epochs(self, epoch):

        """
        Does one series of ppo updates
        """

        # Samples a set of molecules of size self.episode_size (along with rewards, actions, etc)
        (
            rewards,
            advantages,
            actions,
            old_log_probs,
            smiles,
        ) = self.sample_and_process_episode()
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.episode_size)),
            self.batch_size,
            drop_last=False,
        )

        # Randomly samples batches of size self.batch_size and then performs one ppo_update.
        # This is repeated self.ppo_epochs times.
        self.model.train()
        for indices in sampler:
            rewards_batch = rewards.view(-1, rewards.size(-1))[indices]
            actions_batch = actions.view(-1, actions.size(-1))[indices]
            old_log_probs_batch = old_log_probs.view(-1, old_log_probs.size(-1))[
                indices
            ]
            smiles_batch = list(np.array(smiles)[indices])
            advantages_batch = advantages.view(-1, advantages.size(-1))[indices]

            (
                log_probs_batch,
                values_batch,
                entropies_batch,
                kl_divs_batch,
            ) = self.action_replay.replay(
                model=self.model, prior=self.prior, actions=actions_batch
            )

            ratio = torch.exp(log_probs_batch - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * advantages_batch
            )
            policy_loss = -torch.min(surr1, surr2).mean()  # standard ppo policy loss.

            value_loss = self._calculate_value_loss(
                smiles_batch, values_batch, rewards_batch
            )

            entropy_loss = self._calculate_entropy_loss(smiles_batch, entropies_batch)

            kl_div_loss = self._calculate_kl_div_loss(smiles_batch, kl_divs_batch)

            loss = policy_loss + value_loss + entropy_loss + kl_div_loss

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        self._print_stats(epoch=epoch, smiles=smiles)

    def sample_and_process_episode(self):
        """
        Samples a set of molecules of size self.episode_size

        Returns:
            rewards: Tensor of size episode_size x max_seq_len
            advantages: Tensor of size episode_size x max_seq_length
            actions: Tensor of size episode_size x max_seq_len
            old_log_probs: Tensor of size episode_size x max_seq_len
            smiles: List of smiles of len episode_size
        """

        self.model.eval()
        with torch.no_grad():
            actions = self.sampler.sample(
                model=self.model.smiles_rnn, num_samples=self.episode_size
            )
            old_log_probs, old_values, _, _ = self.action_replay.replay(
                model=self.model, prior=self.prior, actions=actions
            )

            smiles = self.sd.matrix_to_smiles(actions)
            scores = self.optimization_objective.score_list(smiles)
            scores = [
                OptResult(smiles=smiles, score=score)
                for smiles, score in zip(smiles, scores)
            ]
            self.smiles_history.extend(scores)

            self._update_running_reward(scores=scores)

            rewards = self._compute_rewards(actions, scores, smiles)
            advantages = rewards - old_values

            if self.normalize_advantages:
                advantages = self._normalize_advantages(advantages)

        return rewards, advantages, actions, old_log_probs, smiles

    def _calculate_value_loss(self, smiles, values, rewards):
        """
        Calculate the value function contribution to the loss, but take into consideration only the non-padding
        characters!
        """
        count = 0
        value_loss_sum = 0
        for i in range(self.batch_size):
            n_characters = len(smiles[i]) + 1
            value_loss_sum += (
                (rewards[i, :n_characters] - values[i, :n_characters]).pow(2).sum()
            )
            count += n_characters
        value_loss = value_loss_sum / count
        return value_loss

    def _calculate_entropy_loss(self, smiles, entropies):

        """
        Calculate the entropy contribution to the loss, but take into consideration only the non-padding characters!
        """
        count = 0
        entropy = 0
        for i in range(self.batch_size):
            n_characters = len(smiles[i]) + 1
            entropy += entropies[i, :n_characters].sum()
            count += n_characters
        entropy_mean = entropy / count
        return -entropy_mean * self.entropy_weight

    def _calculate_kl_div_loss(self, smiles, kl_divs):
        """
        Calculate the kl div contribution to the loss, but take into consideration only the non-padding characters!
        """
        count = 0
        kl_div = 0
        for i in range(self.batch_size):
            n_characters = len(smiles[i]) + 1
            kl_div += kl_divs[i, :n_characters].sum()
            count += n_characters
        kl_div_mean = kl_div / count
        return kl_div_mean * self.kl_div_weight

    def _compute_rewards(self, actions, scores, smiles):
        rewards = torch.zeros(size=actions.size(), device=actions.device)
        for i in range(self.episode_size):
            rewards_len = len(smiles[i]) + 1
            rewards[i, :rewards_len] = scores[i].score
        return rewards

    def _normalize_advantages(self, advantages):
        eps = 1e-5  # for numerical stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        return advantages

    def _update_running_reward(self, scores):
        s = [v.score for v in scores]
        mean_score = sum(s) / len(s)
        self.running_reward.update(mean_score)

    def _print_stats(self, epoch, smiles):
        if epoch % self.print_every != 0:
            return

        mol_batch = MoleculeBatch(smiles)

        logger.info(
            f"epoch: {epoch:7d} | "
            f"current_reward: {self.running_reward.last_added:.3f} | "
            f"running_reward: {self.running_reward.value:.3f} | "
            f"valid_ratio: {mol_batch.ratio_valid:.3f} | "
            f"unique_ratio: {mol_batch.ratio_unique_among_valid:.3f}"
        )
