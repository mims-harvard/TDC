import logging
import os
from glob import glob
from time import time
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .rnn_utils import save_model, time_since

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmilesRnnTrainer:

    def __init__(self, model, criteria, optimizer, device, log_dir=None, clip_gradients=True) -> None:
        self.model = model.to(device)
        self.criteria = [c.to(device) for c in criteria]
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.clip_gradients = clip_gradients

    def process_batch(self, batch):

        # ship data to device
        inp, tgt = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)

        # process data
        batch_size = inp.size(0)
        hidden = self.model.init_hidden(inp.size(0), self.device)
        output, hidden = self.model(inp, hidden)
        output = output.view(output.size(0) * output.size(1), -1)
        loss = self.criteria[0](output, tgt.view(-1))
        return loss, batch_size

    def train_on_batch(self, batch):

        # setup model for training
        self.model.train()
        self.model.zero_grad()

        # forward / backward
        loss, size = self.process_batch(batch)
        loss.backward()

        # optimize
        if self.clip_gradients:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), size

    def test_on_batch(self, batch):

        # setup model for evaluation
        self.model.eval()

        # forward
        loss, size = self.process_batch(batch)

        return loss.item(), size

    def validate(self, data_loader, n_molecule):
        """Runs validation and reports the average loss"""
        valid_losses = []
        with torch.no_grad():
            for batch in data_loader:
                loss, size = self.test_on_batch(batch)
                valid_losses += [loss]
        return np.array(valid_losses).mean()

    def train_extra_log(self, n_molecules):
        pass

    def valid_extra_log(self, n_molecules):
        pass

    def fit(self, training_data, test_data, n_epochs, batch_size, print_every,
            valid_every, num_workers=0):
        training_round = _ModelTrainingRound(self, training_data, test_data, n_epochs, batch_size, print_every,
                                             valid_every, num_workers)
        return training_round.run()


class _ModelTrainingRound:
    """
    Performs one round of model training.

    Is a separate class from ModelTrainer to allow for more modular functions without too many parameters.
    This class is not to be used outside of ModelTrainer.
    """
    class EarlyStopNecessary(Exception):
        pass

    def __init__(self, model_trainer: SmilesRnnTrainer, training_data, test_data, n_epochs, batch_size, print_every,
                 valid_every, num_workers=0) -> None:
        self.model_trainer = model_trainer
        self.training_data = training_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.valid_every = valid_every
        self.num_workers = num_workers

        self.start_time = time()
        self.unprocessed_train_losses: List[float] = []
        self.all_train_losses: List[float] = []
        self.all_valid_losses: List[float] = []
        self.n_molecules_so_far = 0
        self.has_run = False
        self.min_valid_loss = np.inf
        self.min_avg_train_loss = np.inf

    def run(self):
        if self.has_run:
            raise Exception('_ModelTrainingRound.train() can be called only once.')

        try:
            for epoch_index in range(1, self.n_epochs + 1):
                self._train_one_epoch(epoch_index)

            self._validation_on_final_model()
        except _ModelTrainingRound.EarlyStopNecessary:
            logger.error('Probable explosion during training. Stopping now.')

        self.has_run = True
        return self.all_train_losses, self.all_valid_losses

    def _train_one_epoch(self, epoch_index: int):
        logger.info(f'EPOCH {epoch_index}')

        # shuffle at every epoch
        data_loader = DataLoader(self.training_data,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        epoch_t0 = time()
        self.unprocessed_train_losses.clear()

        for batch_index, batch in enumerate(data_loader):
            self._train_one_batch(batch_index, batch, epoch_index, epoch_t0)

    def _train_one_batch(self, batch_index, batch, epoch_index, train_t0):
        loss, size = self.model_trainer.train_on_batch(batch)

        self.unprocessed_train_losses += [loss]
        self.n_molecules_so_far += size

        # report training progress?
        if batch_index > 0 and batch_index % self.print_every == 0:
            self._report_training_progress(batch_index, epoch_index, epoch_start=train_t0)

        # report validation progress?
        if batch_index >= 0 and batch_index % self.valid_every == 0:
            self._report_validation_progress(epoch_index)

    def _report_training_progress(self, batch_index, epoch_index, epoch_start):
        mols_sec = self._calculate_mols_per_second(batch_index, epoch_start)

        # Update train losses by processing all losses since last time this function was executed
        avg_train_loss = np.array(self.unprocessed_train_losses).mean()
        self.all_train_losses += avg_train_loss
        self.unprocessed_train_losses.clear()

        logger.info(
            'TRAIN | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch|batch : {epoch_index}|{batch_index} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'mols/sec: {mols_sec:.2f} | '
            f'train_loss: {avg_train_loss:.4f}')
        self.model_trainer.train_extra_log(self.n_molecules_so_far)

        self._check_early_stopping_train_loss(avg_train_loss)

    def _calculate_mols_per_second(self, batch_index, epoch_start):
        """
        Calculates the speed so far in the current epoch.
        """
        train_time_in_current_epoch = time() - epoch_start
        processed_batches = batch_index + 1
        molecules_in_current_epoch = self.batch_size * processed_batches
        return molecules_in_current_epoch / train_time_in_current_epoch

    def _report_validation_progress(self, epoch_index):
        avg_valid_loss = self._validate_current_model()

        self._log_validation_step(epoch_index, avg_valid_loss)
        self._check_early_stopping_validation(avg_valid_loss)

        # save model?
        if self.model_trainer.log_dir:
            if avg_valid_loss <= min(self.all_valid_losses):
                self._save_current_model(self.model_trainer.log_dir, epoch_index, avg_valid_loss)

    def _validate_current_model(self):
        """
        Validate the current model.

        Returns: Validation loss.
        """
        test_loader = DataLoader(self.test_data,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        avg_valid_loss = self.model_trainer.validate(test_loader, self.n_molecules_so_far)
        self.all_valid_losses += [avg_valid_loss]
        return avg_valid_loss

    def _log_validation_step(self, epoch_index, avg_valid_loss):
        """
        Log the information about the validation step.
        """
        logger.info(
            'VALID | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch: {epoch_index}/{self.n_epochs} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {avg_valid_loss:.4f}')
        self.model_trainer.valid_extra_log(self.n_molecules_so_far)
        logger.info('')

    def _get_overall_progress(self):
        total_mols = self.n_epochs * len(self.training_data)
        return 100. * self.n_molecules_so_far / total_mols

    def _validation_on_final_model(self):
        """
        Run validation for the final model and save it.
        """
        valid_loss = self._validate_current_model()
        logger.info(
            'VALID | FINAL_MODEL | '
            f'elapsed: {time_since(self.start_time)} | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {valid_loss:.4f}')

        if self.model_trainer.log_dir:
            self._save_model(self.model_trainer.log_dir, 'final', valid_loss)

    def _save_current_model(self, base_dir, epoch, valid_loss):
        """
        Delete previous versions of the model and save the current one.
        """
        for f in glob(os.path.join(base_dir, 'model_*')):
            os.remove(f)

        self._save_model(base_dir, epoch, valid_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'model_{info}_{valid_loss:.3f}'
        logger.info(base_name)
        save_model(self.model_trainer.model, base_dir, base_name)

    def _check_early_stopping_train_loss(self, avg_train_loss):
        """
        This function checks whether the training has exploded by verifying if the avg training loss
        is more than 10 times the minimal loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 10 * self.min_avg_train_loss
        if avg_train_loss > threshold:
            raise _ModelTrainingRound.EarlyStopNecessary()

        # update the min train loss if necessary
        if avg_train_loss < self.min_avg_train_loss:
            self.min_avg_train_loss = avg_train_loss

    def _check_early_stopping_validation(self, avg_valid_loss):
        """
        This function checks whether the training has exploded by verifying if the validation loss
        has more than doubled compared to the minimum validation loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 2 * self.min_valid_loss
        if avg_valid_loss > threshold:
            raise _ModelTrainingRound.EarlyStopNecessary()

        if avg_valid_loss < self.min_valid_loss:
            self.min_valid_loss = avg_valid_loss
