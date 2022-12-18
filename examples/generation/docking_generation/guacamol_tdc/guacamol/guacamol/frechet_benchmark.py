import logging
import os
import pkgutil
import tempfile
import time
from typing import List

import fcd
import numpy as np

from guacamol.distribution_learning_benchmark import (
    DistributionLearningBenchmark,
    DistributionLearningBenchmarkResult,
)
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.data import get_random_subset
from guacamol.utils.sampling_helpers import sample_valid_molecules

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FrechetBenchmark(DistributionLearningBenchmark):
    """
    Calculates the Fréchet ChemNet Distance.

    See http://dx.doi.org/10.1021/acs.jcim.8b00234 for the publication.
    """

    def __init__(
        self,
        training_set: List[str],
        chemnet_model_filename="ChemNet_v0.13_pretrained.h5",
        sample_size=10000,
    ) -> None:
        """
        Args:
            training_set: molecules from the training set
            chemnet_model_filename: name of the file for trained ChemNet model.
                Must be present in the 'fcd' package, since it will be loaded directly from there.
            sample_size: how many molecules to generate the distribution statistics from (both reference data and model)
        """
        self.chemnet_model_filename = chemnet_model_filename
        self.sample_size = sample_size
        super().__init__(
            name="Frechet ChemNet Distance", number_samples=self.sample_size
        )

        self.reference_molecules = get_random_subset(
            training_set, self.sample_size, seed=42
        )

    def assess_model(
        self, model: DistributionMatchingGenerator
    ) -> DistributionLearningBenchmarkResult:
        chemnet = self._load_chemnet()

        start_time = time.time()
        generated_molecules = sample_valid_molecules(
            model=model, number_molecules=self.number_samples
        )
        end_time = time.time()

        if len(generated_molecules) != self.number_samples:
            logger.warning("The model could not generate enough valid molecules.")

        mu_ref, cov_ref = self._calculate_distribution_statistics(
            chemnet, self.reference_molecules
        )
        mu, cov = self._calculate_distribution_statistics(chemnet, generated_molecules)

        FCD = fcd.calculate_frechet_distance(
            mu1=mu_ref, mu2=mu, sigma1=cov_ref, sigma2=cov
        )
        score = np.exp(-0.2 * FCD)

        metadata = {
            "number_reference_molecules": len(self.reference_molecules),
            "number_generated_molecules": len(generated_molecules),
            "FCD": FCD,
        }

        return DistributionLearningBenchmarkResult(
            benchmark_name=self.name,
            score=score,
            sampling_time=end_time - start_time,
            metadata=metadata,
        )

    def _load_chemnet(self):
        """
        Load the ChemNet model from the file specified in the init function.

        This file lives inside a package but to use it, it must always be an actual file.
        The safest way to proceed is therefore:
        1. read the file with pkgutil
        2. save it to a temporary file
        3. load the model from the temporary file
        """
        model_bytes = pkgutil.get_data("fcd", self.chemnet_model_filename)
        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, self.chemnet_model_filename)

        with open(model_path, "wb") as f:
            f.write(model_bytes)

        logger.info(f"Saved ChemNet model to '{model_path}'")

        return fcd.load_ref_model(model_path)

    def _calculate_distribution_statistics(self, model, molecules: List[str]):
        sample_std = fcd.canonical_smiles(molecules)
        gen_mol_act = fcd.get_predictions(model, sample_std)

        mu = np.mean(gen_mol_act, axis=0)
        cov = np.cov(gen_mol_act.T)
        return mu, cov
