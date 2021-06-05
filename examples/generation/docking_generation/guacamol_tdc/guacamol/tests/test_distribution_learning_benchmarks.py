from guacamol.distribution_learning_benchmark import ValidityBenchmark, UniquenessBenchmark, NoveltyBenchmark, \
    KLDivBenchmark
from guacamol.assess_distribution_learning import _assess_distribution_learning
from .mock_generator import MockGenerator
import numpy as np
import tempfile
from os.path import join


def test_validity_does_not_penalize_duplicates():
    generator = MockGenerator(['CCC', 'CCC'])
    benchmark = ValidityBenchmark(number_samples=2)

    assert benchmark.assess_model(generator).score == 1.0


def test_validity_score_is_proportion_of_valid_molecules():
    generator = MockGenerator(['CCC', 'CC(CC)C', 'invalidMolecule'])
    benchmark = ValidityBenchmark(number_samples=3)

    assert benchmark.assess_model(generator).score == 2.0 / 3.0


def test_uniqueness_penalizes_duplicates():
    generator = MockGenerator(['CCC', 'CCC', 'CCC'])
    benchmark = UniquenessBenchmark(number_samples=3)

    assert benchmark.assess_model(generator).score == 1.0 / 3.0


def test_uniqueness_penalizes_duplicates_with_different_smiles_strings():
    generator = MockGenerator(['C(O)C', 'CCO', 'OCC'])
    benchmark = UniquenessBenchmark(number_samples=3)

    assert benchmark.assess_model(generator).score == 1.0 / 3.0


def test_uniqueness_does_not_penalize_invalid_molecules():
    generator = MockGenerator(['C(O)C', 'invalid1', 'invalid2', 'CCC', 'NCCN'])
    benchmark = UniquenessBenchmark(number_samples=3)

    assert benchmark.assess_model(generator).score == 1.0


def test_novelty_score_is_zero_if_no_molecule_is_new():
    molecules = ['CCOCC', 'NNNNONNN', 'C=CC=C']
    generator = MockGenerator(molecules)
    benchmark = NoveltyBenchmark(number_samples=3, training_set=molecules)

    assert benchmark.assess_model(generator).score == 0.0


def test_novelty_score_is_one_if_all_molecules_are_new():
    generator = MockGenerator(['CCOCC', 'NNNNONNN', 'C=CC=C'])
    benchmark = NoveltyBenchmark(number_samples=3, training_set=['CO', 'CC'])

    assert benchmark.assess_model(generator).score == 1.0


def test_novelty_score_does_not_penalize_duplicates():
    generator = MockGenerator(['CCOCC', 'O(CC)CC', 'C=CC=C', 'CC'])
    benchmark = NoveltyBenchmark(number_samples=3, training_set=['CO', 'CC'])

    # Gets 2 out of 3: one of the duplicated molecules is ignored, so the sampled molecules are
    # ['CCOCC', 'C=CC=C', 'CC'],  and 'CC' is not novel
    assert benchmark.assess_model(generator).score == 2.0 / 3.0


def test_novelty_score_penalizes_invalid_molecules():
    generator = MockGenerator(['CCOCC', 'invalid1', 'invalid2', 'CCCC', 'CC'])
    benchmark = NoveltyBenchmark(number_samples=3, training_set=['CO', 'CC'])

    assert benchmark.assess_model(generator).score == 2.0 / 3.0


def test_KLdiv_benchmark_same_dist():
    generator = MockGenerator(['CCOCC', 'NNNNONNN', 'C=CC=C'])
    benchmark = KLDivBenchmark(number_samples=3, training_set=['CCOCC', 'NNNNONNN', 'C=CC=C'])
    result = benchmark.assess_model(generator)
    print(result.metadata)
    assert np.isclose(result.score, 1.0, )


def test_KLdiv_benchmark_different_dist():
    generator = MockGenerator(['CCOCC', 'NNNNONNN', 'C=CC=C'])
    benchmark = KLDivBenchmark(number_samples=3, training_set=['FCCOCC', 'CC(CC)CCCCNONNN', 'C=CC=O'])
    result = benchmark.assess_model(generator)
    print(result.metadata)

    assert result.metadata['number_samples'] == 3
    assert result.metadata.get('kl_divs') is not None
    assert result.metadata['kl_divs'].get('BertzCT') > 0
    assert result.metadata['kl_divs'].get('MolLogP', None) > 0
    assert result.metadata['kl_divs'].get('MolWt', None) > 0
    assert result.metadata['kl_divs'].get('TPSA', None) > 0
    assert result.metadata['kl_divs'].get('NumHAcceptors', None) > 0
    assert result.metadata['kl_divs'].get('NumHDonors', None) > 0
    assert result.metadata['kl_divs'].get('NumRotatableBonds', None) > 0
    assert result.score < 1.0


def test_distribution_learning_suite_v1():
    generator = MockGenerator(
        ['CCl', 'CCOCCCl', 'ClCCF', 'CCCOCCOCCCO', 'CF', 'CCOCC', 'CCF', 'CCCOCC', 'NNNNONNN', 'C=CC=C'] * 10)

    mock_chembl = ['FCCOCC', 'C=CC=O', 'CCl', 'CCOCCCl', 'ClCCF', 'CCCOCCOCCCO', 'CF', 'CCOCC',
                   'CCF']

    temp_dir = tempfile.mkdtemp()
    smiles_path = join(temp_dir, 'mock.smiles')
    with open(smiles_path, 'w') as f:
        for i in mock_chembl:
            f.write(f'{i}\n')
        f.close()

    json_path = join(temp_dir, 'output.json')

    _assess_distribution_learning(model=generator,
                                  chembl_training_file=smiles_path,
                                  json_output_file=json_path,
                                  benchmark_version='v1',
                                  number_samples=4)

    with open(json_path, 'r') as f:
        print(f.read())
