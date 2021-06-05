from guacamol.utils.sampling_helpers import sample_valid_molecules, sample_unique_molecules
from .mock_generator import MockGenerator


def test_sample_valid_molecules_for_valid_only():
    generator = MockGenerator(['CCCC', 'CC'])

    mols = sample_valid_molecules(generator, 2)

    assert mols == ['CCCC', 'CC']


def test_sample_valid_molecules_with_invalid_molecules():
    generator = MockGenerator(['invalid', 'invalid', 'invalid', 'CCCC', 'invalid', 'CC'])

    mols = sample_valid_molecules(generator, 2)

    assert mols == ['CCCC', 'CC']


def test_sample_valid_molecules_if_not_enough_valid_generated():
    # does not raise an exception if
    molecules = ['invalid' for _ in range(20)]
    molecules[-1] = 'CC'
    molecules[-2] = 'CN'
    generator = MockGenerator(molecules)

    # samples a max of 9*2 molecules and just does not sample the good ones
    # in this case the list of generated molecules is empty
    assert not sample_valid_molecules(generator, 2, max_tries=9)

    # with a max of 10*2 molecules two valid molecules can be sampled
    generator = MockGenerator(molecules)
    mols = sample_valid_molecules(generator, 2)
    assert mols == ['CN', 'CC']


def test_sample_unique_molecules_for_valid_only():
    generator = MockGenerator(['CCCC', 'CC'])

    mols = sample_unique_molecules(generator, 2)

    assert mols == ['CCCC', 'CC']


def test_sample_unique_molecules_with_invalid_molecules():
    generator = MockGenerator(['invalid1', 'invalid2', 'inv3', 'CCCC', 'CC'])

    mols = sample_unique_molecules(generator, 2)

    assert mols == ['CCCC', 'CC']


def test_sample_unique_molecules_with_duplicate_molecules():
    generator = MockGenerator(['CO', 'C(O)', 'CCCC', 'CC'])

    mols = sample_unique_molecules(generator, 2)

    assert mols == ['CO', 'CCCC']


def test_sample_unique_molecules_if_not_enough_unique_generated():
    # does not raise an exception if
    molecules = ['CO' for _ in range(20)]
    molecules[-1] = 'CC'
    generator = MockGenerator(molecules)

    # samples a max of 9*2 molecules and just does not sample the other molecule
    # in this case the list of generated molecules contains just 'CO'
    mols = sample_unique_molecules(generator, 2, max_tries=9)
    assert mols == ['CO']

    # with a max of 10*2 molecules two valid molecules can be sampled
    generator = MockGenerator(molecules)
    mols = sample_unique_molecules(generator, 2)
    assert mols == ['CO', 'CC']
