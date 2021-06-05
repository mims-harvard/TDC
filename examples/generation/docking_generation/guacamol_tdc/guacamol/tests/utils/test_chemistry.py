from guacamol.utils.chemistry import canonicalize, canonicalize_list, is_valid, \
    calculate_internal_pairwise_similarities, calculate_pairwise_similarities, parse_molecular_formula


def test_validity_empty_molecule():
    smiles = ''
    assert not is_valid(smiles)


def test_validity_incorrect_syntax():
    smiles = 'CCCincorrectsyntaxCCC'
    assert not is_valid(smiles)


def test_validity_incorrect_valence():
    smiles = 'CCC(CC)(CC)(=O)CCC'
    assert not is_valid(smiles)


def test_validity_correct_molecules():
    smiles_1 = 'O'
    smiles_2 = 'C'
    smiles_3 = 'CC(ONONOC)CCCc1ccccc1'

    assert is_valid(smiles_1)
    assert is_valid(smiles_2)
    assert is_valid(smiles_3)


def test_isomeric_canonicalisation():
    endiandric_acid = r'OC(=O)[C@H]5C2\C=C/C3[C@@H]5CC4[C@H](C\C=C\C=C\c1ccccc1)[C@@H]2[C@@H]34'

    with_stereocenters = canonicalize(endiandric_acid, include_stereocenters=True)
    without_stereocenters = canonicalize(endiandric_acid, include_stereocenters=False)

    expected_with_stereocenters = 'O=C(O)[C@H]1C2C=CC3[C@@H]1CC1[C@H](C/C=C/C=C/c4ccccc4)[C@@H]2[C@@H]31'
    expected_without_stereocenters = 'O=C(O)C1C2C=CC3C1CC1C(CC=CC=Cc4ccccc4)C2C31'

    assert with_stereocenters == expected_with_stereocenters
    assert without_stereocenters == expected_without_stereocenters


def test_list_canonicalization_removes_none():
    m1 = 'CCC(OCOCO)CC(=O)NCC'
    m2 = 'this.is.not.a.molecule'
    m3 = 'c1ccccc1'
    m4 = 'CC(OCON=N)CC'

    molecules = [m1, m2, m3, m4]
    canonicalized_molecules = canonicalize_list(molecules)

    valid_molecules = [m1, m3, m4]
    expected = [canonicalize(smiles) for smiles in valid_molecules]

    assert canonicalized_molecules == expected


def test_internal_sim():
    molz = ['OCCCF', 'c1cc(F)ccc1', 'c1cnc(CO)cc1', 'FOOF']
    sim = calculate_internal_pairwise_similarities(molz)

    assert sim.shape[0] == 4
    assert sim.shape[1] == 4
    # check elements
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            assert sim[i, j] == sim[j, i]
            if i != j:
                assert sim[i, j] < 1.0
            else:
                assert sim[i, j] == 0


def test_external_sim():
    molz1 = ['OCCCF', 'c1cc(F)ccc1', 'c1cnc(CO)cc1', 'FOOF']
    molz2 = ['c1cc(Cl)ccc1', '[Cr][Ac][K]', '[Ca](F)[Fe]']
    sim = calculate_pairwise_similarities(molz1, molz2)

    assert sim.shape[0] == 4
    assert sim.shape[1] == 3
    # check elements
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            assert sim[i, j] < 1.0


def test_parse_molecular_formula():
    formula = 'C6H9NOF2Cl2Br'
    parsed = parse_molecular_formula(formula)

    expected = [
        ('C', 6),
        ('H', 9),
        ('N', 1),
        ('O', 1),
        ('F', 2),
        ('Cl', 2),
        ('Br', 1)
    ]

    assert parsed == expected
