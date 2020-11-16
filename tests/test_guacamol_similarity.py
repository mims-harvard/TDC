from tdc.oracle import Oracle

oracle = Oracle(name = 'aripiprazole_similarity')
# oracle = Oracle(name = 'albuterol_similarity')
# oracle = Oracle(name = 'mestranol_similarity')

smiles_lst = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
			  'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
			  'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
			  'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']

print(oracle(smiles_lst))
print(oracle(smiles_lst[0]))
print(oracle(smiles_lst[1]))



