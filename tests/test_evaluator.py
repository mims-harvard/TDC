from tdc.evaluator import Evaluator

evaluator = Evaluator(name = 'diversity')


smiles_lst = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
			  'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
			  'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
			  'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']

print(evaluator(smiles_lst))


groundtruth = [1,1,1,1,0,0,0]
prediction = [0.9, 0.7, 0.3, 0.6, 0.4, 0.4, 0.6]


names = ['prauc', 'f1', 'rocauc', 'precision', 'recall', 'accuracy']
for name in names:
	evaluator = Evaluator(name = name)
	print(name + ":", str(evaluator(prediction, groundtruth))[:5])





