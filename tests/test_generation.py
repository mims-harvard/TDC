from tdc.generation import MolGenPaired, Evaluator

evaluator = Evaluator(name = 'drd')

s1 = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
s2 = 'N[C@@H](CCCNC(N)=N)C(O)=O'

print(evaluator(s1))
print(evaluator([s1,s2]))

