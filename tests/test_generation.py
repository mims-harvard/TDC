from tdc.generation import MolGenPaired, Evaluator

logp_evaluator = Evaluator(name = 'logp')

s1 = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
s2 = 'N[C@@H](CCCNC(N)=N)C(O)=O'

print(logp_evaluator(s1))
print(logp_evaluator([s1,s2]))

