from tdc.generation import Evaluator, Distribution_Dataloader

# ### test property evaluator 

evaluator = Evaluator(name = 'sa')

s1 = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
s2 = 'N[C@@H](CCCNC(N)=N)C(O)=O'

print(evaluator(s1))
print(evaluator([s1,s2]))


### test distribution's dataloader 
zinc_loader = Distribution_Dataloader(name = 'zinc')
print(zinc_loader.get_data())



