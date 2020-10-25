from tdc.generation import MolGenPaired, Evaluator, ForwardSyn, Retro_Syn

# ### test property evaluator 

# evaluator = Evaluator(name = 'sa')

# s1 = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
# s2 = 'N[C@@H](CCCNC(N)=N)C(O)=O'

# print(evaluator(s1))
# print(evaluator([s1,s2]))


# ### test PairedDataLoader 
paired_dataloader = ForwardSyn(name = 'uspto50k')

pd1 = paired_dataloader.get_data()
# exit()


paired_dataloader2 = Retro_Syn(name = 'uspto50k')
pd2 = paired_dataloader2.get_data()

print(pd1)


print(pd2)

