from tdc.generation import Reaction, RetroSyn


paired_dataloader = Reaction(name = 'uspto50k')

pd1 = paired_dataloader.get_data()
split_pd = paired_dataloader.get_split()


paired_dataloader2 = RetroSyn(name = 'uspto50k')
pd2 = paired_dataloader2.get_data()
split_pd = paired_dataloader2.get_split()




