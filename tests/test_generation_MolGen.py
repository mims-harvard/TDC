from tdc.generation import MolGen


dataloader = MolGen(name = 'zinc')
df = dataloader.get_data()
splits = dataloader.get_split()




dataloader = MolGen(name = 'moses')
df = dataloader.get_data()
splits = dataloader.get_split()




