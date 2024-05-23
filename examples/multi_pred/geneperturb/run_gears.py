from tdc.benchmark_group import geneperturb_group
from gears.utils import filter_pert_in_go
import numpy as np
from gears import PertData, GEARS
import pickle

group = geneperturb_group.GenePerturbGroup()
dataset = 'scperturb_gene_NormanWeissman2019'

train, val = group.get_train_valid_split(dataset = dataset)
test = group.get_test()

set2conditions = {
    'train': train.obs.condition.unique().tolist(),
    'val': val.obs.condition.unique().tolist(),
    'test': test.obs.condition.unique().tolist()
    }

pert_data = PertData('./data') # specific saved folder
pert_data.load(data_path = './data/' + dataset.lower()) # specific dataset name and adata object

train_not_seen = [i for i in set2conditions['train'] if not filter_pert_in_go(i, pert_data.pert_names)]
val_not_seen = [i for i in set2conditions['val'] if not filter_pert_in_go(i, pert_data.pert_names)]
test_not_seen = [i for i in set2conditions['test'] if not filter_pert_in_go(i, pert_data.pert_names)]
print('test perts not in gears',  test_not_seen)
print('train perts not in gears',  train_not_seen)
print('val perts not in gears',  val_not_seen)
set2conditions['train'] = np.setdiff1d(set2conditions['train'], train_not_seen)
set2conditions['val'] = np.setdiff1d(set2conditions['val'], val_not_seen)
set2conditions['test'] = np.setdiff1d(set2conditions['test'], test_not_seen)

pickle.dump(set2conditions, open(dataset + '_set2conditions.pkl', 'wb'))
split_dict_path = dataset + '_set2conditions.pkl'

pert_data.prepare_split(split = 'custom', seed = 1, split_dict_path = split_dict_path)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# set up and train a model
gears_model = GEARS(pert_data, device = 'cuda:1')
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = 20)

# save/load model
gears_model.save_model('gears_' + dataset)