from rdkit.DataStructs import cDataStructs
from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *

from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
parser = ArgumentParser(description='ADMET Benchmark Baselines')
parser.add_argument('-m', '--model', default='RDKit2D', type=str, help='model name. select from CNN, RDKit2D, and Morgan.')

args = parser.parse_args()
drug_encoding = args.model

if drug_encoding not in ['RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred']:
    raise ValueError("You have to specify from 'RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred'!")

if drug_encoding == 'RDKit2D':
    drug_encoding = 'rdkit_2d_normalized'
    
if drug_encoding in ['NeuralFP', 'AttentiveFP']:
    drug_encoding = 'DGL_' + drug_encoding

if drug_encoding in ['AttrMasking', 'ContextPred']:
    drug_encoding = 'DGL_GIN_' + drug_encoding

predictions_all_seeds = {}
results_all_seeds = {}

for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for benchmark in group:
        train, valid = group.get_train_valid_split(benchmark = benchmark['name'], split_type = 'default', seed = seed)

        train = data_process(X_drug = train.Drug.values, y = train.Y.values, 
                        drug_encoding = drug_encoding,
                        split_method='no_split')

        val = data_process(X_drug = valid.Drug.values, y = valid.Y.values, 
                        drug_encoding = drug_encoding,
                        split_method='no_split')

        test = data_process(X_drug = benchmark['test'].Drug.values, y = benchmark['test'].Y.values, 
                        drug_encoding = drug_encoding,
                        split_method='no_split')

        config = generate_config(drug_encoding = drug_encoding, 
                                 cls_hidden_dims = [512], 
                                 train_epoch = 50, 
                                 LR = 0.001, 
                                 batch_size = 128,
                                )

        model = models.model_initialize(**config)
        model.train(train, val, test, verbose = False)
        y_pred = model.predict(test)
        predictions[benchmark['name']] = y_pred

    results = group.evaluate(predictions)
    predictions_all_seeds['seed ' + str(seed)] = predictions
    results_all_seeds['seed ' + str(seed)] = results


def to_submission_format(results):
    import pandas as pd
    df = pd.DataFrame(results)
    def get_metric(x):
        metric = []
        for i in x:
            metric.append(list(i.values())[0])
        return [round(np.mean(metric), 3), round(np.std(metric), 3)]
    return dict(df.apply(get_metric, axis = 1))

print(to_submission_format(results_all_seeds)) 
