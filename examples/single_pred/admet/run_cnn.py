from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *
from tdc.single_pred import ADME

from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')

predictions_all_seeds_CNN = {}
results_all_seeds_CNN = {}

for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for benchmark in group:
        train_val = group.get_auxiliary_train_valid_split(seed, benchmark['name'])
        
        drug_encoding = 'CNN'
        
        train = data_process(X_drug = train_val['train'].Drug.values, y = train_val['train'].Y.values, 
                        drug_encoding = drug_encoding,
                        split_method='no_split')

        val = data_process(X_drug = train_val['valid'].Drug.values, y = train_val['valid'].Y.values, 
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
    predictions_all_seeds_CNN['seed ' + str(seed)] = predictions
    results_all_seeds_CNN['seed ' + str(seed)] = results


def to_submission_format(results):
    df = pd.DataFrame(results)
    def get_metric(x):
        metric = []
        for i in x:
            metric.append(list(i.values())[0])
        return [round(np.mean(metric), 3), round(np.std(metric), 3)]
    return dict(df.apply(get_metric, axis = 1))

print(to_submission_format(results_all_seeds_CNN))