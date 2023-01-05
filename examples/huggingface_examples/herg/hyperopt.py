import pandas as pd
from sklearn.metrics import average_precision_score

# TDC package
from tdc.single_pred import Tox

# ML model pacakage
from DeepPurpose import utils as dp_utils, CompoundPred

# hyperparams package
from ray import tune
from ray.tune.search.ax import AxSearch
from ray import air
from ray.air.callbacks.mlflow import MLflowLoggerCallback

def prepare_data(drug_encoding):
    return dp_utils.data_process(
        X_drug=X, y=y, drug_encoding=drug_encoding, random_seed=RANDOM_SEED
    )


def train_fn(hparams, optimal = False):
        
    if drug_encoding == "Morgan":
        config = dp_utils.generate_config(
            drug_encoding=hparams["drug_encoding"],
            train_epoch=int(hparams["train_epochs"]),
            LR=hparams["lr"],
            batch_size=int(hparams["batch_size"]),
            mlp_hidden_dims_drug=[int(hparams["mlp_hidden_dims_drug"])] * int(hparams["mlp_num_layers_drug"])
        )
       
    elif drug_encoding == 'CNN':
        filter_base_size = int(hparams["cnn_drug_filters_base"])
        kernel_base_size = int(hparams["cnn_drug_kernels_base"])

        config = dp_utils.generate_config(
            drug_encoding=hparams["drug_encoding"],
            train_epoch=int(hparams["train_epochs"]),
            LR=hparams["lr"],
            batch_size=int(hparams["batch_size"]),
            cnn_drug_filters=[filter_base_size, filter_base_size*2, filter_base_size*3],
            cnn_drug_kernels=[kernel_base_size, kernel_base_size*2, kernel_base_size*3]
        )
        
    elif drug_encoding == 'DGL_GCN':
        config = dp_utils.generate_config(
            drug_encoding=hparams["drug_encoding"],
            train_epoch=int(hparams["train_epochs"]),
            LR=hparams["lr"],
            batch_size=int(hparams["batch_size"]),
            gnn_hid_dim_drug=int(hparams["gnn_hid_dim_drug"]),
            gnn_num_layers=int(hparams["gnn_num_layers"])
        )
        
    elif drug_encoding == 'DGL_NeuralFP':
        
        config = dp_utils.generate_config(
            drug_encoding=hparams["drug_encoding"],
            train_epoch=int(hparams["train_epochs"]),
            LR=hparams["lr"],
            batch_size=int(hparams["batch_size"]),
            gnn_hid_dim_drug=int(hparams["gnn_hid_dim_drug"]),
            gnn_num_layers=int(hparams["gnn_num_layers"]),
            neuralfp_max_degree=int(hparams["neuralfp_max_degree"]),
            neuralfp_predictor_hid_dim=int(hparams["neuralfp_predictor_hid_dim"])
        )
        
    elif drug_encoding == 'DGL_AttentiveFP':
        config = dp_utils.generate_config(
            drug_encoding=hparams["drug_encoding"],
            train_epoch=int(hparams["train_epochs"]),
            LR=hparams["lr"],
            batch_size=int(hparams["batch_size"]),
            gnn_hid_dim_drug=int(hparams["gnn_hid_dim_drug"]),
            gnn_num_layers=int(hparams["gnn_num_layers"]),
            attentivefp_num_timesteps=int(hparams["attentivefp_num_timesteps"])
        )
        
    config['device'] = 'cuda:0'
    model = CompoundPred.model_initialize(**config)
    model.train(train, val, test, verbose = False)
    
    if optimal:
        model.save_model('_'.join([drug_encoding, data_name, 'optimal']))
        scores = model.predict(test, verbose = False)
        return {'pr_auc': average_precision_score(test.Label.values, scores)}
    else:
        scores = model.predict(val, verbose = False)
        return {'pr_auc': average_precision_score(val.Label.values, scores)}

data_name = "herg_karim"
drug_encoding = "Morgan"

X, y = Tox(name=data_name).get_data(format="DeepPurpose")
train, val, test = prepare_data(drug_encoding)

search_space = {
    "drug_encoding": drug_encoding,
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128, 256, 512]),
    "train_epochs": tune.choice([3,5,10,15,20])
}

if drug_encoding == "Morgan":
    search_space.update({
        "mlp_hidden_dims_drug": tune.choice([32, 64, 128, 256, 512]),
        "mlp_num_layers_drug": tune.choice([1,2,3,4,5,6])
    })
elif drug_encoding == 'CNN':
    search_space.update({
        "cnn_drug_filters_base": tune.choice([8, 16, 32, 64, 128]),
        "cnn_drug_kernels_base": tune.choice([2, 4, 6, 8, 12])
    })
elif drug_encoding == 'DGL_GCN':
    search_space.update({
        "gnn_hid_dim_drug": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        "gnn_num_layers": tune.choice([1,2,3,4])
    })
elif drug_encoding == 'DGL_NeuralFP':
    search_space.update({
       "gnn_hid_dim_drug": tune.choice([8, 16, 32, 64, 128, 256, 512]),
       "gnn_num_layers": tune.choice([1,2,3,4]),
       "neuralfp_max_degree": tune.choice([5, 10, 20]),
       "neuralfp_predictor_hid_dim": tune.choice([8, 16, 32, 64, 128, 256, 512])
    })    
elif drug_encoding == 'DGL_AttentiveFP':
    search_space.update({
       "gnn_hid_dim_drug": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        "gnn_num_layers": tune.choice([1,2,3,4]),
        "attentivefp_num_timesteps": tune.choice([2,3,4,5])
    })

ax_search = AxSearch(metric="pr_auc", mode="max")

tuner = tune.Tuner(
    tune.with_resources(
            tune.with_parameters(train_fn),
            resources={"gpu": 1}
        ),
    tune_config=tune.TuneConfig(search_alg=ax_search, num_samples=100),
    run_config=air.RunConfig(
        name="mlflow",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="./mlruns",
                experiment_name="test",
                save_artifact=True,
            )
        ],
    ),
    param_space=search_space,
)

analysis = tuner.fit()
df = analysis.get_dataframe()
df.to_csv('_'.join([drug_encoding, data_name, 'tune_df']) + '.csv')
optimal_hparam = dict(df.sort_values('pr_auc').iloc[-1])
optimal_hparam = {i[7:] if i[:7] == 'config/' else i: j for i,j in optimal_hparam.items()}
train_fn(optimal_hparam, optimal = True)