import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy
import pickle

import numpy as np
import PIL
import torch
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

parser = argparse.ArgumentParser(description="Domain generalization")
parser.add_argument("--data_dir", type=str, default="./domainbed/data/")
parser.add_argument("--dataset", type=str, default="TdcDtiDg")
parser.add_argument("--algorithm", type=str, default="IRM")
parser.add_argument(
    "--task",
    type=str,
    default="domain_generalization",
    choices=["domain_generalization", "domain_adaptation"],
)
parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
parser.add_argument(
    "--hparams_seed",
    type=int,
    default=0,
    help='Seed for random hparams (0 means "default hparams")',
)
parser.add_argument(
    "--trial_seed",
    type=int,
    default=0,
    help="Trial number (used for seeding split_dataset and " "random_hparams).",
)
parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
parser.add_argument(
    "--steps",
    type=int,
    default=2000,
    help="Number of steps. Default is dataset-dependent.",
)
parser.add_argument(
    "--checkpoint_freq",
    type=int,
    default=None,
    help="Checkpoint every N steps. Default is dataset-dependent.",
)
parser.add_argument("--test_envs", type=int, nargs="+", default=[6, 7, 8])
parser.add_argument("--output_dir", type=str, default="train_output")
parser.add_argument("--holdout_fraction", type=float, default=0.2)
parser.add_argument(
    "--uda_holdout_fraction",
    type=float,
    default=0,
    help="For domain adaptation, % of test to use unlabeled for training.",
)
parser.add_argument("--skip_model_save", action="store_true")
parser.add_argument("--save_model_every_checkpoint", action="store_true")
args = parser.parse_args()

# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
start_step = 0
algorithm_dict = None

os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tPIL: {}".format(PIL.__version__))

print("Args:")
for k, v in sorted(vars(args).items()):
    print("\t{}: {}".format(k, v))

if args.hparams_seed == 0:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
else:
    hparams = hparams_registry.random_hparams(
        args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed)
    )

if args.hparams:
    hparams.update(json.loads(args.hparams))

print("HParams:")
for k, v in sorted(hparams.items()):
    print("\t{}: {}".format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from tdc import Evaluator

evaluator = Evaluator(name="PCC")

print("preparing datasets...")
ENVIRONMENTS = [str(i) for i in list(range(2013, 2022))]
TRAIN_ENV = [str(i) for i in list(range(2013, 2019))]
TEST_ENV = ["2019", "2020", "2021"]
idx2train_env = dict(zip(range(len(TRAIN_ENV)), TRAIN_ENV))
idx2test_env = dict(zip(range(len(TEST_ENV)), TEST_ENV))
dataset = datasets.TdcDtiDg(args.data_dir, args.test_envs, hparams)

in_splits = []
out_splits = []
uda_splits = []

test_set = []

print("constructing in(train)/out(validation) splits with 80%/20% for training dataset")

for env_i, env in enumerate(dataset):
    uda = []

    if env_i in args.test_envs:
        ## testing
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )

        test_set.append((in_, None))

    else:
        ## validation
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )

        in_weights, out_weights, uda_weights = None, None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))


print("creating training data loaders...")
train_loaders = [
    InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams["batch_size"],
        num_workers=dataset.N_WORKERS,
    )
    for i, (env, env_weights) in enumerate(in_splits)
    if i not in args.test_envs
]

print("creating validation data loaders...")
val_loaders = [
    FastDataLoader(dataset=env, batch_size=256, num_workers=dataset.N_WORKERS)
    for env, _ in (out_splits)
]

val_weights = [None for _, weights in (out_splits)]

val_loader_names = ["env_{}".format(idx2train_env[i]) for i in range(len(out_splits))]

print("creating test data loaders...")

eval_loaders = [
    FastDataLoader(dataset=env, batch_size=256, num_workers=dataset.N_WORKERS)
    for env, _ in (test_set)
]

eval_weights = [None for _, weights in (test_set)]

eval_loader_names = ["env_{}".format(idx2test_env[i]) for i in range(len(test_set))]

print("getting model...")

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(
    dataset.input_shape,
    dataset.num_classes,
    len(dataset) - len(args.test_envs),
    hparams,
)

if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)

algorithm.to(device)

## early stopping
algorithm_best = copy.deepcopy(algorithm)
best_pcc = -1

print("prepare for training...")

train_minibatches_iterator = zip(*train_loaders)
checkpoint_vals = collections.defaultdict(lambda: [])

steps_per_epoch = min([len(env) / hparams["batch_size"] for env, _ in in_splits])

n_steps = args.steps or dataset.N_STEPS
checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

print("steps_per_epoch: " + str(steps_per_epoch))
print("n_steps: " + str(n_steps))


def save_checkpoint(filename):
    if args.skip_model_save:
        return
    save_dict = {
        "args": vars(args),
        "model_input_shape": dataset.input_shape,
        "model_num_classes": dataset.num_classes,
        "model_num_domains": len(dataset) - len(args.test_envs),
        "model_hparams": hparams,
        "model_dict": algorithm.cpu().state_dict(),
    }
    torch.save(save_dict, os.path.join(args.output_dir, filename))


print("start training...")

last_results_keys = None

for step in range(start_step, n_steps):
    step_start_time = time.time()

    ## each item is a batch of drugs, targets, ys for env i
    minibatches_device = [
        (d.float().to(device), t.float().to(device), y.float().to(device))
        for d, t, y in next(train_minibatches_iterator)
    ]

    uda_device = None
    step_vals = algorithm.update(minibatches_device, uda_device)
    checkpoint_vals["step_time"].append(time.time() - step_start_time)

    if step % 5 == 0:

        print(
            "Training at Step "
            + str(step + 1)
            + " with loss "
            + str(step_vals["loss"])
            + " and with training pcc "
            + str(step_vals["training_pcc"])
        )

    for key, val in step_vals.items():
        checkpoint_vals[key].append(val)

    if (step % checkpoint_freq == 0) or (step == n_steps - 1):
        results = {
            "step": step,
            "epoch": step / steps_per_epoch,
        }

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        vals = zip(val_loader_names, val_loaders, val_weights)

        pred_all_envs_val = []
        y_all_envs_val = []

        for name, loader, weights in vals:

            pred, y, pcc = misc.pcc(algorithm, loader, weights, device)
            results["val_" + name + "_pcc"] = pcc
            pred_all_envs_val = pred_all_envs_val + pred
            y_all_envs_val = y_all_envs_val + y

        results["val_all_pcc"] = evaluator(y_all_envs_val, pred_all_envs_val)

        if results["val_all_pcc"] > best_pcc:
            algorithm_best = copy.deepcopy(algorithm)
            best_pcc = results["val_all_pcc"]

        print(" --- Validation at Step " + str(step + 1) + " --- ")

        for i, j in results.items():
            print(i + ": " + str(j))

        if step % 500 == 0:
            evals = zip(eval_loader_names, eval_loaders, eval_weights)

            pred_all_envs_test = []
            y_all_envs_test = []

            for name, loader, weights in evals:

                pred, y, pcc = misc.pcc(algorithm, loader, weights, device)
                results[name + "_pcc"] = pcc
                pred_all_envs_test = pred_all_envs_test + pred
                y_all_envs_test = y_all_envs_test + y

            results["test_all_pcc"] = evaluator(y_all_envs_test, pred_all_envs_test)

            print(" --- Testing at Step " + str(step + 1) + " --- ")

            for i, j in results.items():
                print(i + ": " + str(j))


print("Finalized and Final Testing on Early Stopped Model")
evals = zip(eval_loader_names, eval_loaders, eval_weights)

pred_all_envs_test = []
y_all_envs_test = []

for name, loader, weights in evals:

    pred, y, pcc = misc.pcc(algorithm_best, loader, weights, device)
    results[name + "_pcc"] = pcc
    pred_all_envs_test = pred_all_envs_test + pred
    y_all_envs_test = y_all_envs_test + y

results["test_all_pcc"] = evaluator(y_all_envs_test, pred_all_envs_test)

print(" --- Early Stopped Testing at Step " + str(step + 1) + " --- ")

for i, j in results.items():
    print(i + ": " + str(j))

with open(
    args.output_dir + "/" + args.algorithm + "_" + str(args.seed) + "_res.pkl", "wb"
) as f:
    pickle.dump(results, f)
