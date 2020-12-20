import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from . import utils


class DataLoader:
    def __init__(self):
        pass

    def get_data(self, format='df'):
        '''
        Arguments:
            df: return pandas DataFrame; if not true, return np.arrays
        returns:
            self.drugs: drug smiles strings np.array
            self.targets: target Amino Acid Sequence np.array
            self.y: inter   action score np.array
        '''
        if format == 'df':
            return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx,
                                 self.entity1_name: self.entity1, 'Y': self.y})
        elif format == 'dict':
            return {self.entity1_name + '_ID': self.entity1_idx,
                    self.entity1_name: self.entity1, 'Y': self.y}
        elif format == 'DeepPurpose':
            return self.entity1, self.y
        elif format == 'sklearn':
            pass
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        print('There are ' + str(len(np.unique(
            self.entity1))) + ' unique ' + self.entity1_name.lower() + 's',
              flush=True, file=sys.stderr)

    def get_split(self, method='random', seed=42,
                  frac=[0.7, 0.1, 0.2]):
        '''
        Arguments:
            method: splitting schemes: random, cold_drug, cold_target
            seed: default 42
            frac: train/val/test split
        '''

        df = self.get_data(format='df')

        if method == 'random':
            return utils.create_fold(df, seed, frac)
        elif method == 'cold_' + self.entity1_name.lower():
            return utils.create_fold_setting_cold(df, seed, frac,
                                                  self.entity1_name)
        else:
            raise AttributeError("Please specify the correct splitting method")

    def label_distribution(self):
        utils.label_dist(self.y, self.name)

    def binarize(self, threshold=None, order='descending'):
        if threshold is None:
            raise AttributeError(
                "Please specify the threshold to binarize the data by "
                "'binarize(threshold = N)'!")

        if (len(np.unique(self.y)) == 2):
            print("The data is already binarized!", flush=True,
                  file=sys.stderr)
        else:
            print("Binariztion using threshold " + str(
                threshold) + ", default, we assume the smaller values are 1 "
                             "and larger ones is 0, you can change the order "
                             "by 'binarize(order = 'ascending')'",
                  flush=True, file=sys.stderr)
            if np.unique(self.y).reshape(-1, ).shape[0] < 2:
                raise AttributeError(
                    "Adjust your threshold, there is only one class.")
            self.y = utils.binarize(self.y, threshold, order)
        return self

    def __len__(self):
        return len(self.get_data(format='df'))

    def convert_to_log(self):
        print('To log space...', flush=True, file=sys.stderr)
        self.y = utils.convert_to_log(self.y)

    def convert_from_log(self):
        print('Convert Back To Original space...', flush=True, file=sys.stderr)
        self.y = utils.convert_back_log(self.y)

    def get_label_meaning(self, output_format='dict'):
        return utils.get_label_map(self.name, self.path, self.target,
                                   file_format=self.file_format,
                                   output_format=output_format)

    def balanced(self, oversample=False, seed=42):

        if len(np.unique(self.y)) > 2:
            raise AttributeError(
                "You should binarize the data first by calling "
                "data.binarize(threshold)",
                flush=True, file=sys.stderr)

        val = self.get_data()

        class_ = val.Y.value_counts().keys().values
        major_class = class_[0]
        minor_class = class_[1]

        if not oversample:
            print(
                " Subsample the majority class is used, if you want to do "
                "oversample the minority class, set 'balanced(oversample = True)'. ",
                flush=True, file=sys.stderr)
            val = pd.concat(
                [val[val.Y == major_class].sample(
                    n=len(val[val.Y == minor_class]), replace=False,
                    random_state=seed), val[val.Y == minor_class]]).sample(
                frac=1,
                replace=False,
                random_state=seed).reset_index(
                drop=True)
        else:
            print(" Oversample of minority class is used. ", flush=True,
                  file=sys.stderr)
            val = pd.concat(
                [val[val.Y == minor_class].sample(
                    n=len(val[val.Y == major_class]), replace=True,
                    random_state=seed), val[val.Y == major_class]]).sample(
                frac=1,
                replace=False,
                random_state=seed).reset_index(
                drop=True)
        return val
