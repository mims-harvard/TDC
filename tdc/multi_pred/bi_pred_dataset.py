import warnings

warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *


class DataLoader(base_dataset.DataLoader):
    """Add a Class description here.

    Parameters
    ----------
    name :
        Add a variable description here.

    path :
        Add a variable description here.

    label_name :
        Add a variable description here.

    print_stats :
        Add a variable description here.

    dataset_names :
        Add a variable description here.
    """

    def __init__(self, name, path, label_name, print_stats, dataset_names):
        if name.lower() in dataset2target_lists.keys():
            # print_sys("Tip: Use tdc.utils.retrieve_label_name_list(
            # '" + name.lower() + "') to retrieve all available label names.")
            if label_name is None:
                raise ValueError(
                    "Please select a label name. "
                    "You can use tdc.utils.retrieve_label_name_list('" +
                    name.lower() + "') to retrieve all available label names.")

        entity1, entity2, raw_y, entity1_idx, entity2_idx = \
            interaction_dataset_load(name, path, label_name, dataset_names)

        self.name = name
        self.entity1 = entity1
        self.entity2 = entity2
        self.raw_y = raw_y
        self.y = raw_y
        self.entity1_idx = entity1_idx
        self.entity2_idx = entity2_idx
        self.path = path
        self.file_format = 'csv'
        self.label_name = label_name

        self.entity1_name = 'Entity1'
        self.entity2_name = 'Entity2'

        self.two_types = False

    def get_data(self, format='df'):
        """Add a method description here.

        Parameters
        ----------
        format : str, optional (default="df")
            If True, return Pandas DF; return numpy array otherwise.

        Returns
        -------
        drugs : numpy array
            Drug smiles strings.

        targets : numpy array
            Target Amino Acid Sequence.

        y : numpy array
            Interaction score.
        """
        if format == 'df':
            return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx,
                                 self.entity1_name: self.entity1,
                                 self.entity2_name + '_ID': self.entity2_idx,
                                 self.entity2_name: self.entity2, 'Y': self.y})
        elif format == 'DeepPurpose':
            return self.entity1.values, self.entity2.values, self.y.values
        elif format == 'dict':
            return {self.entity1_name + '_ID': self.entity1_idx.values,
                    self.entity1_name: self.entity1.values,
                    self.entity2_name + '_ID': self.entity2_idx.values,
                    self.entity2_name: self.entity2.values, 'Y': self.y.values}
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        """Add a method description here.
        """
        print_sys('--- Dataset Statistics ---')
        try:
            x = np.unique(self.entity1)
        except:
            x = np.unique(self.entity1_idx)

        try:
            y = np.unique(self.entity2)
        except:
            y = np.unique(self.entity2_idx)

        print(str(len(x)) + ' unique ' + self.entity1_name.lower() + 's.',
              flush=True, file=sys.stderr)
        print(str(len(y)) + ' unique ' + self.entity2_name.lower() + 's.',
              flush=True, file=sys.stderr)
        print(str(len(self.y)) + ' ' + self.entity1_name.lower() +
              '-' + self.entity2_name.lower() + ' pairs.',
              flush=True, file=sys.stderr)
        print_sys('--------------------------')
    
    def get_split(self, method='random', seed=42,
                  frac=[0.7, 0.1, 0.2], column_name=None):
        """Add a method description here.

        Parameters
        ----------
        method : splitting schemes
            Splitting schemes: {"random", "cold_drug", "cold_target"}

        seed : int
            Add a variable description here.

        frac : list, optional (default=[0.7, 0.1, 0.2])
            Train/val/test split fractions.

        column_name : str, optional (default=None)
            Add a variable description here.

        Returns
        -------

        """

        df = self.get_data(format='df')

        if method == 'random':
            return create_fold(df, seed, frac)
        elif method == 'cold_' + self.entity1_name.lower():
            return create_fold_setting_cold(df, seed, frac, self.entity1_name)
        elif method == 'cold_' + self.entity2_name.lower():
            return create_fold_setting_cold(df, seed, frac, self.entity2_name)
        elif (column_name is not None) and (column_name in df.columns.values):
            if method == 'cold_split':
                return create_fold_setting_cold(df, seed, frac, column_name)
        elif method == 'combination':
            return create_combination_split(df, seed, frac)
        else:
            raise AttributeError("Please select from random_split, "
                                 "or cold_split. If cold split, "
                                 "please specify the column name!")

    def neg_sample(self, frac=1):
        """Add a method description here.

        Parameters
        ----------
        frac : int or float, optional (default=1)
            Add a variable description here.
        """
        df = NegSample(df=self.get_data(format='df'),
                       column_names=[self.entity1_name + '_ID',
                                     self.entity1_name,
                                     self.entity2_name + '_ID',
                                     self.entity2_name], frac=frac, two_types = self.two_types)
        self.entity1_idx = df[self.entity1_name + '_ID']
        self.entity2_idx = df[self.entity2_name + '_ID']
        self.entity1 = df[self.entity1_name]
        self.entity2 = df[self.entity2_name]
        self.y = df['Y']
        self.raw_y = self.y
        return self

    def to_graph(self, threshold=None, format='edge_list', split=True,
                 frac=[0.7, 0.1, 0.2], seed=42, order='descending'):
        """Add a method description here.

        Parameters
        ----------
        threshold :
            Add a variable description here.

        format :
            Add a variable description here.

        split :
            Add a variable description here.

        frac : list, optional (default=frac=[0.7, 0.1, 0.2])
            Train/val/test split fractions.

        seed : int
            Add a variable description here.

        order :
            Add a variable description here.

        Returns
        -------

        """
        '''
        Arguments:
            format: edge_list / dgl / pyg df object
        '''

        df = self.get_data(format='df')

        if len(np.unique(self.raw_y)) > 2:
            print("The dataset label consists of affinity scores. "
                  "Binarization using threshold " +
                  str(threshold) +
                  " is conducted to construct the positive edges in the network. "
                  "Adjust the threshold by to_graph(threshold = X)",
                  flush=True, file=sys.stderr)
            if threshold is None:
                raise AttributeError(
                    "Please specify the threshold to binarize the data by "
                    "'to_graph(threshold = N)'!")
            df['label_binary'] = label_transform(self.raw_y, True, threshold,
                                                 False, verbose=False,
                                                 order=order)
        else:
            # already binary
            df['label_binary'] = df['Y']

        df[self.entity1_name + '_ID'] = df[self.entity1_name + '_ID'].astype(str)
        df[self.entity2_name + '_ID'] = df[self.entity2_name + '_ID'].astype(str)
        df_pos = df[df.label_binary == 1]
        df_neg = df[df.label_binary == 0]

        return_dict = {}

        pos_edges = df_pos[
            [self.entity1_name + '_ID', self.entity2_name + '_ID']].values
        neg_edges = df_neg[
            [self.entity1_name + '_ID', self.entity2_name + '_ID']].values
        edges = df[
            [self.entity1_name + '_ID', self.entity2_name + '_ID']].values

        if format == 'edge_list':
            return_dict['edge_list'] = pos_edges
            return_dict['neg_edges'] = neg_edges
        elif format == 'dgl':
            try:
                import dgl
            except:
                install("dgl")
                import dgl
            unique_entities = np.unique(pos_edges.T.flatten()).tolist()
            index = list(range(len(unique_entities)))
            dict_ = dict(zip(unique_entities, index))
            edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
            edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])
            return_dict['dgl_graph'] = dgl.DGLGraph((edge_list1, edge_list2))
            return_dict['index_to_entities'] = dict_

        elif format == 'pyg':
            try:
                import torch
                from torch_geometric.data import Data
            except:
                raise ImportError(
                    "Please see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install pytorch geometric!")

            unique_entities = np.unique(pos_edges.T.flatten()).tolist()
            index = list(range(len(unique_entities)))
            dict_ = dict(zip(unique_entities, index))
            edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
            edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])

            edge_index = torch.tensor([edge_list1, edge_list2],
                                      dtype=torch.long)
            x = torch.tensor(np.array(index), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            return_dict['pyg_graph'] = data
            return_dict['index_to_entities'] = dict_

        elif format == 'df':
            return_dict['df'] = df

        if split:
            return_dict['split'] = create_fold(df, seed, frac)

        return return_dict
