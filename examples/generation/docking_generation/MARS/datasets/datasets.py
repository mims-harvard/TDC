import dgl
import torch
from torch.utils import data


class GraphDataset(data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    @staticmethod
    def collate_fn(batch):
        g = dgl.batch(batch)
        return g

class GraphClassificationDataset(GraphDataset):
    def __init__(self, graphs, labels):
        '''
        @params:
            graphs (list): dgl.DGLGraphs
            labels (list): classification targets
        '''
        super().__init__(graphs)
        self.labels = labels

        assert len(graphs) == len(labels)

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index]

    @staticmethod
    def collate_fn(batch):
        graphs, labels = list(zip(*batch))
        g = dgl.batch(graphs)
        labels = torch.tensor(labels).long()
        return g, labels

class ImitationDataset(GraphDataset):
    def __init__(self, graphs, edits):
        '''
        @params:
            graphs: DGLGraphs of molecules
            edits (dict): prediction targets
                act (list): 0/1, (dataset_size,)
                del (list): 0/1, (dataset_size, n_edge)
                add (list): 0/1, (dataset_size, n_node)
                arm (list): 0 ... vocab_size-1, (dataset_size, n_node)
        '''
        super().__init__(graphs)
        self.edits = edits

        assert len(graphs) == len(edits['act'])

    def __getitem__(self, index):
        targ = {}
        for key in self.edits.keys():
            targ[key] = self.edits[key][index]
        return self.graphs[index], targ
    
    def merge_(self, dataset):
        if not isinstance(dataset, ImitationDataset):
            dataset = ImitationDataset.reconstruct(dataset)
        self.graphs += dataset.graphs
        for key in self.edits.keys():
            self.edits[key] += dataset.edits[key]
        
    @staticmethod
    def reconstruct(dataset):
        graphs, edits = ImitationDataset.collate_fn(
            [item for item in dataset], tensorize=False)
        return ImitationDataset(graphs, edits)

    @staticmethod
    def collate_fn(batch, tensorize=True):
        graphs, targs_list = list(zip(*batch))
        if tensorize: graphs = dgl.batch(graphs)

        edits = {}
        for key in targs_list[0].keys():
            edits[key] = [t[key] for t in targs_list]
            if tensorize: edits[key] = \
                torch.tensor(edits[key]).long() # (batch_size,)
        return graphs, edits
    