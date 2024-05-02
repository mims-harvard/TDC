import os

from .PINNACLE.pinnacle.model import Pinnacle
from .PINNACLE.pinnacle.generate_input import read_data, get_metapaths

def pinnacle_callback(model_class, base_dir):
    # pre-process
    if not os.path.exists(base_dir+"data"):
        os.mkdir(base_dir+"data")
        os.mkdir(base_dir+"data/networks")
        os.mkdir(base_dir+"data/networks/ppi_edgelists")
    files_to_change = [ [], [], []]
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            fn = os.path.join(root, f)
            if "global_ppi_edgelist" in f:
                blob = os.readlink(fn).split("/")[-1]
                fn = base_dir + "blobs/{}".format(blob)
                files_to_change[0].append(fn)
            elif "ppi_edgelists" in f:
                fnn = f.split("/")[-1]
                blob = os.readlink(fn).split("/")[-1]
                fn = base_dir + "blobs/{}".format(blob)
                files_to_change[1].append((fn, fnn))
            elif "mg_edgelist" in f:
                blob = os.readlink(fn).split("/")[-1]
                fn = base_dir + "blobs/{}".format(blob)
                files_to_change[-1].append(fn)
    fn = files_to_change[0][0]
    os.rename(fn, base_dir + "data/networks/global_ppi_edgelist.txt")
    fn = files_to_change[-1][0]
    os.rename(fn, base_dir + "data/networks/mg_edgelist.txt")
    for fn, fnn in files_to_change[1]:
        os.rename(fn, base_dir + "data/networks/ppi_edgelists/" + fnn)

    # load data
    g_f = base_dir + "data/networks/global_ppi_edgelist.txt"
    ppi_dir = base_dir + "data/networks/ppi_edgelists/"
    mg_f = base_dir + "data/networks/mg_edgelist.txt"
    feat_mat = 2048
    ppi_data, mg_data, edge_attr_dict, mg_mapping, tissue_neighbors, orig_ppi_layers, orig_mg = read_data(g_f, ppi_dir, mg_f, feat_mat)
    ppi_metapaths, mg_metapaths = get_metapaths()
    # initialize pytorch model
    out = model_class(mg_data.x.shape[1], 16, 8, len(ppi_metapaths), len(mg_metapaths), ppi_data, 8, 8, 0.5)
    return out

_MODELS = {
    "PINNACLE": Pinnacle,
}

_CALLBACKS = {
    "PINNACLE": pinnacle_callback, 
}