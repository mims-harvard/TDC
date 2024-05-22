for dataset in ["scperturb_gene_NormanWeissman2019",
    "scperturb_gene_ReplogleWeissman2022_rpe1",
    "scperturb_gene_ReplogleWeissman2022_k562_essential"]:

    from tdc.benchmark_group import geneperturb_group
    group = geneperturb_group.GenePerturbGroup()
    train, val = group.get_train_valid_split(dataset = dataset)
    test = group.get_test()

    import anndata as ad
    adata = ad.concat([train, val, test])
    adata.obs['cell_type'] = adata.obs['cell_line']
    adata.var['gene_name'] = adata.var.index.values
    from scipy.sparse import csr_matrix
    adata.X = csr_matrix(adata.X)

    from gears import PertData

    pert_data = PertData('./data') # specific saved folder
    pert_data.new_data_process(dataset_name = dataset, adata = adata) # specific dataset name and adata object
