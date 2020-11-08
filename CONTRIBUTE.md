# Thanks for contributing! 

It is very easy to add your dataset to our repository! Here are the steps:

1. Write up your data processing scripts and the output should be the expected format (described in the end of this file) of each task formulation. 
2. Name the dataset file as the `dataset name.XXX`. The name must be in lower case. Ideally, it should be a csv, pkl file.
3. Send [us](kexinhuang@hsph.harvard.edu) your 1) processed data file. 2) data processing scripts (not public, just for us to check if there is an error). 3) a text dataset description, text description, dataset statistics, data split methods, and references. This is for the website.

If you want to contribute a new task or a new data function, please reachout to kexinhuang@hsph.harvard.edu, thanks!

## Expected Format

Single-instance Prediction: 

a file with three columns: 'ID': the ids of the data (e.g. DrugBank ID), X': the standard data format (e.g. SMILES), 'Y': the label values (e.g. IC50). In the case of multiple values available, you can specify the column name as the specific label description (e.g. 'NR-AR'), this way, the user can call "X(name = 'Your new dataset', label_name = 'NR-AR')" to retrieve the data with the wanted label information.

Multi-instance Prediction: 

1. for two entities pair interaction, provide a file with four columns: 'ID1', 'ID2': the IDs of each entity pairs; 'X1', 'X2': the standard data input format for entity 1, 2 (e.g. SMILES); if the label is available, then put it in 'Y' column or if you have multiple labels, then specify the label name as the column name. 

2. for more entities interaction, provide a regular files containing the following columns: for every entity, provides 'Entity_ID' and 'Entity' columns (e.g. 'Drug_ID' is the drugs IDs and 'Drug' is the SMILES string). If there are multiple ones for the same entity type, we suggest to use 'Entity1_ID', 'Entity1' to index it. 

Generation:


If it is a multi-class problem, the label should be integer and then you can create a separate column under name `Map` that specifies the meaning of each label. Then, you can use the `get_label_map` function to retrieve the meaning of each label. 

## Dataset Naming Convention

The overall naming convention follows this order, although not all items should be included:

1. most distinctive info for some tasks (e.g. for ADME, the property; for HTS, the disease)
2. database name
3. the first author's last name of the paper who derives from the database

Please follow existing examples when in doubt.

Thanks a lot and feel free to reachout if you have any question!