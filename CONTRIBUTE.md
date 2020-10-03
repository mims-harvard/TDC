# Thanks for contributing! 

It is very easy to add your dataset to our repository! Here are the steps:

1. Write up your data processing scripts and the output should be the expected format (described in the end of this file) of each task formulation. 
2. Name the dataset file as the `dataset name.csv`. Send us your processed data to upload to the data storing server. 
3. Wite description of the dataset in the corresponding section in the README file. 
4. Add the dataset name to the dataset name list in the end of `utils.py`.



## Expected Format

Property Prediction: a csv file with three columns: 'ID': the ids of the data (e.g. DrugBank ID), X': the standard data format (e.g. SMILES), 'Y': the label values (e.g. IC50). In the case of multiple values available, you can specify the column name as the specific label description (e.g. 'NR-AR'), this way, the user can call "X(name = 'Your new dataset', target = 'NR-AR')" to retrieve the data with the wanted label information.

Interaction Prediction: a csv file with four columns: 'ID1', 'ID2': the IDs of each entity pairs; 'X1', 'X2': the standard data input format for entity 1, 2 (e.g. SMILES); if the label is available, then put it in 'Y' column or if you have multiple labels, then specify the label name as the column name. 

Generation:
	Reaction:
	MolGenPaired:
	MolGenDist: