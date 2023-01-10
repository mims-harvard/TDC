# Contribution Guide

Thanks for your interest in contributing to TDC!  TDC is a community driven effort. Currently, we are looking for contributions to enrich 1) dataset for existing therapeutics tasks; 2) new meaningful therapeutic task; 3) data functions; 4) leaderboards. Below are the guideline for contributing each of the above categories.

## 0) Development Environment
Before starting to contribute, we suggest you build an environment for TDC, which installs all dependencies, following the below command:

```bash
conda create -n tdc_env python=3.7
conda activate tdc_env

conda install -c conda-forge rdkit
conda install -c conda-forge notebook

# for visualization and testing
pip install seaborn scikit-learn scipy networkx pytest fuzzywuzzy requests tdqm
# for graph transformation
# DGL installation page: https://www.dgl.ai/pages/start.html
# PyG installation page: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# check with 
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import dgl; print(dgl.__version__)"

git clone https://github.com/mims-harvard/TDC.git
cd TDC
```

Before making a pull request to TDC repository, run the following commands to pass all development tests:

```bash
conda activate tdc_env
cd TDC
pytest tdc/test/
```


## 1) New Dataset for Existing Task

- Step 1: Create a **New Issue with label `new-dataset`**. In the issue, describe the new dataset with its justification to add to TDC. TDC team would respond to the issue ASAP. If the new dataset is discussed to be added, proceed to the next step. 

- Step 2: Generate data. 

	- 2.1 First, write up your data processing scripts from the raw data source and stored it (this helps us to keep track what changes are made to the raw data). 
	- 2.2 The data output should be the expected format of TDC for each task. You can download the similar files in the same task and imitate the column names and data structure. Detailed format is also provided in the bottom of this file. **Note**: if you use pandas to process the dataset, remember to do `pd.to_csv('XXX.csv', index = False)`.
	- 2.3 Name the dataset file as the `dataset_name.XXX`. The name must be in lower case and follows the naming convention described in the bottom of this file. Ideally, it should be a csv file. Use pkl file to store tables with values in list/array etc.
	- 2.4 Write text dataset description, task description, dataset statistics, data split methods, and references for the website.
	- **The easiest way is to imitate the existing data files in the same task! Load a dataset and find the data copy in the local `data' folder**

- Step 3: Upload to Harvard Dataverse.

	- 3.1 Create an account in [Harvard Dataverse](https://dataverse.harvard.edu/). 
	- 3.2 For the first time, click "Add a dataset" bottom and filled in the form.
	- 3.3 Upload and publish the generated dataset to the dataverse dataset space. In the published dataset page, scroll to the bottom, click the eye icon that says `View Data' to go into the detailed page of the uploaded data. Then click the metadata tab and in the Download URL section, copy the last several ID numbers, which will be used in Step4.
	- 3.4 If the uploaded file is a csv file. Then, dataverse would process it to a tab file. This is expected. 

- Step 4: Fork the TDC master branch and make changes to the *metadata.py* file.

	- 4.1 Find the `XXX_dataset_names` variable for your task (e.g. for ADME, it is `adme_dataset_names`). Add the dataset name to the bottom of the list.
	- 4.2 In the `name2type` variable, add an entry with the dataset name as the key and the file type as the value. The file types are in the title of the dataverse dataset page. Normally, it should be tab, csv, or pkl files.
	- 4.3 In the `name2id` variable, add an entry with dataset name as the key and the ID in Step 3 as the value.
	- 4.4 **Important**: test your dataset! You can use the same data loader class but change the name input to your dataset name. If there is error, feel free to discuss it in the issue in Step 1.

- Step 5: Create a Pull Request with the changes made above with the following information 1) data processing scripts from 2.1. 2) text description from 2.4.

- Step 6: TDC team would review it and include it in TDC! 


## 2) New Therapeutics Task

- Step 1: Create a **New Issue with label `new-task`**. In the issue, describe the new therapeutics task with its justification to add to TDC. TDC team would respond to the issue ASAP. If the new task is discussed to be added, proceed to the next step. 

- Step 2: During the issue discussion, we decide on the "problem" this new task falls into. Then, go to the corresponding folder in the source code and added a data loader wrapper. You can copy paste the ones in the same "problem". The class name should be the task name and the name of the task should be informative. 

- Step 3: Follow the same steps in adding a new dataset to add the first dataset of this task.

- Step 4: In the `metadata.py` file, create a new variable `XXX_dataset_names` where XXX is the lower case task name. In the `category_names` variable, add the task name (not lower cased). In the `dataset_names` variable, add an entry with key as the task name and value the `XXX_dataset_names` variable. 

**[Example](https://github.com/mims-harvard/TDC/commit/322bddc88acf9617a1fc191d17b8f6b775f7fa8c)**


## 3) New Data Functions

Create a **New Issue with label `new-function`**. Describe the function usage and its importance. TDC team would follow up with you to discuss potential implementation. As data function maybe a bit different depending on different types, we don't provide additional information here.

## 4) New Leaderboard

Create a **New Issue with label `new-leaderboard`**.  Describe a leaderboard design with meaningful task, split and evaluation settings.

## Expected Format

**The easiest way is to imitate the existing data files in the same task! Load a dataset and find the data copy in the local folder**

- Single-instance Prediction: 

A file with three columns: 'ID': the ids of the data (e.g. DrugBank ID), X': the standard data format (e.g. SMILES), 'Y': the label values (e.g. IC50). In the case of multiple values available, you can specify the column name as the specific label description (e.g. 'NR-AR'), this way, the user can call "X(name = 'Your new dataset', label_name = 'NR-AR')" to retrieve the data with the wanted label information. Then, add all the label names as a list in the `tdc.label_name_list.py` file and add the list variable to the `dataset2target_lists` variable.

- Multi-instance Prediction: 

	- for two entities pair interaction, provide a file with four columns: 'ID1', 'ID2': the IDs of each entity pairs; 'X1', 'X2': the standard data input format for entity 1, 2 (e.g. SMILES); if the label is available, then put it in 'Y' column or if you have multiple labels, then specify the label name as the column name. 

	- for more entities interaction, provide a regular files containing the following columns: for every entity, provides 'Entity_ID' and 'Entity' columns (e.g. 'Drug_ID' is the drugs IDs and 'Drug' is the SMILES string). If there are multiple ones for the same entity type, we suggest to use 'Entity1_ID', 'Entity1' to index it. 

- Generation:
	- copy paste the existing column names for each individual task type.

If it is a multi-class problem, the label should be integer and then you can create a separate column under column name `Map` that specifies the meaning of each label. This way, you can use the `utils.get_label_map` function to automatically retrieve the meaning of each label. 

## Dataset Naming Convention

The overall naming convention follows this order, although not all items should be included:

1. most distinctive info for some tasks (e.g. for ADME, the property; for HTS, the disease)
2. database name
3. the first author's last name of the paper who derives from the database

Please follow existing examples when in doubt.

Thanks a lot and feel free to reachout if you have any question!