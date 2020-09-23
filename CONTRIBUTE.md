# Thanks for contributing! 

It is very easy to add your dataset to Drug Data Loader! Here are the steps:

1. Select from the following dataset types: DTI, DrugProp, ProtFunc, Drugs, Proteins, Targets and go to the correct X_utils.py file in the DrugDataLoader folder.

2. Suppose the dataset we are creating is called DAVIS. Write a function in the end of the X_utils.py file with the function name DAVIS_process and this function should have the arguments and returns corresponding to the task types; e.g. for DTI task, we expect arguments ```name, path, binary, convert_to_log, threshold``` and returns ```drugs, targets, y``` where they are all np.array. You can follow the examples in each file or follow the documentation in the end of this file. 


The default values should be addressed explicitly
The print function should have ```print('XXX', flush = True, file = sys.stderr)```


### Note:

Add a URL in the URLs dictionary in the utils.py; The URL should point to a S3 / Google Drive folder.zip file where the zip folder name is set to be the dataset name e.g. DAVIS.zip. 



3. Write a script in the __init__ function in the DataLoader class

Add an elif clause that returns the correct output (self.Xs), e.g.

For DTI.py, add dataset name 'XXX' 

```python


```

Here are the expected outputs for each dataset type:

DTI: self.drugs: np.array, self.targets: np.array, self.y: np.array

Add to the README

function arguments default values

Unit Test

Run get_data get_split