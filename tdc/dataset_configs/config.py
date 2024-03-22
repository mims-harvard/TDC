"""General class for representing dataset configs"""


class ConfigBase(dict):
    """Class for representing configs"""
    pass


class DatasetConfig(ConfigBase):
    """Class for representing dataset configs"""

    def __init__(self,
                 dataset_name=None,
                 data_processing_class=None,
                 functions_to_run=None,
                 args_for_functions=None,
                 var_map=None,
                 **kwargs):
        self.dataset_name = dataset_name
        self.data_processing_class = data_processing_class
        self.functions_to_run = functions_to_run
        self.args_for_functions = args_for_functions
        self.var_map = var_map
        self["kwargs"] = kwargs

    @property
    def name(self):
        return self.dataset_name

    @property
    def processing_class(self):
        return self.data_processing_class

    @property
    def functions(self):
        return self.functions_to_run

    @property
    def args(self):
        return self.args_for_functions

    @property
    def functions_and_args(self):
        return zip(self.functions, self.args)

    @property
    def processing_callback(self):
        """Returns a callback for processing a dataset according to this config"""
        return lambda dataset: self.data_processing_class.process_data(
            dataset, self.functions, self.args
        ) if self.data_processing_class is not None else None

    @property
    def tdc_cols_callback(self, dataset_type=None):
        """Add the standard  TDC columns (e.g., 'X1', 'X2', ...) to a given DataFrame."""

        def bi_pred_var_map_helper(dataset):
            assert dataset is not None, "Must provide a dataset."
            dataset["X1"] = dataset[self.var_map["X1"]]
            dataset["X2"] = dataset[self.var_map["X2"]]
            dataset["ID1"] = dataset[self.var_map["ID1"]]
            dataset["ID2"] = dataset[self.var_map["ID2"]]
            return dataset

        dataset_type = dataset_type if dataset_type is not None else "bi_pred_dataset"
        if dataset_type == "bi_pred_dataset" and self.var_map is not None:
            return bi_pred_var_map_helper
        else:
            return None
