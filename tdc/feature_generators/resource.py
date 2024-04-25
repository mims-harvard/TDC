import inspect
import pandas as pd

from data_feature_generator import DataFeatureGenerator

class ResourceFeatureGenerator(DataFeatureGenerator):

    class decorators:
        
        @classmethod
        def parse_args(cls, func):
            """Sets self.dataset to census_data"""

            def check(*args, **kwargs):
                for idx, arg in enumerate(args):
                    if isinstance(arg, tuple) and arg[0] == "self" and len(arg)==2:
                        if type(arg[1]) not in [list]:
                            args[idx] = args[0][arg[1]] # convert to variable in the DataLoader
                        else:
                            tmp = args[0]
                            for a in arg[1]:
                                tmp = tmp[a]
                            args[idx] = tmp
                        
                for k, v in kwargs.items():
                    if isinstance(v, tuple) and v[0] == "self" and len(v)==2:
                        kwargs[k] = args[0][v[1]] # convert to variable in the DataLoader
                return func(*args, **kwargs)

            return check
    
    @decorators.parse_args
    @classmethod
    def join(cls, dataloader, variable_to_update, left, right, on_left, on_right=None, spec=None):
        spec = spec or "inner"
        assert spec in ["inner", "outer", "left", "right"], "spec must be one of inner, outer, left, right"
        assert isinstance(left, pd.DataFrame)
        assert isinstance(right, pd.DataFrame)
        assert isinstance(on_left, list)
        if on_right:
            assert isinstance(on_right, list)
            assert len(on_right) == len(on_left)
            for idx, col in enumerate(on_right):
                right[on_left[idx]] = on_right[col]  # rename columns to match the ones specified
        if spec == "inner":
            dataloader[variable_to_update]  = pd.merge(left, right, on=on_left, how="inner")
        elif spec == "left":
            raise ValueError("Left joins are not supported.")
        elif spec == "right":
            raise ValueError("Right joins are not supported.")
        elif spec == "outer":
            raise ValueError("Outer joins are not supported.")
        return dataloader[variable_to_update]

    @decorators.parse_args
    @classmethod
    def variable_plug(cls, dataloader, variable_to_update, column_name, value):
        dataloader[variable_to_update][column_name] = value
        return dataloader[variable_to_update]

    @decorators.parse_args
    @classmethod
    def split(cls, dataloader, variable_to_update, dataset, pos_train=None, pos_dev=None, pos_test=None, neg_train=None, neg_dev=None, neg_test=None):
        if pos_train:
            dataloader[variable_to_update]["train_split_pos"] = dataset.iloc[pos_train]
            dataloader[variable_to_update]["train_split_pos"]["Y"] = 1
        if pos_dev:
            dataloader[variable_to_update]["dev_split_pos"] = dataset.iloc[pos_dev]
            dataloader[variable_to_update]["dev_split_pos"]["Y"] = 1
        if pos_test:
            dataloader[variable_to_update]["test_split_pos"] = dataset.iloc[pos_test]
            dataloader[variable_to_update]["test_split_pos"]["Y"] = 1
        if neg_train:
            dataloader[variable_to_update]["train_split_neg"] = dataset.iloc[neg_train]
            dataloader[variable_to_update]["train_split_neg"]["Y"] = 0
        if neg_dev:
            dataloader[variable_to_update]["dev_split_neg"] = dataset.iloc[neg_dev]
            dataloader[variable_to_update]["dev_split_neg"]["Y"] = 0
        if neg_test:
            dataloader[variable_to_update]["test_split_neg"] = dataset.iloc[neg_test]
            dataloader[variable_to_update]["test_split_neg"]["Y"] = 0
        df = pd.DataFrame(columns=dataset.columns)
        if pos_train:
            df = pd.concat(df, dataloader[variable_to_update]["train_split_pos"], axis=0)
        if pos_dev:
            df = pd.concat(df, dataloader[variable_to_update]["dev_split_pos"], axis=0)
        if pos_test:
            df = pd.concat(df, dataloader[variable_to_update]["test_split_pos"], axis=0)
        if neg_train:
            df = pd.concat(df, dataloader[variable_to_update]["train_split_neg"], axis=0)
        if neg_dev:
            df = pd.concat(df, dataloader[variable_to_update]["dev_split_neg"], axis=0)
        if neg_test:
            df = pd.concat(df, dataloader[variable_to_update]["test_split_neg"], axis=0)
        dataloader[variable_to_update]["data"] = df
        return dataloader[variable_to_update]

    @classmethod
    def process_loader(cls, loader, keys, functions, args):
        """Run a series of modifications to a dataloader's variables."""
        functions_dict = {
            name: getattr(cls, name) for name, _ in inspect.getmembers(cls)
        }
        for idx, f in enumerate(functions):
            a = args[idx]
            a["dataloader"] = loader
            a["variable_to_update"] = keys[idx]
            try:
                loader = functions_dict[f](**a)
            except KeyError as e:
                raise ValueError("Unknown function '{}'".format(f),
                                 functions_dict.keys(), e)
        return loader
    
    @decorators.parse_args
    @classmethod
    def concat(cls, dataloader, variable_to_update, ds_list, axis=None):
        """Concatenate two datasets along some axis."""
        dataloader[variable_to_update] = pd.concat([dataloader[x]["data"] for x in ds_list], axis=axis)
        return dataloader[variable_to_update]