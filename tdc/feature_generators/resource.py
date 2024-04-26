from functools import wraps
import inspect
import pandas as pd

from .data_feature_generator import DataFeatureGenerator


def parse_args(func):

    def check(self, *args, **kwargs):
        for idx, arg in enumerate(args):
            if isinstance(arg, tuple) and arg[0] == "self" and len(arg) == 2:
                if type(arg[1]) not in [list]:
                    args[idx] = args[0][
                        arg[1]]  # convert to variable in the DataLoader
                else:
                    tmp = args[0]
                    for a in arg[1]:
                        tmp = tmp[a]
                    args[idx] = tmp

        for k, v in kwargs.items():
            if isinstance(v, tuple) and v[0] == "self" and len(v) == 2:
                if type(v[1]) not in [list]:
                    try:
                        kwargs[k] = kwargs["dataloader"][
                            v[1]]  # convert to variable in the DataLoader
                    except:
                        raise Exception(
                            "exception assigning kwargs",
                            k,
                            v,
                            kwargs["dataloader"].keys(),
                        )
                else:
                    tmp = kwargs["dataloader"]
                    for a in v[1]:
                        tmp = tmp[a]
                    kwargs[k] = tmp
        return func(self, *args, **kwargs)

    return check


class ResourceFeatureGenerator(DataFeatureGenerator):

    @parse_args
    def join(self,
             dataloader,
             variable_to_update,
             left,
             right,
             on_left,
             on_right=None,
             spec=None):
        spec = spec or "inner"
        assert spec in ["inner", "outer", "left", "right"
                       ], "spec must be one of inner, outer, left, right"
        assert isinstance(left, pd.DataFrame)
        assert isinstance(right, pd.DataFrame)
        assert isinstance(on_left, list)
        if on_right:
            assert isinstance(on_right, list)
            assert len(on_right) == len(on_left)
            for idx, col in enumerate(on_right):
                right[on_left[idx]] = on_right[
                    col]  # rename columns to match the ones specified
        if spec == "inner":
            dataloader[variable_to_update] = pd.merge(left,
                                                      right,
                                                      on=on_left,
                                                      how="inner")
        elif spec == "left":
            raise ValueError("Left joins are not supported.")
        elif spec == "right":
            raise ValueError("Right joins are not supported.")
        elif spec == "outer":
            raise ValueError("Outer joins are not supported.")
        return dataloader[variable_to_update]

    @parse_args
    def variable_plug(self, dataloader, variable_to_update, column_name, value):
        dataloader[variable_to_update][column_name] = value
        return dataloader[variable_to_update]

    @parse_args
    def split(self,
              dataloader=None,
              variable_to_update=None,
              dataset=None,
              column_name=None,
              pos_train=None,
              pos_dev=None,
              pos_test=None,
              neg_train=None,
              neg_dev=None,
              neg_test=None):
        if pos_train is not None:
            # raise Exception("dataset", dataset.columns, len(dataset), max(pos_train.values))
            dataloader[variable_to_update]["train_split_pos"] = dataset[
                dataset[column_name].isin(pos_train)]
            dataloader[variable_to_update]["train_split_pos"]["Y"] = 1
        if pos_dev is not None:
            dataloader[variable_to_update]["dev_split_pos"] = dataset[
                dataset[column_name].isin(pos_dev)]
            dataloader[variable_to_update]["dev_split_pos"]["Y"] = 1
        if pos_test is not None:
            dataloader[variable_to_update]["test_split_pos"] = dataset[
                dataset[column_name].isin(pos_test)]
            dataloader[variable_to_update]["test_split_pos"]["Y"] = 1
        if neg_train is not None:
            dataloader[variable_to_update]["train_split_neg"] = dataset[
                dataset[column_name].isin(neg_train)]
            dataloader[variable_to_update]["train_split_neg"]["Y"] = 0
        if neg_dev is not None:
            dataloader[variable_to_update]["dev_split_neg"] = dataset[
                dataset[column_name].isin(neg_dev)]
            dataloader[variable_to_update]["dev_split_neg"]["Y"] = 0
        if neg_test is not None:
            dataloader[variable_to_update]["test_split_neg"] = dataset[
                dataset[column_name].isin(neg_test)]
            dataloader[variable_to_update]["test_split_neg"]["Y"] = 0
        df = pd.DataFrame(columns=dataset.columns)
        splits_dict = {
            "train": pd.DataFrame(columns=dataset.columns),
            "dev": pd.DataFrame(columns=dataset.columns),
            "test": pd.DataFrame(columns=dataset.columns)
        }
        if pos_train is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["train_split_pos"]], axis=0)
            splits_dict["train"] = pd.concat([
                splits_dict["train"],
                dataloader[variable_to_update]["train_split_pos"]
            ],
                                             axis=0)
        if pos_dev is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["dev_split_pos"]], axis=0)
            splits_dict["dev"] = pd.concat([
                splits_dict["dev"],
                dataloader[variable_to_update]["dev_split_pos"]
            ],
                                           axis=0)
        if pos_test is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["test_split_pos"]], axis=0)
            splits_dict["test"] = pd.concat([
                splits_dict["test"],
                dataloader[variable_to_update]["test_split_pos"]
            ],
                                            axis=0)
        if neg_train is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["train_split_neg"]], axis=0)
            splits_dict["train"] = pd.concat([
                splits_dict["train"],
                dataloader[variable_to_update]["train_split_neg"]
            ],
                                             axis=0)
        if neg_dev is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["dev_split_neg"]], axis=0)
            splits_dict["dev"] = pd.concat([
                splits_dict["dev"],
                dataloader[variable_to_update]["dev_split_neg"]
            ],
                                           axis=0)
        if neg_test is not None:
            df = pd.concat(
                [df, dataloader[variable_to_update]["test_split_neg"]], axis=0)
            splits_dict["test"] = pd.concat([
                splits_dict["test"],
                dataloader[variable_to_update]["test_split_neg"]
            ],
                                            axis=0)
        dataloader[variable_to_update]["data"] = df
        dataloader['split'] = splits_dict
        return dataloader[variable_to_update]

    def process_loader(self, loader, keys, functions, args):
        """Run a series of modifications to a dataloader's variables."""
        functions_dict = {
            name: getattr(self, name) for name, _ in inspect.getmembers(self)
        }
        for idx, f in enumerate(functions):
            a = args[idx]
            a["dataloader"] = loader
            a["variable_to_update"] = keys[idx]
            # a = (loader, keys[idx])
            try:
                functions_dict[f](**a)
            except KeyError as e:
                raise ValueError("Unknown function '{}'".format(f),
                                 functions_dict.keys(), e)
            except Exception as e:
                print("Error in function {} with args {}".format(f, a))
                raise e
        return loader

    @parse_args
    def concat(self, dataloader, variable_to_update, ds_list, axis=None):
        """Concatenate two datasets along some axis."""
        dataloader[variable_to_update] = pd.concat(
            [dataloader[x]["data"] for x in ds_list], axis=axis)
        return dataloader[variable_to_update]
