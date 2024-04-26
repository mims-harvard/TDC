"""
Class encapsulating general data processing functions. Also supports running them in sequence.
Goal is to make it easier to integrate custom datasets not yet in TDC format.
"""
import inspect
from pandas import DataFrame, isnull

from .base import FeatureGenerator


class DataFeatureGenerator(FeatureGenerator):
    """
    Class encapsulating general data processing functions. Also supports running them in sequence.
    Goals are to make it easier to integrate custom datasets not yet in TDC format.
    """

    def __init__(self):
        pass

    @classmethod
    def autofill_identifier(cls, dataset, autofill_column, key_column):
        """Autofill a column based on base column. Assumes one-to-one mapping between both.
        Modifications done in-place.
        
        Args:
            dataset (pandas.DataFrame): dataset to modify.
            autofill_column (str): name of the column to autofill.
            key_column (str): name of the column used for indexing.
            
        Returns:
            pandas.DataFrame: The modified dataset.
        """
        # Create a mapping from key_column to autofill_column
        mapping = dataset.dropna(subset=[autofill_column]).drop_duplicates(
            subset=[key_column]).set_index(
                key_column)[autofill_column].to_dict()

        # Apply the mapping to fill missing values in autofill_column based on key_column values
        dataset[autofill_column] = dataset[key_column].map(mapping)

        return dataset

    @classmethod
    def create_range(cls, dataset, column, keys=None, subs=None):
        """From a column with numeric +/- values, create upper,lower, and expected columns
        Modifies dataset in-place.
        If special keys are provided, corresponding entries are replaced for the numerical value in subs"""

        def helper(entry):
            if isnull(entry):
                return (None, None)
            buffer = ""
            assert isinstance(entry, str), ("Entry must be a string", entry)
            for idx, char in enumerate(entry):
                if char.isdigit() or char == ".":
                    buffer += char
                else:
                    break
            rest = entry[idx + 1:]
            return float(buffer), float(rest)

        keys = [] if keys is None else keys
        subs = [] if subs is None else subs
        assert isinstance(keys, list)
        assert isinstance(subs, list)
        assert len(keys) == len(subs)
        subs_dict = {k: s for k, s in zip(keys, subs)}
        if None not in subs_dict:
            subs_dict[None] = None
        entries = [
            helper(x) if x not in subs_dict else (subs_dict[x], subs_dict[x])
            for x in dataset[column]
        ]
        bounds = [[x1 - x2, x1, x1 +
                   x2] if x1 not in subs_dict else [x1, x1, x1]
                  for x1, x2 in entries]
        df_bounds = DataFrame(bounds, columns=['lower', 'expected', 'upper'])
        dataset["lower"] = df_bounds["lower"]
        dataset["expected"] = df_bounds["expected"]
        dataset["upper"] = df_bounds["upper"]
        return dataset

    @classmethod
    def process_data(cls, dataset, functions, args):
        """Run a series of data transformation on the input dataset

        Args:
            dataset (no specificaton): The original dataset. As different functions can require different data types, this is left unspecified
            functions (list[str]): list of string with function names
            args (list[dict(str: object)]): a list of dicts containing the arguments for each function.
        """
        functions_dict = {
            name: getattr(cls, name) for name, _ in inspect.getmembers(cls)
        }
        df = dataset
        for idx, f in enumerate(functions):
            a = args[idx]
            a["dataset"] = df
            try:
                df = functions_dict[f](**a)
            except KeyError as e:
                raise ValueError("Unknown function '{}'".format(f),
                                 functions_dict.keys(), e)
        return df
