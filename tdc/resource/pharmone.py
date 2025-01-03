from ..utils import general_load
"""
Resource class for the Eve Bio (https://evebio.org/) Pharmone Map.
"""


class PharmoneMap(object):

    def __init__(self, path="./data"):
        self.path = path

    def get_data(self):
        return general_load('evebio_pharmone_v1_detailed_result_table',
                            self.path, '\t')  # Load the Pharmone Map data

    def get_obs_metadata(self):
        return general_load("evebio_pharmone_v1_observed_points_table",
                            self.path, "\t")

    def get_control_data(self):
        return general_load("evebio_pharmone_v1_control_table", self.path,
                            "\t")  # Load the control data

    def get_compound_data(self):
        return general_load("evebio_pharmone_v1_compound_table", self.path,
                            "\t")

    def get_target_data(self):
        return general_load("evebio_pharmone_v1_target_table", self.path, "\t")
