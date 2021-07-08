.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/logo.png
    :alt: logo

----


.. image:: https://img.shields.io/badge/website-live-brightgreen
   :target: https://tdcommons.ai
   :alt: Documentation status


.. image:: https://badge.fury.io/py/PyTDC.svg
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://tdcommons.ai
   :alt: Documentation status

.. image:: https://img.shields.io/github/stars/mims-harvard/TDC.svg
   :target: https://github.com/yzhao062/pyod/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/mims-harvard/TDC.svg?color=blue
   :target: https://github.com/yzhao062/pyod/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/pytdc
   :target: https://pepy.tech/project/pytdc
   :alt: Downloads


.. image:: https://pepy.tech/badge/pytdc/month
   :target: https://pepy.tech/project/pytdc
   :alt: Downloads

.. image:: https://travis-ci.org/mims-harvard/TDC.svg?branch=master
   :target: https://travis-ci.org/github/mims-harvard/TDC
   :alt: Build Status


.. image:: https://circleci.com/gh/mims-harvard/TDC.svg?style=svg
   :target: https://app.circleci.com/pipelines/github/mims-harvard/TDC
   :alt: Circle CI

-----


**Therapeutics Data Commons (TDC)** is the first unifying framework to systematically access and evaluate machine learning across the entire range of therapeutics.

The collection of curated datasets, learning tasks, and benchmarks in TDC serves as a meeting point for domain and machine learning scientists. We envision that TDC can considerably accelerate machine-learning model development, validation and transition into biomedical and clinical implementation.

Features
^^^^^^^^

- *Diverse areas of therapeutics development*: TDC covers a wide range of learning tasks, including target discovery, activity screening, efficacy, safety, and manufacturing across biomedical products, including small molecules, antibodies, and vaccines.
- *Ready-to-use datasets*: TDC is minimally dependent on external packages. Any TDC dataset can be retrieved using only 3 lines of code.
- *Data functions*: TDC provides extensive data functions, including data evaluators, meaningful data splits, data processors, and molecule generation oracles.
- *Leaderboards*: TDC provides benchmarks for fair model comparison and a systematic model development and evaluation.
- *Open-source initiative*: TDC is an open-source initiative. If you want to get involved, let us know.


.. figure:: https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/tdc_overview.png
    :alt: overview


Tutorials
^^^^^^^^^

We provide a series of tutorials for you to get started using TDC:

==========================================================================================================================  =========================================================================================================
 Name                                                                                                                       Description
==========================================================================================================================  =========================================================================================================
`101 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_101_Data_Loader.ipynb>`_                                Introduce TDC Data Loaders
`102 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_102_Data_Functions.ipynb>`_                             Introduce TDC Data Functions
`103.1 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_103.1_Datasets_Small_Molecules.ipynb>`_               Walk through TDC Small Molecule Datasets
`103.2 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_103.2_Datasets_Biologics.ipynb>`_                     Walk through TDC Biologics Datasets
`104 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_104_ML_Model_DeepPurpose.ipynb>`_                       Generate 21 ADME ML Predictors with 15 Lines of Code
`105 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_105_Oracle.ipynb>`_                                     Molecule Generation Oracles
`106 <https://github.com/mims-harvard/TDC/blob/master/tutorials/TDC_106_BenchmarkGroup_Submission_Demo.ipynb>`_             Benchmark submission
`DGL <https://github.com/mims-harvard/TDC/blob/master/tutorials/DGL_User_Group_Demo.ipynb>`_                                Demo for DGL GNN User Group Meeting
==========================================================================================================================  =========================================================================================================


TDC Data Loaders
^^^^^^^^^^^^^^^^^

TDC provides a collection of workflows with intuitive, high-level APIs for both beginners and experts to create machine learning models in Python. Building off the modularized "Problem--Learning Task--Data Set" structure (see above) in TDC, we provide a three-layer API to access any learning task and dataset. This hierarchical API design allows us to easily incorporate new tasks and datasets.

For a concrete example, to obtain the HIA dataset from ADME therapeutic learning task in the single-instance prediction problem:

.. code-block:: python


    from tdc.single_pred import ADME
    data = ADME(name = 'HIA_Hou')
    # split into train/val/test with scaffold split methods
    split = data.get_split(method = 'scaffold')
    # get the entire data in the various formats
    data.get_data(format = 'df')


----


.. toctree::
   :maxdepth: 2
   :caption: Contents: Getting Started

   install



.. toctree::
   :maxdepth: 2
   :caption: Contents: Documentation

   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
