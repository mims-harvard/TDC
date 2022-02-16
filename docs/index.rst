.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


<<<<<<< HEAD
.. figure:: https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/logo.png
=======
.. figure:: https://kexinhuang.com/s/tdc_logo_horizontal.jpg
>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193
    :alt: logo

----


.. image:: https://img.shields.io/badge/website-live-brightgreen
   :target: https://tdcommons.ai
<<<<<<< HEAD
   :alt: Documentation status
=======
   :alt: Website


.. image:: https://badgen.net/badge/icon/github?icon=github&label
   :target: https://github.com/mims-harvard/TDC
   :alt: GitHub
>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193


.. image:: https://badge.fury.io/py/PyTDC.svg
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version


<<<<<<< HEAD
.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://tdcommons.ai
   :alt: Documentation status

=======
>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193
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

<<<<<<< HEAD
=======

>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193
.. image:: https://travis-ci.org/mims-harvard/TDC.svg?branch=master
   :target: https://travis-ci.org/github/mims-harvard/TDC
   :alt: Build Status

<<<<<<< HEAD

=======
>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193
.. image:: https://circleci.com/gh/mims-harvard/TDC.svg?style=svg
   :target: https://app.circleci.com/pipelines/github/mims-harvard/TDC
   :alt: Circle CI

<<<<<<< HEAD
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


=======
.. image:: https://readthedocs.org/projects/tdc/badge/?version=latest
   :target: http://tdc.readthedocs.io/?badge=latest
   :alt: Doc Status

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: license

.. image:: https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ProjectTDC
   :target: https://twitter.com/ProjectTDC
   :alt: Twitter


`Website <https://tdcommons.ai>`_ \| `GitHub <https://github.com/mims-harvard/TDC>`_ \| `NeurIPS 2021 Paper <https://openreview.net/pdf?id=8nvgnORnoWr>`_  \| `Long Paper <https://arxiv.org/abs/2102.09548>`_  \| `Slack <https://join.slack.com/t/pytdc/shared_invite/zt-t59hu2x1-akJ72p2f4fgEBS0LrCFHEw>`_  \| `TDC Mailing List <https://groups.io/g/tdc>`_  


-----


**Therapeutics Data Commons** is an open-science platform with AI/ML-ready datasets and learning tasks for therapeutics, spanning the discovery and development of safe and effective medicines. TDC also provides an ecosystem of tools, libraries, leaderboards, and community resources, including data functions, strategies for systematic model evaluation, meaningful data splits, data processors, and molecule generation oracles. All resources are integrated and accessible via an open Python library.

**Our Vision**: Therapeutics machine learning is an exciting field with incredible opportunities for expansion, innovation, and impact. The collection of curated datasets, learning tasks, and benchmarks in Therapeutics Data Commons (TDC) serves as a meeting point for domain and machine learning scientists. TDC is the first unifying framework to systematically access and evaluate machine learning across the entire range of therapeutics. We envision that TDC can facilitate algorithmic and scientific advances and considerably accelerate machine-learning model development, validation and transition into biomedical and clinical implementation.

----


.. note::
   See the `TDC website <https://tdcommons.ai/>`_ to learn about machine learning for drug development and discovery and get more information on datasets, tasks, leaderboards, data functions, and other features available in Therapeutics Data Commons. 

----

Cite our `NeurIPS 2021 Datasets and Benchmarks Paper: <https://openreview.net/pdf?id=8nvgnORnoWr>`_

.. code-block:: latex

   @article{Huang2021tdc,
    title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
    author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, 
            Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
    journal={Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks},
    year={2021}
  }

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   notebooks

.. toctree::
   :maxdepth: 12
   :caption: API References

   tdc.single_pred
   tdc.multi_pred
   tdc.generation
   tdc.benchmark_group
   tdc.utils
   tdc.chem_utils
   tdc.base_dataset
   tdc.evaluator
   tdc.metadata
   tdc.oracles
>>>>>>> 3ebda027e09694a68c0ac627f44948cab7a80193

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
