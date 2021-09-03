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


This site hosts the documentation for **Therapeutics Data Commons (TDC)**, the first unifying framework to systematically access and evaluate machine learning across the entire range of therapeutics.

The collection of curated datasets, learning tasks, and benchmarks in TDC serves as a meeting point for domain and machine learning scientists. We envision that TDC can considerably accelerate machine-learning model development, validation and transition into biomedical and clinical implementation.

----


.. note::
   If you would like to know detailed descriptions about datasets, tasks, leaderboards, functions, please visit our `website <https://tdcommons.ai/>`_.

----

If you find TDC useful, please consider cite us!

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
