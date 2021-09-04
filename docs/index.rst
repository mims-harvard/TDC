.. TDC documentation master file, created by
   sphinx-quickstart on Wed Jul  7 12:08:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: https://kexinhuang.com/s/tdc_logo_horizontal.jpg
    :alt: logo

----


.. image:: https://img.shields.io/badge/website-live-brightgreen
   :target: https://tdcommons.ai
   :alt: Website


.. image:: https://badgen.net/badge/icon/github?icon=github&label
   :target: https://github.com/mims-harvard/TDC
   :alt: GitHub


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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
